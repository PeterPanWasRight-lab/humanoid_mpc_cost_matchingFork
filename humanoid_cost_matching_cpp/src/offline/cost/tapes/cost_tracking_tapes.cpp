#include "humanoid_cost_matching_cpp/offline/cost/tapes/cost_tracking_tapes.h"

#include <ocs2_core/misc/LoadData.h>
#include <stdexcept>

namespace ocs2::humanoid_cost_matching {

// hardcode task.info path
static const std::string kTaskInfoPath =
    "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task_cost_matching.info";

// ---------- stage theta tape: variables=theta_qr, params=[t|x|u|xref|uref] ----------

TrackingStageThetaAD::TrackingStageThetaAD(
    size_t nx, size_t nu, const std::string& modelName, const std::string& modelFolder, bool recompile, bool verbose)
    : nx_(nx), nu_(nu) {
  using ad_scalar_t = ocs2::ad_scalar_t;
  using ad_vector_t = ocs2::ad_vector_t;

  const size_t varDim = nx_ + nu_;
  const size_t parDim = 1 + nx_ + nu_ + nx_ + nu_;

  // Load nominal Q/R from task.info
  ocs2::matrix_t Q(nx_, nx_);
  ocs2::matrix_t R(nu_, nu_);
  loadData::loadEigenMatrix(kTaskInfoPath, "Q", Q);
  loadData::loadEigenMatrix(kTaskInfoPath, "R", R);

  const ocs2::vector_t Qdiag = Q.diagonal();
  const ocs2::vector_t Rdiag = R.diagonal();

  auto fun = [=](const ad_vector_t& th, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(1);

    const ad_vector_t x = p.segment(1, nx_);
    const ad_vector_t u = p.segment(1 + nx_, nu_);
    const ad_vector_t xref = p.segment(1 + nx_ + nu_, nx_);
    const ad_vector_t uref = p.segment(1 + nx_ + nu_ + nx_, nu_);

    const ad_vector_t tq = th.head(nx_);
    const ad_vector_t tr = th.tail(nu_);

    const ad_vector_t ex = x - xref;
    const ad_vector_t eu = u - uref;

    ad_scalar_t cost = ad_scalar_t(0.0);

    // Match OCS2: 0.5 * e^T Q e + 0.5 * e^T R e
    // Here: Q,R are nominal; theta is elementwise multiplier on diagonal.
    for (int i = 0; i < ex.size(); ++i) {
      const ad_scalar_t w = tq(i) * ad_scalar_t(Qdiag(i));
      cost += ad_scalar_t(0.5) * w * ex(i) * ex(i);
    }
    for (int j = 0; j < eu.size(); ++j) {
      const ad_scalar_t w = tr(j) * ad_scalar_t(Rdiag(j));
      cost += ad_scalar_t(0.5) * w * eu(j) * eu(j);
    }

    y(0) = cost;
  };

  ad_.reset(new ocs2::CppAdInterface(fun, varDim, parDim, modelName, modelFolder));

  if (recompile)
    ad_->createModels(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
  else
    ad_->loadModelsIfAvailable(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
}

ocs2::scalar_t TrackingStageThetaAD::value(const ocs2::vector_t& theta_qr, const ocs2::vector_t& params) const {
  return ad_->getFunctionValue(theta_qr, params)(0);
}

ocs2::vector_t TrackingStageThetaAD::grad_theta(const ocs2::vector_t& theta_qr, const ocs2::vector_t& params) const {
  const ocs2::matrix_t J = ad_->getJacobian(theta_qr, params);  // 1 x (nx+nu)
  return J.row(0).transpose();
}

// ---------- terminal theta tape: variables=theta_qf, params=[t|x|xref] ----------

TrackingTerminalThetaAD::TrackingTerminalThetaAD(
    size_t nx, const std::string& modelName, const std::string& modelFolder, bool recompile, bool verbose)
    : nx_(nx) {
  using ad_scalar_t = ocs2::ad_scalar_t;
  using ad_vector_t = ocs2::ad_vector_t;

  const size_t varDim = nx_;
  const size_t parDim = 1 + nx_ + nx_;

  // Load nominal Q_final + terminalCostScaling
  ocs2::matrix_t Qf(nx_, nx_);
  loadData::loadEigenMatrix(kTaskInfoPath, "Q_final", Qf);
  const ocs2::vector_t Qfdiag = Qf.diagonal();

  ocs2::scalar_t terminalCostScaling = 1.0;
  loadData::loadCppDataType<ocs2::scalar_t>(kTaskInfoPath, "terminalCostScaling", terminalCostScaling);

  auto fun = [=](const ad_vector_t& th, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(1);

    const ad_vector_t x = p.segment(1, nx_);
    const ad_vector_t xref = p.segment(1 + nx_, nx_);
    const ad_vector_t ex = x - xref;

    ad_scalar_t cost = ad_scalar_t(0.0);
    for (int i = 0; i < ex.size(); ++i) {
      const ad_scalar_t w = th(i) * ad_scalar_t(Qfdiag(i));
      cost += ad_scalar_t(0.5) * w * ex(i) * ex(i);
    }

    // Match OCS2 terminal scaling
    cost *= ad_scalar_t(terminalCostScaling);

    y(0) = cost;
  };

  ad_.reset(new ocs2::CppAdInterface(fun, varDim, parDim, modelName, modelFolder));

  if (recompile)
    ad_->createModels(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
  else
    ad_->loadModelsIfAvailable(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
}

ocs2::scalar_t TrackingTerminalThetaAD::value(const ocs2::vector_t& theta_qf, const ocs2::vector_t& params) const {
  return ad_->getFunctionValue(theta_qf, params)(0);
}

ocs2::vector_t TrackingTerminalThetaAD::grad_theta(const ocs2::vector_t& theta_qf, const ocs2::vector_t& params) const {
  const ocs2::matrix_t J = ad_->getJacobian(theta_qf, params);  // 1 x nx
  return J.row(0).transpose();
}

// ---------------- stage XU tape: variables=[x;u], params=[t|xNom|uNom|theta_q|theta_r] ----------------

TrackingStageXUAD::TrackingStageXUAD(
    size_t nx, size_t nu, const std::string& modelName, const std::string& modelFolder, bool recompile, bool verbose)
    : nx_(nx), nu_(nu) {
  using ad_scalar_t = ocs2::ad_scalar_t;
  using ad_vector_t = ocs2::ad_vector_t;

  const size_t varDim = nx_ + nu_;
  const size_t parDim = 1 + nx_ + nu_ + nx_ + nu_;

  // Load nominal Q/R
  ocs2::matrix_t Q(nx_, nx_);
  ocs2::matrix_t R(nu_, nu_);
  loadData::loadEigenMatrix(kTaskInfoPath, "Q", Q);
  loadData::loadEigenMatrix(kTaskInfoPath, "R", R);
  const ocs2::vector_t Qdiag = Q.diagonal();
  const ocs2::vector_t Rdiag = R.diagonal();

  auto fun = [=](const ad_vector_t& z, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(1);

    const ad_vector_t x = z.head(nx_);
    const ad_vector_t u = z.tail(nu_);

    const ad_vector_t xNom = p.segment(1, nx_);
    const ad_vector_t uNom = p.segment(1 + nx_, nu_);

    const ad_vector_t tq = p.segment(1 + nx_ + nu_, nx_);
    const ad_vector_t tr = p.segment(1 + nx_ + nu_ + nx_, nu_);

    const ad_vector_t ex = x - xNom;
    const ad_vector_t eu = u - uNom;

    ad_scalar_t cost = ad_scalar_t(0.0);
    for (int i = 0; i < ex.size(); ++i) {
      const ad_scalar_t w = tq(i) * ad_scalar_t(Qdiag(i));
      cost += ad_scalar_t(0.5) * w * ex(i) * ex(i);
    }
    for (int j = 0; j < eu.size(); ++j) {
      const ad_scalar_t w = tr(j) * ad_scalar_t(Rdiag(j));
      cost += ad_scalar_t(0.5) * w * eu(j) * eu(j);
    }

    y(0) = cost;
  };

  ad_.reset(new ocs2::CppAdInterface(fun, varDim, parDim, modelName, modelFolder));

  if (recompile)
    ad_->createModels(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
  else
    ad_->loadModelsIfAvailable(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
}

ocs2::scalar_t TrackingStageXUAD::value(const ocs2::vector_t& xu, const ocs2::vector_t& params) const {
  return ad_->getFunctionValue(xu, params)(0);
}

ocs2::vector_t TrackingStageXUAD::grad_xu(const ocs2::vector_t& xu, const ocs2::vector_t& params) const {
  const ocs2::matrix_t J = ad_->getJacobian(xu, params);  // 1 x (nx+nu)
  return J.row(0).transpose();
}

// ---------------- terminal X tape: variables=x, params=[t|xNom|theta_qf] ----------------

TrackingTerminalXAD::TrackingTerminalXAD(
    size_t nx, const std::string& modelName, const std::string& modelFolder, bool recompile, bool verbose)
    : nx_(nx) {
  using ad_scalar_t = ocs2::ad_scalar_t;
  using ad_vector_t = ocs2::ad_vector_t;

  const size_t varDim = nx_;
  const size_t parDim = 1 + nx_ + nx_;

  // Load nominal Q_final + terminalCostScaling
  ocs2::matrix_t Qf(nx_, nx_);
  loadData::loadEigenMatrix(kTaskInfoPath, "Q_final", Qf);
  const ocs2::vector_t Qfdiag = Qf.diagonal();

  ocs2::scalar_t terminalCostScaling = 1.0;
  loadData::loadCppDataType<ocs2::scalar_t>(kTaskInfoPath, "terminalCostScaling", terminalCostScaling);

  auto fun = [=](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(1);

    const ad_vector_t xNom = p.segment(1, nx_);
    const ad_vector_t tqf = p.segment(1 + nx_, nx_);

    const ad_vector_t ex = x - xNom;

    ad_scalar_t cost = ad_scalar_t(0.0);
    for (int i = 0; i < ex.size(); ++i) {
      const ad_scalar_t w = tqf(i) * ad_scalar_t(Qfdiag(i));
      cost += ad_scalar_t(0.5) * w * ex(i) * ex(i);
    }

    cost *= ad_scalar_t(terminalCostScaling);

    y(0) = cost;
  };

  ad_.reset(new ocs2::CppAdInterface(fun, varDim, parDim, modelName, modelFolder));

  if (recompile)
    ad_->createModels(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
  else
    ad_->loadModelsIfAvailable(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
}

ocs2::scalar_t TrackingTerminalXAD::value(const ocs2::vector_t& x, const ocs2::vector_t& params) const {
  return ad_->getFunctionValue(x, params)(0);
}

ocs2::vector_t TrackingTerminalXAD::grad_x(const ocs2::vector_t& x, const ocs2::vector_t& params) const {
  const ocs2::matrix_t J = ad_->getJacobian(x, params);  // 1 x nx
  return J.row(0).transpose();
}

}  // namespace ocs2::humanoid_cost_matching
