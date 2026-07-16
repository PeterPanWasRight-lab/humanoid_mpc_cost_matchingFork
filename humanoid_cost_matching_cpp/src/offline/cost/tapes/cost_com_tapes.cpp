#include "humanoid_cost_matching_cpp/offline/cost/tapes/cost_com_tapes.h"

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/frames.hpp>

#include <ocs2_core/misc/LoadData.h>

#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

#include <stdexcept>

namespace ocs2::humanoid_cost_matching {

static const std::string kTaskInfoPath =
    "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task_cost_matching.info";

// helper: read ptree value icp_cost_weights.icpErrorWeight (single scalar) and expand to 2D weights
static inline ocs2::vector_t loadIcpWeights2_(bool verbose) {
  boost::property_tree::ptree pt;
  boost::property_tree::read_info(kTaskInfoPath, pt);
  ocs2::scalar_t icpErrorWeight = 0.0;
  loadData::loadPtreeValue(pt, icpErrorWeight, "icp_cost_weights.icpErrorWeight", verbose);
  ocs2::vector_t weights(2);
  weights << icpErrorWeight, icpErrorWeight;
  return weights;
}

// ---------------- theta tape (variables=theta_com, params=x) ----------------

ComStageThetaAD::ComStageThetaAD(size_t nx,
                                 const ocs2::PinocchioInterfaceCppAd& pinocchioInterfaceCppAd,
                                 const ocs2::humanoid::CentroidalMpcRobotModel<ocs2::ad_scalar_t>& mpcRobotModelAd,
                                 const std::string& modelName,
                                 const std::string& modelFolder,
                                 bool recompile,
                                 bool verbose)
    : nx_(nx), pinCppAd_(pinocchioInterfaceCppAd) {
  robotAdPtr_.reset(mpcRobotModelAd.clone());
  if (!robotAdPtr_) throw std::runtime_error("[ComStageThetaAD] mpcRobotModelAd.clone() returned null");

  using ad_scalar_t = ocs2::ad_scalar_t;
  using ad_vector_t = ocs2::ad_vector_t;
  using ad_vector2_t = Eigen::Matrix<ad_scalar_t, 2, 1>;

  const size_t varDim = kComErrDim;  // theta_com dim (=2)
  const size_t parDim = nx_;         // x only

  // load nominal weights w0 from task.info (2 dims)
  const ocs2::vector_t w0 = loadIcpWeights2_(verbose);
  if (w0.size() != kComErrDim) {
    throw std::runtime_error("[ComStageThetaAD] icp weights size mismatch");
  }

  auto fun = [this, w0](const ad_vector_t& th, const ad_vector_t& x_param, ad_vector_t& y) {
    y.resize(1);

    const ad_vector_t state = x_param;  // (nx)

    const auto& model = pinCppAd_.getModel();
    auto& data = pinCppAd_.getData();

    const ad_vector_t q = robotAdPtr_->getGeneralizedCoordinates(state);

    pinocchio::centerOfMass(model, data, q, false);
    const ad_vector2_t com = data.com[0].template head<2>();

    pinocchio::updateFramePlacements(model, data);

    const auto contactPositions = ocs2::humanoid::getContactPositions<ad_scalar_t>(pinCppAd_, *robotAdPtr_);
    const ad_vector2_t desiredCOMPosition = (contactPositions[0] + contactPositions[1]).template head<2>() / ad_scalar_t(2.0);

    const ad_vector2_t capturePoint = com;

    ad_vector_t e(kComErrDim);
    e(0) = desiredCOMPosition(0) - capturePoint(0);
    e(1) = desiredCOMPosition(1) - capturePoint(1);

    // cost = 0.5 * sum_i ( w0_i * theta_i * e_i^2 )
    ad_scalar_t cost = ad_scalar_t(0.0);
    for (int i = 0; i < kComErrDim; ++i) {
      cost += ad_scalar_t(0.5) * ad_scalar_t(w0(i)) * th(i) * e(i) * e(i);
    }
    y(0) = cost;
  };

  ad_.reset(new ocs2::CppAdInterface(fun, varDim, parDim, modelName, modelFolder));

  if (recompile)
    ad_->createModels(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
  else
    ad_->loadModelsIfAvailable(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
}

ocs2::scalar_t ComStageThetaAD::value(const ocs2::vector_t& theta_com, const ocs2::vector_t& x_params) const {
  return ad_->getFunctionValue(theta_com, x_params)(0);
}

ocs2::vector_t ComStageThetaAD::grad_theta(const ocs2::vector_t& theta_com, const ocs2::vector_t& x_params) const {
  const ocs2::matrix_t J = ad_->getJacobian(theta_com, x_params);  // 1 x kComErrDim
  return J.row(0).transpose();
}

// ---------------- XU tape (variables=[x;u], params=theta_com) ----------------

ComStageXUAD::ComStageXUAD(size_t nx,
                           size_t nu,
                           const ocs2::PinocchioInterfaceCppAd& pinocchioInterfaceCppAd,
                           const ocs2::humanoid::CentroidalMpcRobotModel<ocs2::ad_scalar_t>& mpcRobotModelAd,
                           const std::string& modelName,
                           const std::string& modelFolder,
                           bool recompile,
                           bool verbose)
    : nx_(nx), nu_(nu), pinCppAd_(pinocchioInterfaceCppAd) {
  robotAdPtr_.reset(mpcRobotModelAd.clone());
  if (!robotAdPtr_) throw std::runtime_error("[ComStageXUAD] mpcRobotModelAd.clone() returned null");

  using ad_scalar_t = ocs2::ad_scalar_t;
  using ad_vector_t = ocs2::ad_vector_t;
  using ad_vector2_t = Eigen::Matrix<ad_scalar_t, 2, 1>;

  const size_t varDim = nx_ + nu_;   // [x;u] (u unused)
  const size_t parDim = kComErrDim;  // theta_com only

  const ocs2::vector_t w0 = loadIcpWeights2_(verbose);
  if (w0.size() != kComErrDim) {
    throw std::runtime_error("[ComStageXUAD] icp weights size mismatch");
  }

  auto fun = [this, w0](const ad_vector_t& xu, const ad_vector_t& th_param, ad_vector_t& y) {
    y.resize(1);

    const ad_vector_t state = xu.head(nx_);
    const ad_vector_t th = th_param;  // size kComErrDim

    const auto& model = pinCppAd_.getModel();
    auto& data = pinCppAd_.getData();

    const ad_vector_t q = robotAdPtr_->getGeneralizedCoordinates(state);

    pinocchio::centerOfMass(model, data, q, false);
    const ad_vector2_t com = data.com[0].template head<2>();

    pinocchio::updateFramePlacements(model, data);

    const auto contactPositions = ocs2::humanoid::getContactPositions<ad_scalar_t>(pinCppAd_, *robotAdPtr_);
    const ad_vector2_t desiredCOMPosition = (contactPositions[0] + contactPositions[1]).template head<2>() / ad_scalar_t(2.0);

    const ad_vector2_t capturePoint = com;

    ad_vector_t e(kComErrDim);
    e(0) = desiredCOMPosition(0) - capturePoint(0);
    e(1) = desiredCOMPosition(1) - capturePoint(1);

    ad_scalar_t cost = ad_scalar_t(0.0);
    for (int i = 0; i < kComErrDim; ++i) {
      cost += ad_scalar_t(0.5) * ad_scalar_t(w0(i)) * th(i) * e(i) * e(i);
    }
    y(0) = cost;
  };

  ad_.reset(new ocs2::CppAdInterface(fun, varDim, parDim, modelName, modelFolder));

  if (recompile)
    ad_->createModels(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
  else
    ad_->loadModelsIfAvailable(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
}

ocs2::scalar_t ComStageXUAD::value(const ocs2::vector_t& xu, const ocs2::vector_t& theta_params) const {
  return ad_->getFunctionValue(xu, theta_params)(0);
}

ocs2::vector_t ComStageXUAD::grad_xu(const ocs2::vector_t& xu, const ocs2::vector_t& theta_params) const {
  const ocs2::matrix_t J = ad_->getJacobian(xu, theta_params);  // 1 x (nx+nu)
  return J.row(0).transpose();
}

}  // namespace ocs2::humanoid_cost_matching
