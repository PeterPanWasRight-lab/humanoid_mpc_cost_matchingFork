#include "humanoid_cost_matching_cpp/offline/cost/tapes/cost_torque_tapes.h"

#include <ocs2_core/misc/LoadData.h>
#include <pinocchio/algorithm/frames.hpp>

#include <stdexcept>

namespace ocs2::humanoid_cost_matching {

using ad_scalar_t = ocs2::ad_scalar_t;
using ad_vector_t = ocs2::ad_vector_t;
using ad_matrix_t = ocs2::ad_matrix_t;

static const std::string kTaskInfoPath =
    "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task_cost_matching.info";

// helper: load weights vector (m x 1) from task.info section.
static ocs2::vector_t loadTorqueWeights(const std::string& fieldPrefix, size_t m) {
  ocs2::vector_t w(static_cast<long>(m));
  // loadData::loadEigenMatrix(filename, fieldname + "weights", weights);
  loadData::loadEigenMatrix(kTaskInfoPath, fieldPrefix + "weights", w);
  if (static_cast<size_t>(w.size()) != m) {
    throw std::runtime_error("[TorqueTapes] loaded weights size mismatch. got=" + std::to_string(w.size()) +
                             " expected=" + std::to_string(m) + " for prefix=" + fieldPrefix);
  }
  return w;
}

static inline ad_vector_t computeTauExtActive(const ad_vector_t& state,
                                              const ad_vector_t& input,
                                              ocs2::PinocchioInterfaceCppAd& pinCppAd,
                                              ocs2::humanoid::MpcRobotModelBase<ad_scalar_t>& robotAd,
                                              pinocchio::FrameIndex frameID,
                                              size_t contactIndex,
                                              const std::vector<size_t>& tauActiveIndices) {
  const auto& model = pinCppAd.getModel();
  auto& data = pinCppAd.getData();

  // Online code does NOT call forwardKinematics here; it only calls computeFrameJacobian(q)
  const ad_vector_t q = robotAd.getGeneralizedCoordinates(state);

  ad_matrix_t J_ee = ad_matrix_t::Zero(6, robotAd.getGenCoordinatesDim());
  pinocchio::computeFrameJacobian(model, data, q, frameID, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, J_ee);

  const ad_vector_t tauExt = J_ee.transpose() * robotAd.getContactWrench(input, contactIndex);

  ad_vector_t tauAct(static_cast<long>(tauActiveIndices.size()));
  for (size_t i = 0; i < tauActiveIndices.size(); ++i) {
    tauAct(static_cast<long>(i)) = tauExt(static_cast<long>(tauActiveIndices[i]));
  }
  return tauAct;
}

// ---------------- Theta tape ----------------

TorqueStageThetaAD::TorqueStageThetaAD(size_t nx,
                                       size_t nu,
                                       size_t contactIndex,
                                       pinocchio::FrameIndex frameID,
                                       std::vector<size_t> tauActiveIndices,
                                       const ocs2::PinocchioInterfaceCppAd& pinocchioCppAd,
                                       const ocs2::humanoid::MpcRobotModelBase<ad_scalar_t>& robotModelAd,
                                       const std::string& modelName,
                                       const std::string& modelFolder,
                                       bool recompile,
                                       bool verbose)
    : nx_(nx),
      nu_(nu),
      contactIndex_(contactIndex),
      frameID_(frameID),
      tauActiveIndices_(std::move(tauActiveIndices)),
      pinCppAd_(pinocchioCppAd) {
  if (tauActiveIndices_.empty()) throw std::runtime_error("[TorqueStageThetaAD] tauActiveIndices empty");

  robotAdPtr_.reset(robotModelAd.clone());
  if (!robotAdPtr_) throw std::runtime_error("[TorqueStageThetaAD] robotModelAd.clone() returned null");

  const size_t m = tauActiveIndices_.size();
  const size_t varDim = m;                  // theta (learned per-active-joint multiplier)
  const size_t parDim = 1 + nx_ + nu_ + 1;  // [t|x|u|impact]

  const std::string prefix = (contactIndex_ == 0) ? "left_leg_torque_cost." : "right_leg_torque_cost.";
  const ocs2::vector_t w0 = loadTorqueWeights(prefix, m);  // (m)

  auto fun = [this, m, w0](const ad_vector_t& th, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(1);

    const ad_vector_t x = p.segment(1, static_cast<long>(nx_));
    const ad_vector_t u = p.segment(1 + static_cast<long>(nx_), static_cast<long>(nu_));
    const ad_scalar_t impact = p(1 + static_cast<long>(nx_) + static_cast<long>(nu_));

    // online: midSwingScaler = 1 - impactProximityScaler
    const ad_scalar_t midSwing = ad_scalar_t(1.0) - impact;
    const ad_scalar_t ms2 = midSwing * midSwing;

    const ad_vector_t tauAct = computeTauExtActive(x, u, pinCppAd_, *robotAdPtr_, frameID_, contactIndex_, tauActiveIndices_);

    // Match online expanded form:
    //   cost = 0.5 * ms^2 * sum_i ( w0_i * theta_i * tau_i^2 )
    ad_scalar_t acc = ad_scalar_t(0.0);
    for (size_t i = 0; i < m; ++i) {
      const ad_scalar_t ti = tauAct(static_cast<long>(i));
      acc += th(static_cast<long>(i)) * ad_scalar_t(w0(static_cast<long>(i))) * ti * ti;
    }

    y(0) = ad_scalar_t(0.5) * ms2 * acc;
  };

  ad_.reset(new ocs2::CppAdInterface(fun, varDim, parDim, modelName, modelFolder));
  if (recompile)
    ad_->createModels(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
  else
    ad_->loadModelsIfAvailable(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
}

ocs2::scalar_t TorqueStageThetaAD::value(const ocs2::vector_t& theta, const ocs2::vector_t& params) const {
  return ad_->getFunctionValue(theta, params)(0);
}
ocs2::vector_t TorqueStageThetaAD::grad_theta(const ocs2::vector_t& theta, const ocs2::vector_t& params) const {
  const ocs2::matrix_t J = ad_->getJacobian(theta, params);  // 1 x m
  return J.row(0).transpose();
}

// ---------------- XU tape ----------------

TorqueStageXUAD::TorqueStageXUAD(size_t nx,
                                 size_t nu,
                                 size_t contactIndex,
                                 pinocchio::FrameIndex frameID,
                                 std::vector<size_t> tauActiveIndices,
                                 const ocs2::PinocchioInterfaceCppAd& pinocchioCppAd,
                                 const ocs2::humanoid::MpcRobotModelBase<ad_scalar_t>& robotModelAd,
                                 const std::string& modelName,
                                 const std::string& modelFolder,
                                 bool recompile,
                                 bool verbose)
    : nx_(nx),
      nu_(nu),
      contactIndex_(contactIndex),
      frameID_(frameID),
      tauActiveIndices_(std::move(tauActiveIndices)),
      pinCppAd_(pinocchioCppAd) {
  if (tauActiveIndices_.empty()) throw std::runtime_error("[TorqueStageXUAD] tauActiveIndices empty");

  robotAdPtr_.reset(robotModelAd.clone());
  if (!robotAdPtr_) throw std::runtime_error("[TorqueStageXUAD] robotModelAd.clone() returned null");

  const size_t m = tauActiveIndices_.size();
  const size_t varDim = nx_ + nu_;  // [x;u]
  const size_t parDim = 1 + 1 + m;  // [t|impact|theta]

  const std::string prefix = (contactIndex_ == 0) ? "left_leg_torque_cost." : "right_leg_torque_cost.";
  const ocs2::vector_t w0 = loadTorqueWeights(prefix, m);

  auto fun = [this, m, w0](const ad_vector_t& xu, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(1);

    const ad_scalar_t impact = p(1);
    const ad_vector_t th = p.segment(2, static_cast<long>(m));

    const ad_scalar_t midSwing = ad_scalar_t(1.0) - impact;
    const ad_scalar_t ms2 = midSwing * midSwing;

    const ad_vector_t x = xu.head(static_cast<long>(nx_));
    const ad_vector_t u = xu.tail(static_cast<long>(nu_));

    const ad_vector_t tauAct = computeTauExtActive(x, u, pinCppAd_, *robotAdPtr_, frameID_, contactIndex_, tauActiveIndices_);

    ad_scalar_t acc = ad_scalar_t(0.0);
    for (size_t i = 0; i < m; ++i) {
      const ad_scalar_t ti = tauAct(static_cast<long>(i));
      acc += th(static_cast<long>(i)) * ad_scalar_t(w0(static_cast<long>(i))) * ti * ti;
    }

    y(0) = ad_scalar_t(0.5) * ms2 * acc;
  };

  ad_.reset(new ocs2::CppAdInterface(fun, varDim, parDim, modelName, modelFolder));
  if (recompile)
    ad_->createModels(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
  else
    ad_->loadModelsIfAvailable(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
}

ocs2::scalar_t TorqueStageXUAD::value(const ocs2::vector_t& xu, const ocs2::vector_t& params) const {
  return ad_->getFunctionValue(xu, params)(0);
}
ocs2::vector_t TorqueStageXUAD::grad_xu(const ocs2::vector_t& xu, const ocs2::vector_t& params) const {
  const ocs2::matrix_t J = ad_->getJacobian(xu, params);  // 1 x (nx+nu)
  return J.row(0).transpose();
}

}  // namespace ocs2::humanoid_cost_matching
