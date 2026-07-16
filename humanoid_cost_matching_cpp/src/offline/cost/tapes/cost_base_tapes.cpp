#include "humanoid_cost_matching_cpp/offline/cost/tapes/cost_base_tapes.h"

#include <stdexcept>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include "humanoid_common_mpc/cost/EndEffectorKinematicCostHelpers.h"

namespace ocs2::humanoid_cost_matching {

using ad_scalar_t = ocs2::ad_scalar_t;
using ad_vector_t = ocs2::ad_vector_t;

static const std::string kTaskInfoPath =
    "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task_cost_matching.info";

// hardcode from task.info
static const std::string kBaseWeightsPrefix = "task_space_costs.torso.weights.";

static inline ad_vector_t taskSpaceVec13(const ad_vector_t& state,
                                         const ad_vector_t& input,
                                         ocs2::PinocchioInterfaceCppAd& pinCppAd,
                                         ocs2::humanoid::MpcRobotModelBase<ad_scalar_t>& robotAd,
                                         pinocchio::FrameIndex frameID) {
  const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;

  const auto& model = pinCppAd.getModel();
  auto& data = pinCppAd.getData();

  const ad_vector_t q = robotAd.getGeneralizedCoordinates(state);
  const ad_vector_t v = robotAd.getGeneralizedVelocities(state, input);

  pinocchio::forwardKinematics(model, data, q, v);
  const auto frameData = pinocchio::updateFramePlacement(model, data, frameID);

  ad_vector_t position = frameData.translation();  // 3
  const auto R = frameData.rotation();
  const auto quat = matrixToQuaternion(R);                                               // ad_quaternion_t, coeffs() is (x,y,z,w)
  ad_vector_t linVel = pinocchio::getFrameVelocity(model, data, frameID, rf).linear();   // 3
  ad_vector_t angVel = pinocchio::getFrameVelocity(model, data, frameID, rf).angular();  // 3

  ad_vector_t ts(13);
  ts << position, quat.coeffs(), linVel, angVel;
  return ts;
}

static inline ad_vector_t taskSpaceError12(const ad_vector_t& taskSpace13, const ad_vector_t& taskRef13) {
  // exactly like EndEffectorKinematicsQuadraticCost::costVectorFunction
  const auto elem = ocs2::humanoid::EndEffectorKinematicsCostElement<ad_scalar_t>(taskSpace13);
  const auto ref = ocs2::humanoid::EndEffectorKinematicsCostElement<ad_scalar_t>(taskRef13);
  return ocs2::humanoid::computeTaskSpaceErrors(elem, ref);  // 12
}

// ---------------- Stage Theta Tape ----------------

BaseStageThetaAD::BaseStageThetaAD(size_t nx,
                                   size_t nu,
                                   pinocchio::FrameIndex frameID,
                                   const ocs2::PinocchioInterfaceCppAd& pinocchioCppAd,
                                   const ocs2::humanoid::MpcRobotModelBase<ad_scalar_t>& robotModelAd,
                                   const std::string& modelName,
                                   const std::string& modelFolder,
                                   bool recompile,
                                   bool verbose)
    : nx_(nx), nu_(nu), frameID_(frameID), pinCppAd_(pinocchioCppAd) {
  robotAdPtr_.reset(robotModelAd.clone());
  if (!robotAdPtr_) throw std::runtime_error("[BaseStageThetaAD] robotModelAd.clone() returned null");

  const size_t varDim = kBaseErrDim;                  // theta_base_local (12)
  const size_t parDim = 1 + nx_ + nu_ + kBaseRefDim;  // [t|x|u|taskRef13]

  // Load nominal weights from task.info:
  // online: sqrtWeights_ = weights.toVector().cwiseSqrt()
  // expanded scalar cost uses weights directly.
  auto weightsStruct = ocs2::humanoid::EndEffectorKinematicsWeights::getWeights(kTaskInfoPath, kBaseWeightsPrefix, verbose);
  const ocs2::vector_t w0 = weightsStruct.toVector();  // (12)
  if (w0.size() != kBaseErrDim) {
    throw std::runtime_error("[BaseStageThetaAD] base weights size mismatch. got=" + std::to_string(w0.size()) +
                             " expected=" + std::to_string(kBaseErrDim) + " prefix=" + kBaseWeightsPrefix);
  }

  auto fun = [this, w0](const ad_vector_t& th, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(1);

    const ad_vector_t x = p.segment(1, nx_);
    const ad_vector_t u = p.segment(1 + nx_, nu_);
    const ad_vector_t taskRef = p.segment(1 + nx_ + nu_, kBaseRefDim);

    const ad_vector_t ts = taskSpaceVec13(x, u, pinCppAd_, *robotAdPtr_, frameID_);
    const ad_vector_t e = taskSpaceError12(ts, taskRef);

    // Match online expanded:
    //   cost = 0.5 * sum_i ( w0_i * theta_i * e_i^2 )
    ad_scalar_t cost = ad_scalar_t(0.0);
    for (int i = 0; i < kBaseErrDim; ++i) {
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

ocs2::scalar_t BaseStageThetaAD::value(const ocs2::vector_t& theta_base, const ocs2::vector_t& params) const {
  return ad_->getFunctionValue(theta_base, params)(0);
}
ocs2::vector_t BaseStageThetaAD::grad_theta(const ocs2::vector_t& theta_base, const ocs2::vector_t& params) const {
  const ocs2::matrix_t J = ad_->getJacobian(theta_base, params);  // 1 x 12
  return J.row(0).transpose();
}

// ---------------- Stage XU Tape ----------------
// variables=[x;u], params=[t|taskRef13|theta_base]

BaseStageXUAD::BaseStageXUAD(size_t nx,
                             size_t nu,
                             pinocchio::FrameIndex frameID,
                             const ocs2::PinocchioInterfaceCppAd& pinocchioCppAd,
                             const ocs2::humanoid::MpcRobotModelBase<ad_scalar_t>& robotModelAd,
                             const std::string& modelName,
                             const std::string& modelFolder,
                             bool recompile,
                             bool verbose)
    : nx_(nx), nu_(nu), frameID_(frameID), pinCppAd_(pinocchioCppAd) {
  robotAdPtr_.reset(robotModelAd.clone());
  if (!robotAdPtr_) throw std::runtime_error("[BaseStageXUAD] robotModelAd.clone() returned null");

  const size_t varDim = nx_ + nu_;                      // [x;u]
  const size_t parDim = 1 + kBaseRefDim + kBaseErrDim;  // [t|taskRef13|theta_base]

  // same nominal weights
  auto weightsStruct = ocs2::humanoid::EndEffectorKinematicsWeights::getWeights(kTaskInfoPath, kBaseWeightsPrefix, verbose);
  const ocs2::vector_t w0 = weightsStruct.toVector();  // (12)
  if (w0.size() != kBaseErrDim) {
    throw std::runtime_error("[BaseStageXUAD] base weights size mismatch. got=" + std::to_string(w0.size()) +
                             " expected=" + std::to_string(kBaseErrDim) + " prefix=" + kBaseWeightsPrefix);
  }

  auto fun = [this, w0](const ad_vector_t& xu, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(1);

    const ad_vector_t taskRef = p.segment(1, kBaseRefDim);
    const ad_vector_t th = p.segment(1 + kBaseRefDim, kBaseErrDim);

    const ad_vector_t x = xu.head(nx_);
    const ad_vector_t u = xu.tail(nu_);

    const ad_vector_t ts = taskSpaceVec13(x, u, pinCppAd_, *robotAdPtr_, frameID_);
    const ad_vector_t e = taskSpaceError12(ts, taskRef);

    ad_scalar_t cost = ad_scalar_t(0.0);
    for (int i = 0; i < kBaseErrDim; ++i) {
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

ocs2::scalar_t BaseStageXUAD::value(const ocs2::vector_t& xu, const ocs2::vector_t& params) const {
  return ad_->getFunctionValue(xu, params)(0);
}
ocs2::vector_t BaseStageXUAD::grad_xu(const ocs2::vector_t& xu, const ocs2::vector_t& params) const {
  const ocs2::matrix_t J = ad_->getJacobian(xu, params);  // 1 x (nx+nu)
  return J.row(0).transpose();
}

}  // namespace ocs2::humanoid_cost_matching
