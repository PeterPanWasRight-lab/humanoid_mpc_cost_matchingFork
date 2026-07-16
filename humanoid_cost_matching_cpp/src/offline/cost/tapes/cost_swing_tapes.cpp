#include "humanoid_cost_matching_cpp/offline/cost/tapes/cost_swing_tapes.h"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <ocs2_core/misc/LoadData.h>

#include <stdexcept>
#include "humanoid_common_mpc/cost/EndEffectorKinematicCostHelpers.h"

namespace ocs2::humanoid_cost_matching {

using ad_scalar_t = ocs2::ad_scalar_t;
using ad_vector_t = ocs2::ad_vector_t;
using ad_matrix3_t = Eigen::Matrix<ad_scalar_t, 3, 3>;
using ad_vector3_t = Eigen::Matrix<ad_scalar_t, 3, 1>;

static const std::string kTaskInfoPath =
    "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task_cost_matching.info";

static inline ad_vector_t swingErrorFlatGroundOnline(const ad_vector_t& state,
                                                     const ad_vector_t& input,
                                                     ocs2::PinocchioInterfaceCppAd& pinCppAd,
                                                     ocs2::humanoid::MpcRobotModelBase<ad_scalar_t>& robotAd,
                                                     pinocchio::FrameIndex frameID,
                                                     ad_scalar_t impactScaler) {
  const pinocchio::ReferenceFrame rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;

  const auto& model = pinCppAd.getModel();
  auto& data = pinCppAd.getData();

  const ad_vector_t q = robotAd.getGeneralizedCoordinates(state);
  const ad_vector_t v = robotAd.getGeneralizedVelocities(state, input);

  pinocchio::forwardKinematics(model, data, q, v);
  const auto frameData = pinocchio::updateFramePlacement(model, data, frameID);

  const ad_vector3_t pos = frameData.translation();
  const ad_matrix3_t R = frameData.rotation();
  const ad_vector3_t linVel = pinocchio::getFrameVelocity(model, data, frameID, rf).linear();
  const ad_vector3_t angVel = pinocchio::getFrameVelocity(model, data, frameID, rf).angular();

  // -------- online reference (getParameters): flat ground all zeros --------
  const ad_vector3_t posRef(ad_scalar_t(0.0), ad_scalar_t(0.0), ad_scalar_t(0.0));
  const ad_vector3_t planeNormalRef(ad_scalar_t(0.0), ad_scalar_t(0.0), ad_scalar_t(1.0));
  const ad_vector3_t linVelRef(ad_scalar_t(0.0), ad_scalar_t(0.0), ad_scalar_t(0.0));
  const ad_vector3_t angVelRef(ad_scalar_t(0.0), ad_scalar_t(0.0), ad_scalar_t(0.0));

  // online uses rotationMatrixDistanceToPlane(orientation, reference.getPlaneNormal())
  const ad_vector3_t rotPlaneErr = ocs2::rotationMatrixDistanceToPlane<ad_scalar_t>(R, planeNormalRef);

  ad_vector_t e(kSwingErrDim);
  e.segment(0, 3) = (pos - posRef);
  e.segment(3, 3) = rotPlaneErr;
  e.segment(6, 3) = (linVel - linVelRef) * impactScaler;
  e.segment(9, 3) = (angVel - angVelRef);
  return e;
}

// ---------------- Stage Theta Tape ----------------

SwingStageThetaAD::SwingStageThetaAD(size_t nx,
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
  if (!robotAdPtr_) throw std::runtime_error("[SwingStageThetaAD] robotModelAd.clone() returned null");

  const size_t varDim = kSwingErrDim;       // theta_swing (12)
  const size_t parDim = 1 + nx_ + nu_ + 1;  // [t|x|u|impactScaler]

  // Load nominal weights from task.info (pos/ori/linvel/angvel), 12 dims.
  // online: sqrtWeights_ = weights.toVector().cwiseSqrt()
  // expanded scalar cost uses weights directly.
  auto weightsStruct = ocs2::humanoid::EndEffectorKinematicsWeights::getWeights(kTaskInfoPath, "task_space_foot_cost_weights.", verbose);
  const ocs2::vector_t w0 = weightsStruct.toVector();  // (12)

  if (w0.size() != kSwingErrDim) {
    throw std::runtime_error("[SwingStageThetaAD] task_space_foot_cost_weights size mismatch. got=" + std::to_string(w0.size()) +
                             " expected=" + std::to_string(kSwingErrDim));
  }

  auto fun = [this, w0](const ad_vector_t& th, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(1);

    const ad_vector_t x = p.segment(1, nx_);
    const ad_vector_t u = p.segment(1 + nx_, nu_);
    const ad_scalar_t impactScaler = p(1 + nx_ + nu_);

    const ad_vector_t e = swingErrorFlatGroundOnline(x, u, pinCppAd_, *robotAdPtr_, frameID_, impactScaler);

    // Match online expanded:
    //   cost = 0.5 * sum_i ( w0_i * theta_i * e_i^2 )
    ad_scalar_t cost = ad_scalar_t(0.0);
    for (int i = 0; i < kSwingErrDim; ++i) {
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

ocs2::scalar_t SwingStageThetaAD::value(const ocs2::vector_t& theta_swing, const ocs2::vector_t& params) const {
  return ad_->getFunctionValue(theta_swing, params)(0);
}

ocs2::vector_t SwingStageThetaAD::grad_theta(const ocs2::vector_t& theta_swing, const ocs2::vector_t& params) const {
  const ocs2::matrix_t J = ad_->getJacobian(theta_swing, params);  // 1 x 12
  return J.row(0).transpose();
}

// ---------------- Stage XU Tape ----------------
// variables=[x;u], params=[t|impactScaler|theta]

SwingStageXUAD::SwingStageXUAD(size_t nx,
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
  if (!robotAdPtr_) throw std::runtime_error("[SwingStageXUAD] robotModelAd.clone() returned null");

  const size_t varDim = nx_ + nu_;             // [x;u]
  const size_t parDim = 1 + 1 + kSwingErrDim;  // [t|impactScaler|theta]

  // same nominal weights
  auto weightsStruct = ocs2::humanoid::EndEffectorKinematicsWeights::getWeights(kTaskInfoPath, "task_space_foot_cost_weights.", verbose);
  const ocs2::vector_t w0 = weightsStruct.toVector();  // (12)
  if (w0.size() != kSwingErrDim) {
    throw std::runtime_error("[SwingStageXUAD] task_space_foot_cost_weights size mismatch. got=" + std::to_string(w0.size()) +
                             " expected=" + std::to_string(kSwingErrDim));
  }

  auto fun = [this, w0](const ad_vector_t& xu, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(1);

    const ad_scalar_t impactScaler = p(1);
    const ad_vector_t th = p.segment(2, kSwingErrDim);

    const ad_vector_t x = xu.head(nx_);
    const ad_vector_t u = xu.tail(nu_);

    const ad_vector_t e = swingErrorFlatGroundOnline(x, u, pinCppAd_, *robotAdPtr_, frameID_, impactScaler);

    ad_scalar_t cost = ad_scalar_t(0.0);
    for (int i = 0; i < kSwingErrDim; ++i) {
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

ocs2::scalar_t SwingStageXUAD::value(const ocs2::vector_t& xu, const ocs2::vector_t& params) const {
  return ad_->getFunctionValue(xu, params)(0);
}

ocs2::vector_t SwingStageXUAD::grad_xu(const ocs2::vector_t& xu, const ocs2::vector_t& params) const {
  const ocs2::matrix_t J = ad_->getJacobian(xu, params);  // 1 x (nx+nu)
  return J.row(0).transpose();
}

}  // namespace ocs2::humanoid_cost_matching
