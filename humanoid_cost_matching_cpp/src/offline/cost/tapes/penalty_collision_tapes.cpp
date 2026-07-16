#include "humanoid_cost_matching_cpp/offline/cost/tapes/penalty_collision_tapes.h"

#include <stdexcept>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include "humanoid_cost_matching_cpp/offline/cost/helpers/PieceWisePolynomialBarrierPenaltyAD.h"

namespace ocs2::humanoid_cost_matching {

CollisionStageXAD::CollisionStageXAD(size_t nx,
                                     const FootCollisionConfigOffline& cfg,
                                     const PiecewiseBarrierConfig& penCfg,
                                     const ocs2::PinocchioInterfaceCppAd& pinocchioAd,
                                     const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>& robotModelAd,
                                     const std::string& modelName,
                                     const std::string& modelFolder,
                                     bool recompile,
                                     bool verbose)
    : nx_(nx), cfg_(cfg), penCfg_(penCfg), pinAd_(&pinocchioAd), robotAd_(&robotModelAd) {
  if (!pinAd_ || !robotAd_) throw std::runtime_error("[CollisionStageXAD] pinAd/robotAd null");

  const auto& model = pinAd_->getModel();

  // Resolve frame ids once
  fid_leftAnkle_ = model.getFrameId(cfg_.leftAnkleFrame);
  fid_rightAnkle_ = model.getFrameId(cfg_.rightAnkleFrame);

  fid_leftFoot_ = model.getFrameId(cfg_.leftFootCenterFrame);
  fid_rightFoot_ = model.getFrameId(cfg_.rightFootCenterFrame);
  fid_leftFoot_p1_ = model.getFrameId(cfg_.leftFootFrame1);
  fid_rightFoot_p1_ = model.getFrameId(cfg_.rightFootFrame1);
  fid_leftFoot_p2_ = model.getFrameId(cfg_.leftFootFrame2);
  fid_rightFoot_p2_ = model.getFrameId(cfg_.rightFootFrame2);

  fid_leftKnee_ = model.getFrameId(cfg_.leftKneeFrame);
  fid_rightKnee_ = model.getFrameId(cfg_.rightKneeFrame);

  using ad_scalar_t = ocs2::ad_scalar_t;
  using ad_vector_t = ocs2::ad_vector_t;
  using ad_vector3_t = ocs2::ad_vector3_t;

  const size_t varDim = nx_;
  const size_t parDim = 0;

  auto fun = [this](const ad_vector_t& x, const ad_vector_t& /*p*/, ad_vector_t& y) {
    y.resize(1);

    const auto& model = pinAd_->getModel();
    auto data = pinAd_->getData();

    const ad_vector_t q = robotAd_->getGeneralizedCoordinates(x);
    pinocchio::forwardKinematics(model, data, q);

    // ankle
    const ad_vector3_t pos_ankle_l = pinocchio::updateFramePlacement(model, data, fid_leftAnkle_).translation();
    const ad_vector3_t pos_ankle_r = pinocchio::updateFramePlacement(model, data, fid_rightAnkle_).translation();

    // foot points
    const ad_vector3_t pos_f_l = pinocchio::updateFramePlacement(model, data, fid_leftFoot_).translation();
    const ad_vector3_t pos_f_r = pinocchio::updateFramePlacement(model, data, fid_rightFoot_).translation();
    const ad_vector3_t pos_f_l_p1 = pinocchio::updateFramePlacement(model, data, fid_leftFoot_p1_).translation();
    const ad_vector3_t pos_f_r_p1 = pinocchio::updateFramePlacement(model, data, fid_rightFoot_p1_).translation();
    const ad_vector3_t pos_f_l_p2 = pinocchio::updateFramePlacement(model, data, fid_leftFoot_p2_).translation();
    const ad_vector3_t pos_f_r_p2 = pinocchio::updateFramePlacement(model, data, fid_rightFoot_p2_).translation();

    // knee
    const ad_vector3_t pos_k_l = pinocchio::updateFramePlacement(model, data, fid_leftKnee_).translation();
    const ad_vector3_t pos_k_r = pinocchio::updateFramePlacement(model, data, fid_rightKnee_).translation();

    // EXACT online:
    // minDistFoot = 2 * parameters[0] (radius)
    // minDistKnee = 2 * parameters[1]
    const ad_scalar_t minDistFoot = ad_scalar_t(2.0) * ad_scalar_t(cfg_.footCollisionSphereRadius);
    const ad_scalar_t minDistKnee = ad_scalar_t(2.0) * ad_scalar_t(cfg_.kneeCollisionSphereRadius);

    ad_vector_t h(kNumConstraints);

    // EXACT online constraints (same ordering)
    h[0] = ((pos_f_l_p1 - pos_f_r_p1).norm() - minDistFoot);
    h[1] = ((pos_f_l_p1 - pos_f_r_p2).norm() - minDistFoot);
    h[2] = ((pos_f_l_p2 - pos_f_r_p1).norm() - minDistFoot);
    h[3] = ((pos_f_l_p2 - pos_f_r_p2).norm() - minDistFoot);

    h[4] = ((pos_f_l - pos_f_r_p1).norm() - minDistFoot);
    h[5] = ((pos_f_l - pos_f_r_p2).norm() - minDistFoot);
    h[6] = ((pos_f_r - pos_f_l_p1).norm() - minDistFoot);
    h[7] = ((pos_f_r - pos_f_l_p2).norm() - minDistFoot);
    h[8] = ((pos_f_l - pos_f_r).norm() - minDistFoot);

    h[9] = ((pos_k_l - pos_k_r).norm() - minDistKnee);

    h[10] = ((pos_f_l - pos_ankle_r).norm() - minDistFoot);
    h[11] = ((pos_f_l_p1 - pos_ankle_r).norm() - minDistFoot);
    h[12] = ((pos_f_l_p2 - pos_ankle_r).norm() - minDistFoot);
    h[13] = ((pos_f_r - pos_ankle_l).norm() - minDistFoot);
    h[14] = ((pos_f_r_p1 - pos_ankle_l).norm() - minDistFoot);
    h[15] = ((pos_f_r_p2 - pos_ankle_l).norm() - minDistFoot);

    const ad_scalar_t mu = ad_scalar_t(penCfg_.mu);
    const ad_scalar_t delta = ad_scalar_t(penCfg_.delta);

    // StateSoftConstraint online:
    // penalty_.getValue(time, h_vec)  -> elementwise barrier sum
    ad_scalar_t cost = ad_scalar_t(0.0);
    for (int i = 0; i < kNumConstraints; ++i) {
      cost += piecewiseBarrierValueExact(h[i], mu, delta);
    }

    y(0) = cost;
  };

  ad_.reset(new ocs2::CppAdInterface(fun, varDim, parDim, modelName, modelFolder));
  if (recompile) {
    ad_->createModels(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
  } else {
    ad_->loadModelsIfAvailable(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
  }
}

ocs2::scalar_t CollisionStageXAD::value(const ocs2::vector_t& x) const {
  ocs2::vector_t p;
  return ad_->getFunctionValue(x, p)(0);
}

ocs2::vector_t CollisionStageXAD::grad_x(const ocs2::vector_t& x) const {
  ocs2::vector_t p;
  const ocs2::matrix_t J = ad_->getJacobian(x, p);  // 1 x nx
  return J.row(0).transpose();
}

// -------- terminal wrapper --------

CollisionTerminalXAD::CollisionTerminalXAD(size_t nx,
                                           const FootCollisionConfigOffline& cfg,
                                           const PiecewiseBarrierConfig& penCfg,
                                           const ocs2::PinocchioInterfaceCppAd& pinocchioAd,
                                           const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>& robotModelAd,
                                           const std::string& modelName,
                                           const std::string& modelFolder,
                                           bool recompile,
                                           bool verbose)
    : impl_(nx, cfg, penCfg, pinocchioAd, robotModelAd, modelName, modelFolder, recompile, verbose) {}

ocs2::scalar_t CollisionTerminalXAD::value(const ocs2::vector_t& x) const {
  return impl_.value(x);
}
ocs2::vector_t CollisionTerminalXAD::grad_x(const ocs2::vector_t& x) const {
  return impl_.grad_x(x);
}

}  // namespace ocs2::humanoid_cost_matching
