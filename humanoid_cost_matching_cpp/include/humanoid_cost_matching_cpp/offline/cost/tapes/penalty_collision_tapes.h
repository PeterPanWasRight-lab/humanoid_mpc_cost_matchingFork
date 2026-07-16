#pragma once

#include <memory>
#include <string>

#include <ocs2_core/Types.h>
#include <ocs2_core/automatic_differentiation/CppAdInterface.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <pinocchio/multibody/model.hpp>

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_cost_matching_cpp/offline/cost/helpers/PieceWisePolynomialBarrierPenaltyAD.h"

namespace ocs2::humanoid_cost_matching {

struct FootCollisionConfigOffline {
  // Foot and ankle (loaded from task.info)
  std::string leftAnkleFrame;
  std::string rightAnkleFrame;

  // Foot frames: MUST match online defaults
  std::string leftFootCenterFrame{"foot_l_contact"};
  std::string rightFootCenterFrame{"foot_r_contact"};
  std::string leftFootFrame1{"foot_l_contact_collision_p_1"};
  std::string rightFootFrame1{"foot_r_contact_collision_p_1"};
  std::string leftFootFrame2{"foot_l_contact_collision_p_2"};
  std::string rightFootFrame2{"foot_r_contact_collision_p_2"};

  ocs2::scalar_t footCollisionSphereRadius{0.0};

  // Knee (loaded from task.info)
  std::string leftKneeFrame;
  std::string rightKneeFrame;
  ocs2::scalar_t kneeCollisionSphereRadius{0.0};
};

class CollisionStageXAD {
 public:
  static constexpr int kNumConstraints = 16;

  CollisionStageXAD(size_t nx,
                    const FootCollisionConfigOffline& cfg,
                    const PiecewiseBarrierConfig& penCfg,
                    const ocs2::PinocchioInterfaceCppAd& pinocchioAd,
                    const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>& robotModelAd,
                    const std::string& modelName,
                    const std::string& modelFolder,
                    bool recompile,
                    bool verbose);

  ocs2::scalar_t value(const ocs2::vector_t& x) const;
  ocs2::vector_t grad_x(const ocs2::vector_t& x) const;

 private:
  size_t nx_{0};

  FootCollisionConfigOffline cfg_;
  PiecewiseBarrierConfig penCfg_;

  const ocs2::PinocchioInterfaceCppAd* pinAd_{nullptr};
  const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>* robotAd_{nullptr};

  // pre-resolved frame ids (from AD model)
  pinocchio::FrameIndex fid_leftAnkle_{0}, fid_rightAnkle_{0};
  pinocchio::FrameIndex fid_leftFoot_{0}, fid_rightFoot_{0};
  pinocchio::FrameIndex fid_leftFoot_p1_{0}, fid_rightFoot_p1_{0};
  pinocchio::FrameIndex fid_leftFoot_p2_{0}, fid_rightFoot_p2_{0};
  pinocchio::FrameIndex fid_leftKnee_{0}, fid_rightKnee_{0};

  std::unique_ptr<ocs2::CppAdInterface> ad_;
};

class CollisionTerminalXAD {
 public:
  CollisionTerminalXAD(size_t nx,
                       const FootCollisionConfigOffline& cfg,
                       const PiecewiseBarrierConfig& penCfg,
                       const ocs2::PinocchioInterfaceCppAd& pinocchioAd,
                       const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>& robotModelAd,
                       const std::string& modelName,
                       const std::string& modelFolder,
                       bool recompile,
                       bool verbose);

  ocs2::scalar_t value(const ocs2::vector_t& x) const;
  ocs2::vector_t grad_x(const ocs2::vector_t& x) const;

 private:
  CollisionStageXAD impl_;
};

}  // namespace ocs2::humanoid_cost_matching
