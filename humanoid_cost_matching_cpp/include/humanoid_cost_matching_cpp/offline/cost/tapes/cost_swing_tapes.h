#pragma once

#include <memory>
#include <string>

#include <ocs2_core/Types.h>
#include <ocs2_core/automatic_differentiation/CppAdInterface.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_pinocchio_interface/PinocchioStateInputMapping.h>
#include <pinocchio/algorithm/frames.hpp>
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"

#include <ocs2_robotic_tools/common/RotationTransforms.h>

namespace ocs2::humanoid_cost_matching {

static constexpr int kSwingErrDim = 12;

/**
 * Swing stage theta tape (flat-ground 1:1 online):
 * variables = theta_swing (12)   (diag weights)
 * params    = [t | x(nx) | u(nu) | impactScaler(1)]
 *
 * reference (online getParameters):
 *   posRef = 0, planeNormal=(0,0,1), linVelRef=0, angVelRef=0
 * impactProximityScaler = getImpactProximityFactor(contactIndex,t)
 */
class SwingStageThetaAD {
 public:
  SwingStageThetaAD(size_t nx,
                    size_t nu,
                    pinocchio::FrameIndex frameID,
                    const ocs2::PinocchioInterfaceCppAd& pinocchioCppAd,
                    const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>& robotModelAd,
                    const std::string& modelName,
                    const std::string& modelFolder,
                    bool recompile,
                    bool verbose);

  ocs2::scalar_t value(const ocs2::vector_t& theta_swing, const ocs2::vector_t& params) const;
  ocs2::vector_t grad_theta(const ocs2::vector_t& theta_swing, const ocs2::vector_t& params) const;

 private:
  size_t nx_{0}, nu_{0};
  pinocchio::FrameIndex frameID_{0};
  ocs2::PinocchioInterfaceCppAd pinCppAd_;
  std::unique_ptr<ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>> robotAdPtr_;
  std::unique_ptr<ocs2::CppAdInterface> ad_;
};

/**
 * Swing stage XU tape:
 * variables = [x;u] (nx+nu)
 * params    = [t | impactScaler(1) | theta_swing(12)]
 */
class SwingStageXUAD {
 public:
  SwingStageXUAD(size_t nx,
                 size_t nu,
                 pinocchio::FrameIndex frameID,
                 const ocs2::PinocchioInterfaceCppAd& pinocchioCppAd,
                 const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>& robotModelAd,
                 const std::string& modelName,
                 const std::string& modelFolder,
                 bool recompile,
                 bool verbose);

  ocs2::scalar_t value(const ocs2::vector_t& xu, const ocs2::vector_t& params) const;
  ocs2::vector_t grad_xu(const ocs2::vector_t& xu, const ocs2::vector_t& params) const;

 private:
  size_t nx_{0}, nu_{0};
  pinocchio::FrameIndex frameID_{0};
  ocs2::PinocchioInterfaceCppAd pinCppAd_;
  std::unique_ptr<ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>> robotAdPtr_;
  std::unique_ptr<ocs2::CppAdInterface> ad_;
};

}  // namespace ocs2::humanoid_cost_matching
