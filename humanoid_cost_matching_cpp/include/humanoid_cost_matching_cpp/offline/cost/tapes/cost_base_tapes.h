#pragma once

#include <memory>
#include <string>
#include <vector>

#include <ocs2_core/Types.h>
#include <ocs2_core/automatic_differentiation/CppAdInterface.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/fwd.hpp>

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/cost/EndEffectorKinematicsQuadraticCost.h"

namespace ocs2::humanoid_cost_matching {

static constexpr int kBaseErrDim = 12;  // task-space error dim
static constexpr int kBaseRefDim = 13;  // task-space element dim (pos3 + quat4 + linVel3 + angVel3)

/**
 * Base (task-space kinematics) stage theta tape
 * variables = theta_base_local (12)   (diag weights)
 * params    = [t | x(nx) | u(nu) | taskRef(13)]
 */
class BaseStageThetaAD {
 public:
  BaseStageThetaAD(size_t nx,
                   size_t nu,
                   pinocchio::FrameIndex frameID,
                   const ocs2::PinocchioInterfaceCppAd& pinocchioCppAd,
                   const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>& robotModelAd,
                   const std::string& modelName,
                   const std::string& modelFolder,
                   bool recompile,
                   bool verbose);

  ocs2::scalar_t value(const ocs2::vector_t& theta_base, const ocs2::vector_t& params) const;
  ocs2::vector_t grad_theta(const ocs2::vector_t& theta_base, const ocs2::vector_t& params) const;

 private:
  size_t nx_{0}, nu_{0};
  pinocchio::FrameIndex frameID_{0};
  ocs2::PinocchioInterfaceCppAd pinCppAd_;
  std::unique_ptr<ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>> robotAdPtr_;
  std::unique_ptr<ocs2::CppAdInterface> ad_;
};

/**
 * Base (task-space kinematics) stage XU tape
 * variables = [x;u] (nx+nu)
 * params    = [t | taskRef(13) | theta_base_local(12)]
 */
class BaseStageXUAD {
 public:
  BaseStageXUAD(size_t nx,
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
