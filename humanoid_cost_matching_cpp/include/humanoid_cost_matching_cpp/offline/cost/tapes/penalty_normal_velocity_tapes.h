#pragma once

#include <memory>
#include <string>

#include <ocs2_core/Types.h>
#include <ocs2_core/automatic_differentiation/CppAdInterface.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/model.hpp>

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"

namespace ocs2::humanoid_cost_matching {

/**
 * NormalVelocity constraint (offline penalty):
 * Active when foot is NOT in contact (same as NormalVelocityConstraintCppAd::isActive()).
 *
 * Online precomp builds:
 *   b = -zVelRef
 *   Av = [0 0 1]
 *   if posGainZ != 0:
 *     b -= posGainZ * zPosRef
 *     Ax = [0 0 posGainZ]
 *
 * And constraint is: g = b + Ax * pos + Av * vel  (scalar)
 *
 * Here we implement stage penalty cost:
 *   cost = 0.5 * w * g^2
 *
 * Notes:
 * - w and posGainZ are baked in the term, NOT passed to tape params.
 * - zVelRef/zPosRef are computed in term using swingTrajectoryPlanner, so tape only needs (x,u).
 */
class NormalVelStageXUAD {
 public:
  using ad_scalar_t = ocs2::ad_scalar_t;
  using ad_vector_t = ocs2::ad_vector_t;

  NormalVelStageXUAD(size_t nx,
                     size_t nu,
                     pinocchio::FrameIndex frameId,
                     const ocs2::PinocchioInterfaceCppAd& pinocchioAd,
                     const ocs2::humanoid::MpcRobotModelBase<ad_scalar_t>& robotModelAd,
                     std::string modelName,
                     const std::string& modelFolder,
                     bool recompile,
                     bool verbose);

  // params = [b, ax_z]  (both double, but passed as vector_t at runtime)
  // xu    = [x;u]
  ocs2::scalar_t value(const ocs2::vector_t& xu, const ocs2::vector_t& params) const;
  ocs2::vector_t grad_xu(const ocs2::vector_t& xu, const ocs2::vector_t& params) const;

 private:
  size_t nx_{0}, nu_{0};
  pinocchio::FrameIndex frameId_{0};

  // strategy: tape holds its own copies
  ocs2::PinocchioInterfaceCppAd pinocchioAd_;
  std::unique_ptr<ocs2::humanoid::MpcRobotModelBase<ad_scalar_t>> robotModelAdPtr_;

  std::unique_ptr<ocs2::CppAdInterface> ad_;
};

}  // namespace ocs2::humanoid_cost_matching
