#pragma once

#include <memory>
#include <string>

#include <ocs2_core/Types.h>
#include <ocs2_core/automatic_differentiation/CppAdInterface.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"

#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/model.hpp>

namespace ocs2::humanoid_cost_matching {

/**
 * ZeroVelocity (stance foot) equality constraint → smooth L2 penalty
 *
 * Online:
 *   b[2] = -Ax(2,2) * footRefHeightZ   (in online this comes from precomp / swing planner)
 *   g = b + Ax*footPose(x) + Av*twist(x,u)
 *   cost = 0.5 * w * ||g||^2
 *
 * Here:
 *   - w/posGainZ/oriGain are baked (stored in tape)
 *   - params only contains: [footRefHeightZ]  (dim=1)
 *
 * variables: xu (nx+nu)
 */
class ZeroVelocityStageXUAD final {
 public:
  ZeroVelocityStageXUAD(size_t nx,
                        size_t nu,
                        pinocchio::FrameIndex frameId,
                        double w,
                        double posGainZ,
                        double oriGain,
                        const ocs2::PinocchioInterfaceCppAd& pinocchioAd,
                        const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>& robotModelAd,
                        std::string modelName,
                        const std::string& modelFolder,
                        bool recompile,
                        bool verbose);

  // params: [footRefHeightZ]
  ocs2::scalar_t value(const ocs2::vector_t& xu, const ocs2::vector_t& params) const;
  ocs2::vector_t grad_xu(const ocs2::vector_t& xu, const ocs2::vector_t& params) const;

 private:
  size_t nx_{0};
  size_t nu_{0};
  pinocchio::FrameIndex frameId_{0};

  // baked constants
  double w_{1.0};
  double posGainZ_{0.0};
  double oriGain_{0.0};

  // IMPORTANT: copy pinocchioAd (by value) + clone robotModel (writable)
  ocs2::PinocchioInterfaceCppAd pinocchioAd_;
  std::unique_ptr<ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>> robotModelAdPtr_;

  std::unique_ptr<ocs2::CppAdInterface> ad_;
};

}  // namespace ocs2::humanoid_cost_matching
