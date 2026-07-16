#pragma once

#include <memory>
#include <string>

#include <ocs2_core/Types.h>
#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "humanoid_cost_matching_cpp/offline/core/RefPack.h"
#include "humanoid_cost_matching_cpp/offline/core/ThetaLayout.h"
#include "humanoid_cost_matching_cpp/offline/cost/ICostTerm.h"
#include "humanoid_cost_matching_cpp/offline/cost/tapes/penalty_normal_velocity_tapes.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid_cost_matching {

/**
 * NormalVelocity penalty term (stage only):
 * - Active when foot is NOT in contact (same as online NormalVelocityConstraintCppAd)
 * - Uses swingTrajectoryPlanner for zVelRef/zPosRef (online-identical source)
 * - cost = 0.5 * w * g^2
 *
 * g = (-zVelRef - posGainZ*zPosRef) + posGainZ*pos_z + vel_z
 *   = (vel_z - zVelRef) + posGainZ*(pos_z - zPosRef)
 */
class NormalVelStageTerm final : public ICostTerm {
 public:
  NormalVelStageTerm(int nx,
                     int nu,
                     size_t footIndex,
                     pinocchio::FrameIndex frameId,
                     const ocs2::PinocchioInterfaceCppAd& pinocchioAd,
                     const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>& robotModelAd,
                     const std::string& modelFolder,
                     bool recompile,
                     bool verbose,
                     const ocs2::humanoid::SwitchedModelReferenceManager* refMgr,
                     double w,
                     double posGainZ)
      : nx_(nx), nu_(nu), footIndex_(footIndex), frameId_(frameId), refMgr_(refMgr), w_(w), posGainZ_(posGainZ) {
    if (!refMgr_) throw std::runtime_error("[NormalVelStageTerm] refMgr is null");
    tape_ = std::make_unique<NormalVelStageXUAD>(static_cast<size_t>(nx_), static_cast<size_t>(nu_), frameId_, pinocchioAd, robotModelAd,
                                                 "pen_normalVel_g_foot" + std::to_string(footIndex_), modelFolder, recompile, verbose);
  }

  double value(double t,
               const ocs2::vector_t& x,
               const ocs2::vector_t& u,
               const RefPackKnot& /*refk*/,
               const ocs2::vector_t& /*theta*/,
               const ThetaLayout& /*layout*/) const override {
    if (!isActive(t)) return 0.0;

    const auto& swing = refMgr_->getSwingTrajectoryPlanner();
    const double zVelRef = swing->getZvelocityConstraint(footIndex_, t);
    const double zPosRef = swing->getZpositionConstraint(footIndex_, t);

    // params: [b, ax_z]
    // g = b + ax_z*pos_z + vel_z
    // want g = (vel_z - zVelRef) + posGainZ*(pos_z - zPosRef)
    // => b = -(zVelRef + posGainZ*zPosRef), ax_z = posGainZ
    ocs2::vector_t params(2);
    params(0) = -(zVelRef + posGainZ_ * zPosRef);
    params(1) = posGainZ_;

    ocs2::vector_t xu(nx_ + nu_);
    xu << x, u;

    const double g = tape_->value(xu, params);
    return 0.5 * w_ * g * g;
  }

  ocs2::vector_t grad_theta(double /*t*/,
                            const ocs2::vector_t& /*x*/,
                            const ocs2::vector_t& /*u*/,
                            const RefPackKnot& /*refk*/,
                            const ocs2::vector_t& theta,
                            const ThetaLayout& /*layout*/) const override {
    // this penalty has no theta parameters
    return ocs2::vector_t::Zero(theta.size());
  }

  ocs2::vector_t grad_x(double t,
                        const ocs2::vector_t& x,
                        const ocs2::vector_t& u,
                        const RefPackKnot& /*refk*/,
                        const ocs2::vector_t& /*theta*/,
                        const ThetaLayout& /*layout*/) const override {
    if (!isActive(t)) return ocs2::vector_t::Zero(x.size());

    const auto& swing = refMgr_->getSwingTrajectoryPlanner();
    const double zVelRef = swing->getZvelocityConstraint(footIndex_, t);
    const double zPosRef = swing->getZpositionConstraint(footIndex_, t);

    ocs2::vector_t params(2);
    params(0) = -(zVelRef + posGainZ_ * zPosRef);
    params(1) = posGainZ_;

    ocs2::vector_t xu(nx_ + nu_);
    xu << x, u;

    const double g = tape_->value(xu, params);
    const ocs2::vector_t dg_dxu = tape_->grad_xu(xu, params);

    // d(0.5*w*g^2)/dx = w*g * dg/dx
    return (w_ * g) * dg_dxu.head(nx_);
  }

 private:
  bool isActive(double t) const {
    // NormalVelocityConstraintCppAd::isActive(): active when NOT in contact
    return !refMgr_->getContactFlags(t)[footIndex_];
  }

  int nx_{0}, nu_{0};
  size_t footIndex_{0};
  pinocchio::FrameIndex frameId_{0};

  const ocs2::humanoid::SwitchedModelReferenceManager* refMgr_{nullptr};

  double w_{1.0};
  double posGainZ_{0.0};

  std::unique_ptr<NormalVelStageXUAD> tape_;
};

}  // namespace ocs2::humanoid_cost_matching
