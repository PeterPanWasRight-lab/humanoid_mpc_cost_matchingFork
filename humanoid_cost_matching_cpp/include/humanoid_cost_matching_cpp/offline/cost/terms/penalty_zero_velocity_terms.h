#pragma once

#include <memory>
#include <string>

#include <ocs2_core/Types.h>

#include "humanoid_cost_matching_cpp/offline/core/RefPack.h"
#include "humanoid_cost_matching_cpp/offline/core/ThetaLayout.h"
#include "humanoid_cost_matching_cpp/offline/cost/ICostTerm.h"
#include "humanoid_cost_matching_cpp/offline/cost/tapes/penalty_zero_velocity_tapes.h"

#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid_cost_matching {

/**
 * ZeroVelocity penalty stage term (per foot).
 *
 * w/posGainZ/oriGain baked (stored in tape).
 * params only: [footRefHeightZ] computed online from swingTrajectoryPlanner:
 *   footRefHeightZ := swing->getZpositionConstraint(foot, t)
 *
 * active when contact flag is true (same as online ZeroVelocityConstraintCppAd)
 */
class ZeroVelocityStageTerm final : public ICostTerm {
 public:
  ZeroVelocityStageTerm(int nx,
                        int nu,
                        size_t contactIndex,
                        pinocchio::FrameIndex frameId,
                        double w_task,
                        double posGainZ,
                        double oriGain,
                        const ocs2::PinocchioInterfaceCppAd& pinocchioAd,
                        const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>& robotModelAd,
                        const std::string& modelFolder,
                        bool recompile,
                        bool verbose,
                        const ocs2::humanoid::SwitchedModelReferenceManager* refMgr,
                        std::string modelNamePrefix)
      : nx_(nx), nu_(nu), contactIndex_(contactIndex), refMgr_(refMgr) {
    if (!refMgr_) throw std::runtime_error("[ZeroVelocityStageTerm] refMgr is null");

    tape_ = std::make_unique<ZeroVelocityStageXUAD>(static_cast<size_t>(nx_), static_cast<size_t>(nu_), frameId,
                                                    /*w=*/w_task,
                                                    /*posGainZ=*/posGainZ,
                                                    /*oriGain=*/oriGain, pinocchioAd, robotModelAd,
                                                    std::move(modelNamePrefix) + "_foot" + std::to_string(contactIndex_) + "_stage_xu",
                                                    modelFolder, recompile, verbose);
  }

  double value(double t,
               const ocs2::vector_t& x,
               const ocs2::vector_t& u,
               const RefPackKnot& refk,
               const ocs2::vector_t& theta,
               const ThetaLayout& layout) const override {
    (void)refk;
    (void)theta;
    (void)layout;
    if (!isActive(t)) return 0.0;

    const auto& swing = refMgr_->getSwingTrajectoryPlanner();
    if (!swing) throw std::runtime_error("[ZeroVelocityStageTerm] swingTrajectoryPlanner is null");
    const double footRefHeightZ = swing->getZpositionConstraint(contactIndex_, t);

    ocs2::vector_t xu(nx_ + nu_);
    xu << x, u;

    ocs2::vector_t p(1);
    p(0) = footRefHeightZ;
    return tape_->value(xu, p);
  }

  ocs2::vector_t grad_theta(double t,
                            const ocs2::vector_t& x,
                            const ocs2::vector_t& u,
                            const RefPackKnot& refk,
                            const ocs2::vector_t& theta,
                            const ThetaLayout& layout) const override {
    (void)t;
    (void)x;
    (void)u;
    (void)refk;
    (void)layout;
    return ocs2::vector_t::Zero(theta.size());
  }

  ocs2::vector_t grad_x(double t,
                        const ocs2::vector_t& x,
                        const ocs2::vector_t& u,
                        const RefPackKnot& refk,
                        const ocs2::vector_t& theta,
                        const ThetaLayout& layout) const override {
    (void)refk;
    (void)theta;
    (void)layout;
    if (!isActive(t)) return ocs2::vector_t::Zero(x.size());

    const auto& swing = refMgr_->getSwingTrajectoryPlanner();
    if (!swing) throw std::runtime_error("[ZeroVelocityStageTerm] swingTrajectoryPlanner is null");
    const double footRefHeightZ = swing->getZpositionConstraint(contactIndex_, t);

    ocs2::vector_t xu(nx_ + nu_);
    xu << x, u;

    ocs2::vector_t p(1);
    p(0) = footRefHeightZ;

    const ocs2::vector_t g = tape_->grad_xu(xu, p);
    return g.head(nx_);
  }

 private:
  bool isActive(double t) const { return refMgr_->getContactFlags(t)[contactIndex_]; }

  int nx_{0}, nu_{0};
  size_t contactIndex_{0};

  const ocs2::humanoid::SwitchedModelReferenceManager* refMgr_{nullptr};
  std::unique_ptr<ZeroVelocityStageXUAD> tape_;
};

}  // namespace ocs2::humanoid_cost_matching
