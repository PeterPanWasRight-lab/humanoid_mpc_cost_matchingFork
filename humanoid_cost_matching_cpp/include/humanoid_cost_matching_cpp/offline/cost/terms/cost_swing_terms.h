#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include <ocs2_core/Types.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <pinocchio/multibody/fwd.hpp>
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

#include "humanoid_cost_matching_cpp/offline/core/RefPack.h"
#include "humanoid_cost_matching_cpp/offline/cost/ICostTerm.h"
#include "humanoid_cost_matching_cpp/offline/cost/tapes/cost_swing_tapes.h"

namespace ocs2::humanoid_cost_matching {

class SwingStageTerm final : public ICostTerm {
 public:
  SwingStageTerm(int nx,
                 int nu,
                 size_t contactIndex,
                 pinocchio::FrameIndex frameID,
                 const ocs2::PinocchioInterfaceCppAd& pinocchioCppAd,
                 const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>& robotModelAd,
                 std::string modelFolder,
                 bool recompile,
                 bool verbose,
                 const ocs2::humanoid::SwitchedModelReferenceManager* refMgr)
      : nx_(nx), nu_(nu), contactIndex_(contactIndex), frameID_(frameID), modelFolder_(std::move(modelFolder)), refMgr_(refMgr) {
    if (!refMgr_) throw std::runtime_error("[SwingtStageTerm] refMgr is null");

    thetaTape_.reset(new SwingStageThetaAD(nx_, nu_, frameID_, pinocchioCppAd, robotModelAd,
                                           "swing_stage_theta_contact" + std::to_string(contactIndex_), modelFolder_, recompile, verbose));

    xuTape_.reset(new SwingStageXUAD(nx_, nu_, frameID_, pinocchioCppAd, robotModelAd,
                                     "swing_stage_xu_contact" + std::to_string(contactIndex_), modelFolder_, recompile, verbose));
  }

  double value(double t,
               const ocs2::vector_t& x,
               const ocs2::vector_t& u,
               const RefPackKnot& /*refk*/,
               const ocs2::vector_t& theta,
               const ThetaLayout& layout) const override {
    // online-equivalent isActive(): swing cost only when NOT in contact
    if (refMgr_->isInContact(t, contactIndex_)) return 0.0;

    const double impact = refMgr_->getSwingTrajectoryPlanner()->getImpactProximityFactor(contactIndex_, t);
    const ocs2::vector_t theta_sw = layout.theta_swing(theta, contactIndex_);  // (12)

    // params = [t | x | u | impact]
    const ocs2::vector_t params = packParams_(t, x, u, impact);
    return thetaTape_->value(theta_sw, params);
  }

  ocs2::vector_t grad_theta(double t,
                            const ocs2::vector_t& x,
                            const ocs2::vector_t& u,
                            const RefPackKnot& /*refk*/,
                            const ocs2::vector_t& theta,
                            const ThetaLayout& layout) const override {
    ocs2::vector_t g_full = ocs2::vector_t::Zero(theta.size());
    if (refMgr_->isInContact(t, contactIndex_)) return g_full;

    const double impact = refMgr_->getSwingTrajectoryPlanner()->getImpactProximityFactor(contactIndex_, t);
    const ocs2::vector_t theta_sw = layout.theta_swing(theta, contactIndex_);  // (12)
    const ocs2::vector_t params = packParams_(t, x, u, impact);

    const ocs2::vector_t dldtheta = thetaTape_->grad_theta(theta_sw, params);  // (12)

    g_full.segment(static_cast<long>(layout.off_swing(contactIndex_)), static_cast<long>(ThetaLayout::swing_dim_per_foot)) += dldtheta;
    return g_full;
  }

  ocs2::vector_t grad_x(double t,
                        const ocs2::vector_t& x,
                        const ocs2::vector_t& u,
                        const RefPackKnot& /*refk*/,
                        const ocs2::vector_t& theta,
                        const ThetaLayout& layout) const override {
    if (refMgr_->isInContact(t, contactIndex_)) {
      return ocs2::vector_t::Zero(nx_);
    }

    const double impact = refMgr_->getSwingTrajectoryPlanner()->getImpactProximityFactor(contactIndex_, t);
    const ocs2::vector_t theta_sw = layout.theta_swing(theta, contactIndex_);  // (12)

    // variables z = [x;u]
    ocs2::vector_t xu(nx_ + nu_);
    xu << x, u;

    // params = [t | impact | theta_sw]
    const ocs2::vector_t paramsXU = packParamsXU_(t, impact, theta_sw);

    const ocs2::vector_t dldxu = xuTape_->grad_xu(xu, paramsXU);  // (nx+nu)
    return dldxu.head(nx_);
  }

 private:
  // theta tape params: [t | x | u | impact]
  ocs2::vector_t packParams_(double t, const ocs2::vector_t& x, const ocs2::vector_t& u, double impact) const {
    ocs2::vector_t p(1 + nx_ + nu_ + 1);
    p << t, x, u, impact;
    return p;
  }

  // XU tape params: [t | impact | theta_sw]
  ocs2::vector_t packParamsXU_(double t, double impact, const ocs2::vector_t& theta_sw) const {
    ocs2::vector_t p(1 + 1 + static_cast<int>(theta_sw.size()));
    p << t, impact, theta_sw;
    return p;
  }

 private:
  int nx_ = 0;
  int nu_ = 0;
  size_t contactIndex_ = 0;

  pinocchio::FrameIndex frameID_{0};

  std::string modelFolder_;
  const ocs2::humanoid::SwitchedModelReferenceManager* refMgr_ = nullptr;

  std::unique_ptr<SwingStageThetaAD> thetaTape_;
  std::unique_ptr<SwingStageXUAD> xuTape_;
};

}  // namespace ocs2::humanoid_cost_matching
