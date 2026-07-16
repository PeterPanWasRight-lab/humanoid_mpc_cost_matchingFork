#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <ocs2_core/Types.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

#include "humanoid_cost_matching_cpp/offline/core/RefPack.h"
#include "humanoid_cost_matching_cpp/offline/core/ThetaLayout.h"
#include "humanoid_cost_matching_cpp/offline/cost/ICostTerm.h"
#include "humanoid_cost_matching_cpp/offline/cost/tapes/cost_torque_tapes.h"

namespace ocs2::humanoid_cost_matching {

/**
 * Reproduces online ExternalTorqueQuadraticCostAD:
 * - isActive(time) = referenceManager.getContactFlags(time)[contactIndex]
 * - params: impactProximityScaler = swingPlanner.getImpactProximityFactor(1-contactIndex, time)
 * - midSwing = 1 - impact
 *
 * theta_torque layout: [leg0(6) | leg1(6)]
 */
class TorqueStageTerm final : public ICostTerm {
 public:
  TorqueStageTerm(int nx,
                  int nu,
                  size_t contactIndex,
                  pinocchio::FrameIndex frameID,
                  std::vector<size_t> tauActiveIndices,  // indices into tauExt (6+jointIndex)
                  const ocs2::PinocchioInterfaceCppAd& pinocchioCppAd,
                  const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>& robotModelAd,
                  std::string modelFolder,
                  bool recompile,
                  bool verbose,
                  const ocs2::humanoid::SwitchedModelReferenceManager* refMgr)
      : nx_(nx),
        nu_(nu),
        contactIndex_(contactIndex),
        frameID_(frameID),
        tauActiveIndices_(std::move(tauActiveIndices)),
        modelFolder_(std::move(modelFolder)),
        refMgr_(refMgr) {
    if (!refMgr_) throw std::runtime_error("[TorqueStageTerm] refMgr is null");
    if (tauActiveIndices_.empty()) throw std::runtime_error("[TorqueStageTerm] tauActiveIndices empty");

    // Safety check: ensure tauActiveIndices size matches expected layout dim
    if (tauActiveIndices_.size() != ThetaLayout::torque_dim_per_leg) {
      throw std::runtime_error("[TorqueStageTerm] tauActiveIndices size (" + std::to_string(tauActiveIndices_.size()) +
                               ") does not match layout dim (" + std::to_string(ThetaLayout::torque_dim_per_leg) + ")");
    }

    const std::string suf = "_c" + std::to_string(contactIndex_);
    thetaTape_.reset(new TorqueStageThetaAD(nx_, nu_, contactIndex_, frameID_, tauActiveIndices_, pinocchioCppAd, robotModelAd,
                                            "torque_stage_theta" + suf, modelFolder_, recompile, verbose));
    xuTape_.reset(new TorqueStageXUAD(nx_, nu_, contactIndex_, frameID_, tauActiveIndices_, pinocchioCppAd, robotModelAd,
                                      "torque_stage_xu" + suf, modelFolder_, recompile, verbose));
  }

  double value(double t,
               const ocs2::vector_t& x,
               const ocs2::vector_t& u,
               const RefPackKnot& /*refk*/,
               const ocs2::vector_t& theta,
               const ThetaLayout& layout) const override {
    if (!isActive_(t)) return 0.0;

    const double impact = impact_(t);
    const ocs2::vector_t th = thetaLocal_(theta, layout);

    ocs2::vector_t params(1 + nx_ + nu_ + 1);
    params << t, x, u, impact;

    return thetaTape_->value(th, params);
  }

  ocs2::vector_t grad_theta(double t,
                            const ocs2::vector_t& x,
                            const ocs2::vector_t& u,
                            const RefPackKnot& /*refk*/,
                            const ocs2::vector_t& theta,
                            const ThetaLayout& layout) const override {
    ocs2::vector_t g = ocs2::vector_t::Zero(theta.size());
    if (!isActive_(t)) return g;

    const double impact = impact_(t);
    const ocs2::vector_t th = thetaLocal_(theta, layout);

    ocs2::vector_t params(1 + nx_ + nu_ + 1);
    params << t, x, u, impact;

    const ocs2::vector_t dldth = thetaTape_->grad_theta(th, params);

    const long off = static_cast<long>(layout.off_torque(contactIndex_));
    const long m = static_cast<long>(ThetaLayout::torque_dim_per_leg);
    g.segment(off, m) += dldth;
    return g;
  }

  ocs2::vector_t grad_x(double t,
                        const ocs2::vector_t& x,
                        const ocs2::vector_t& u,
                        const RefPackKnot& /*refk*/,
                        const ocs2::vector_t& theta,
                        const ThetaLayout& layout) const override {
    if (!isActive_(t)) return ocs2::vector_t::Zero(nx_);

    const double impact = impact_(t);
    const ocs2::vector_t th = thetaLocal_(theta, layout);

    ocs2::vector_t xu(nx_ + nu_);
    xu << x, u;

    // params = [t|impact|theta]
    ocs2::vector_t params(1 + 1 + th.size());
    params << t, impact, th;

    const ocs2::vector_t dldxu = xuTape_->grad_xu(xu, params);
    return dldxu.head(nx_);
  }

 private:
  bool isActive_(double t) const { return refMgr_->getContactFlags(t)[contactIndex_]; }

  double impact_(double t) const { return refMgr_->getSwingTrajectoryPlanner()->getImpactProximityFactor((1 - contactIndex_), t); }

  ocs2::vector_t thetaLocal_(const ocs2::vector_t& theta, const ThetaLayout& layout) const {
    return layout.theta_torque(theta, contactIndex_);
  }

 private:
  int nx_{0}, nu_{0};
  size_t contactIndex_{0};
  pinocchio::FrameIndex frameID_{0};

  std::vector<size_t> tauActiveIndices_;
  std::string modelFolder_;
  const ocs2::humanoid::SwitchedModelReferenceManager* refMgr_{nullptr};

  std::unique_ptr<TorqueStageThetaAD> thetaTape_;
  std::unique_ptr<TorqueStageXUAD> xuTape_;
};

}  // namespace ocs2::humanoid_cost_matching
