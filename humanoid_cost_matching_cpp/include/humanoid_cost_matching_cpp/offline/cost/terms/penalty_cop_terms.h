#pragma once

#include <memory>
#include <string>

#include <ocs2_core/Types.h>

#include "humanoid_cost_matching_cpp/offline/core/RefPack.h"
#include "humanoid_cost_matching_cpp/offline/core/ThetaLayout.h"
#include "humanoid_cost_matching_cpp/offline/cost/ICostTerm.h"
#include "humanoid_cost_matching_cpp/offline/cost/tapes/penalty_cop_tapes.h"

#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

namespace ocs2::humanoid_cost_matching {

class CoPStageTerm final : public ICostTerm {
 public:
  CoPStageTerm(int nx,
               int nu,
               size_t contactIndex,
               const ocs2::humanoid::ContactRectangle& contactRectangle,
               const RelaxedBarrierConfig& barrierCfg,
               const ocs2::PinocchioInterfaceCppAd& pinocchioAd,
               const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>& robotModelAd,
               std::string modelFolder,
               bool recompile,
               bool verbose,
               const ocs2::humanoid::SwitchedModelReferenceManager* refMgr,
               std::string modelNamePrefix = "pen_momentxy")
      : nx_(nx), nu_(nu), contactIndex_(contactIndex), refMgr_(refMgr), modelFolder_(std::move(modelFolder)) {
    if (!refMgr_) throw std::runtime_error("[CoPStageTerm] refMgr is null.");

    // params: [mu, delta, x_min, x_max, y_min, y_max]
    params_.resize(6);
    params_(0) = barrierCfg.mu;
    params_(1) = barrierCfg.delta;

    const auto& b = contactRectangle.getBounds();
    params_(2) = b.x_min;
    params_(3) = b.x_max;
    params_(4) = b.y_min;
    params_(5) = b.y_max;

    const std::string modelName = modelNamePrefix + "_c" + std::to_string(contactIndex_);
    tape_ = std::make_unique<CoPStageXUAD>(static_cast<size_t>(nx_), static_cast<size_t>(nu_), contactIndex_, contactRectangle, barrierCfg,
                                           pinocchioAd, robotModelAd, modelName, modelFolder_, recompile, verbose);
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

    // online: active only when this foot is in contact
    if (!refMgr_->getContactFlags(t)[contactIndex_]) return 0.0;

    ocs2::vector_t xu(nx_ + nu_);
    xu << x, u;
    return tape_->value(xu, params_);
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

    if (!refMgr_->getContactFlags(t)[contactIndex_]) return ocs2::vector_t::Zero(x.size());

    ocs2::vector_t xu(nx_ + nu_);
    xu << x, u;
    const ocs2::vector_t gxu = tape_->grad_xu(xu, params_);
    return gxu.head(nx_);
  }

 private:
  int nx_{0}, nu_{0};
  size_t contactIndex_{0};

  const ocs2::humanoid::SwitchedModelReferenceManager* refMgr_{nullptr};

  std::string modelFolder_;
  ocs2::vector_t params_;  // [mu, delta, x_min, x_max, y_min, y_max]

  std::unique_ptr<CoPStageXUAD> tape_;
};

}  // namespace ocs2::humanoid_cost_matching
