#pragma once

#include <memory>
#include <string>

#include <ocs2_core/Types.h>

#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"

#include "humanoid_cost_matching_cpp/offline/core/RefPack.h"
#include "humanoid_cost_matching_cpp/offline/core/ThetaLayout.h"
#include "humanoid_cost_matching_cpp/offline/cost/ICostTerm.h"
#include "humanoid_cost_matching_cpp/offline/cost/tapes/penalty_collision_tapes.h"

namespace ocs2::humanoid_cost_matching {

class CollisionStageTerm final : public ICostTerm {
 public:
  CollisionStageTerm(int nx,
                     const FootCollisionConfigOffline& cfg,
                     const PiecewiseBarrierConfig& penCfg,
                     const ocs2::PinocchioInterfaceCppAd& pinocchioAd,
                     const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>& robotModelAd,
                     std::string modelFolder,
                     bool recompile,
                     bool verbose,
                     const ocs2::humanoid::SwitchedModelReferenceManager* refMgr)
      : nx_(nx), refMgr_(refMgr), modelFolder_(std::move(modelFolder)) {
    if (!refMgr_) throw std::runtime_error("[CollisionStageTerm] refMgr is null");
    tape_ = std::make_unique<CollisionStageXAD>(static_cast<size_t>(nx_), cfg, penCfg, pinocchioAd, robotModelAd,
                                                "pen_collision_stage_x_v1", modelFolder_, recompile, verbose);
  }

  double value(double t,
               const ocs2::vector_t& x,
               const ocs2::vector_t& /*u*/,
               const RefPackKnot& /*refk*/,
               const ocs2::vector_t& /*theta*/,
               const ThetaLayout& /*layout*/) const override {
    // EXACT online isActive:
    // active unless both feet are in contact
    auto cf = refMgr_->getContactFlags(t);
    if (cf[0] && cf[1]) return 0.0;
    return tape_->value(x);
  }

  ocs2::vector_t grad_theta(double,
                            const ocs2::vector_t&,
                            const ocs2::vector_t&,
                            const RefPackKnot&,
                            const ocs2::vector_t& theta,
                            const ThetaLayout&) const override {
    return ocs2::vector_t::Zero(theta.size());
  }

  ocs2::vector_t grad_x(double t,
                        const ocs2::vector_t& x,
                        const ocs2::vector_t& /*u*/,
                        const RefPackKnot& /*refk*/,
                        const ocs2::vector_t& /*theta*/,
                        const ThetaLayout& /*layout*/) const override {
    auto cf = refMgr_->getContactFlags(t);
    if (cf[0] && cf[1]) return ocs2::vector_t::Zero(x.size());
    return tape_->grad_x(x);
  }

 private:
  int nx_{0};
  const ocs2::humanoid::SwitchedModelReferenceManager* refMgr_{nullptr};
  std::string modelFolder_;
  std::unique_ptr<CollisionStageXAD> tape_;
};

class CollisionTerminalTerm final : public ICostTerm {
 public:
  CollisionTerminalTerm(int nx,
                        const FootCollisionConfigOffline& cfg,
                        const PiecewiseBarrierConfig& penCfg,
                        const ocs2::PinocchioInterfaceCppAd& pinocchioAd,
                        const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>& robotModelAd,
                        std::string modelFolder,
                        bool recompile,
                        bool verbose,
                        const ocs2::humanoid::SwitchedModelReferenceManager* refMgr)
      : nx_(nx), refMgr_(refMgr), modelFolder_(std::move(modelFolder)) {
    if (!refMgr_) throw std::runtime_error("[CollisionTerminalTerm] refMgr is null");
    tape_ = std::make_unique<CollisionTerminalXAD>(static_cast<size_t>(nx_), cfg, penCfg, pinocchioAd, robotModelAd,
                                                   "pen_collision_terminal_x_v1", modelFolder_, recompile, verbose);
  }

  double value(double t,
               const ocs2::vector_t& x,
               const ocs2::vector_t& /*u*/,
               const RefPackKnot& /*refk*/,
               const ocs2::vector_t& /*theta*/,
               const ThetaLayout& /*layout*/) const override {
    auto cf = refMgr_->getContactFlags(t);
    if (cf[0] && cf[1]) return 0.0;
    return tape_->value(x);
  }

  ocs2::vector_t grad_theta(double,
                            const ocs2::vector_t&,
                            const ocs2::vector_t&,
                            const RefPackKnot&,
                            const ocs2::vector_t& theta,
                            const ThetaLayout&) const override {
    return ocs2::vector_t::Zero(theta.size());
  }

  ocs2::vector_t grad_x(double t,
                        const ocs2::vector_t& x,
                        const ocs2::vector_t& /*u*/,
                        const RefPackKnot& /*refk*/,
                        const ocs2::vector_t& /*theta*/,
                        const ThetaLayout& /*layout*/) const override {
    auto cf = refMgr_->getContactFlags(t);
    if (cf[0] && cf[1]) return ocs2::vector_t::Zero(x.size());
    return tape_->grad_x(x);
  }

 private:
  int nx_{0};
  const ocs2::humanoid::SwitchedModelReferenceManager* refMgr_{nullptr};
  std::string modelFolder_;
  std::unique_ptr<CollisionTerminalXAD> tape_;
};

}  // namespace ocs2::humanoid_cost_matching
