#pragma once

#include <memory>
#include <string>
#include <utility>

#include <ocs2_core/Types.h>

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"

#include "humanoid_cost_matching_cpp/offline/core/RefPack.h"
#include "humanoid_cost_matching_cpp/offline/core/ThetaLayout.h"
#include "humanoid_cost_matching_cpp/offline/cost/ICostTerm.h"
#include "humanoid_cost_matching_cpp/offline/cost/tapes/penalty_joint_tapes.h"

namespace ocs2::humanoid_cost_matching {

class JointStageTerm final : public ICostTerm {
 public:
  JointStageTerm(int nx,
                 std::pair<ocs2::vector_t, ocs2::vector_t> positionLimits,
                 PiecewiseBarrierConfig cfg,
                 const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>& robotModelAd,
                 std::string modelFolder,
                 bool recompile,
                 bool verbose)
      : nx_(nx), modelFolder_(std::move(modelFolder)) {
    tape_ = std::make_unique<JointStageXAD>(static_cast<size_t>(nx_), std::move(positionLimits), cfg, robotModelAd, "pen_joint_stage_x_v1",
                                            modelFolder_, recompile, verbose);
  }

  double value(double /*t*/,
               const ocs2::vector_t& x,
               const ocs2::vector_t& /*u*/,
               const RefPackKnot& /*refk*/,
               const ocs2::vector_t& /*theta*/,
               const ThetaLayout& /*layout*/) const override {
    return tape_->value(x);
  }

  ocs2::vector_t grad_theta(double /*t*/,
                            const ocs2::vector_t& /*x*/,
                            const ocs2::vector_t& /*u*/,
                            const RefPackKnot& /*refk*/,
                            const ocs2::vector_t& theta,
                            const ThetaLayout& /*layout*/) const override {
    return ocs2::vector_t::Zero(theta.size());
  }

  ocs2::vector_t grad_x(double /*t*/,
                        const ocs2::vector_t& x,
                        const ocs2::vector_t& /*u*/,
                        const RefPackKnot& /*refk*/,
                        const ocs2::vector_t& /*theta*/,
                        const ThetaLayout& /*layout*/) const override {
    return tape_->grad_x(x);
  }

 private:
  int nx_{0};
  std::string modelFolder_;
  std::unique_ptr<JointStageXAD> tape_;
};

class JointTerminalTerm final : public ICostTerm {
 public:
  JointTerminalTerm(int nx,
                    std::pair<ocs2::vector_t, ocs2::vector_t> positionLimits,
                    PiecewiseBarrierConfig cfg,
                    const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>& robotModelAd,
                    std::string modelFolder,
                    bool recompile,
                    bool verbose)
      : nx_(nx), modelFolder_(std::move(modelFolder)) {
    tape_ = std::make_unique<JointTerminalXAD>(static_cast<size_t>(nx_), std::move(positionLimits), cfg, robotModelAd,
                                               "pen_joint_terminal_x_v1", modelFolder_, recompile, verbose);
  }

  double value(double /*t*/,
               const ocs2::vector_t& x,
               const ocs2::vector_t& /*u*/,
               const RefPackKnot& /*refk*/,
               const ocs2::vector_t& /*theta*/,
               const ThetaLayout& /*layout*/) const override {
    return tape_->value(x);
  }

  ocs2::vector_t grad_theta(double /*t*/,
                            const ocs2::vector_t& /*x*/,
                            const ocs2::vector_t& /*u*/,
                            const RefPackKnot& /*refk*/,
                            const ocs2::vector_t& theta,
                            const ThetaLayout& /*layout*/) const override {
    return ocs2::vector_t::Zero(theta.size());
  }

  ocs2::vector_t grad_x(double /*t*/,
                        const ocs2::vector_t& x,
                        const ocs2::vector_t& /*u*/,
                        const RefPackKnot& /*refk*/,
                        const ocs2::vector_t& /*theta*/,
                        const ThetaLayout& /*layout*/) const override {
    return tape_->grad_x(x);
  }

 private:
  int nx_{0};
  std::string modelFolder_;
  std::unique_ptr<JointTerminalXAD> tape_;
};

}  // namespace ocs2::humanoid_cost_matching
