#pragma once

#include <memory>
#include <string>
#include <utility>

#include <ocs2_core/Types.h>
#include <ocs2_core/automatic_differentiation/CppAdInterface.h>

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_cost_matching_cpp/offline/cost/helpers/PieceWisePolynomialBarrierPenaltyAD.h"

namespace ocs2::humanoid_cost_matching {

class JointStageXAD {
 public:
  JointStageXAD(size_t nx,
                std::pair<ocs2::vector_t, ocs2::vector_t> positionLimits,  // {lower, upper} in joint-space
                PiecewiseBarrierConfig cfg,
                const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>& robotModelAd,
                const std::string& modelName,
                const std::string& modelFolder,
                bool recompile,
                bool verbose);

  ocs2::scalar_t value(const ocs2::vector_t& x) const;
  ocs2::vector_t grad_x(const ocs2::vector_t& x) const;

 private:
  size_t nx_{0};
  std::pair<ocs2::vector_t, ocs2::vector_t> limits_;
  PiecewiseBarrierConfig cfg_;
  const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>* robotAd_{nullptr};
  std::unique_ptr<ocs2::CppAdInterface> ad_;
};

class JointTerminalXAD {
 public:
  JointTerminalXAD(size_t nx,
                   std::pair<ocs2::vector_t, ocs2::vector_t> positionLimits,
                   PiecewiseBarrierConfig cfg,
                   const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>& robotModelAd,
                   const std::string& modelName,
                   const std::string& modelFolder,
                   bool recompile,
                   bool verbose);

  ocs2::scalar_t value(const ocs2::vector_t& x) const;
  ocs2::vector_t grad_x(const ocs2::vector_t& x) const;

 private:
  JointStageXAD impl_;
};

}  // namespace ocs2::humanoid_cost_matching
