#pragma once

#include <memory>
#include <string>

#include <ocs2_core/Types.h>
#include <ocs2_core/automatic_differentiation/CppAdInterface.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_common_mpc/contact/ContactRectangle.h"

#include "humanoid_cost_matching_cpp/offline/cost/helpers/RelaxedBarrierPenaltyAD.h"

namespace ocs2::humanoid_cost_matching {

/**
 * CoPStageXUAD soft-constraint penalty (stage):
 *   h(x,u) in R^4  (same as ContactMomentXYConstraintCppAd::constraintFunction)
 *   cost = sum_i RelaxedBarrierPenalty(h_i)
 *
 * variables: xu = [x;u]
 * params:    [mu, delta, x_min, x_max, y_min, y_max]
 */
class CoPStageXUAD {
 public:
  CoPStageXUAD(size_t nx,
               size_t nu,
               size_t contactIndex,
               const ocs2::humanoid::ContactRectangle& contactRectangle,
               const RelaxedBarrierConfig& barrierCfg,
               const ocs2::PinocchioInterfaceCppAd& pinocchioAd,
               const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>& robotModelAd,
               std::string modelName,
               const std::string& modelFolder,
               bool recompile,
               bool verbose);

  ocs2::scalar_t value(const ocs2::vector_t& xu, const ocs2::vector_t& params) const;
  ocs2::vector_t grad_xu(const ocs2::vector_t& xu, const ocs2::vector_t& params) const;  // (nx+nu)

 private:
  size_t nx_{0}, nu_{0};
  size_t contactIndex_{0};

  // we keep refs by pointer (lifetime owned by OfflineQEvaluator)
  const ocs2::PinocchioInterfaceCppAd* pinocchioAd_{nullptr};
  const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>* robotModelAd_{nullptr};

  ocs2::humanoid::ContactRectangle contactRectangle_;
  RelaxedBarrierConfig barrierCfg_;

  std::unique_ptr<ocs2::CppAdInterface> ad_;
};

}  // namespace ocs2::humanoid_cost_matching
