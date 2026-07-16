#pragma once

#include <memory>
#include <string>
#include <vector>

#include <ocs2_core/Types.h>
#include <ocs2_core/automatic_differentiation/CppAdInterface.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <pinocchio/algorithm/jacobian.hpp>

#include "humanoid_common_mpc/common/MpcRobotModelBase.h"

namespace ocs2::humanoid_cost_matching {

/**
 * Torque stage theta tape (learn weights)
 *   variables = theta_torque_local (m)   // NOTE: this is "weights" (NOT sqrtWeights)
 *   params    = [t | x(nx) | u(nu) | impactProximityScaler(1)]
 *
 * Cost matches online ExternalTorqueQuadraticCostAD:
 *   r_i = tau_i * sqrtWeight_i * midSwing
 *   l   = 0.5 * sum_i r_i^2 = 0.5 * midSwing^2 * sum_i (weight_i * tau_i^2)
 */
class TorqueStageThetaAD {
 public:
  TorqueStageThetaAD(size_t nx,
                     size_t nu,
                     size_t contactIndex,
                     pinocchio::FrameIndex frameID,
                     std::vector<size_t> tauActiveIndices,  // indices into tauExt (already 6+jointIndex)
                     const ocs2::PinocchioInterfaceCppAd& pinocchioCppAd,
                     const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>& robotModelAd,
                     const std::string& modelName,
                     const std::string& modelFolder,
                     bool recompile,
                     bool verbose);

  ocs2::scalar_t value(const ocs2::vector_t& theta, const ocs2::vector_t& params) const;
  ocs2::vector_t grad_theta(const ocs2::vector_t& theta, const ocs2::vector_t& params) const;

 private:
  size_t nx_{0}, nu_{0};
  size_t contactIndex_{0};
  pinocchio::FrameIndex frameID_{0};
  std::vector<size_t> tauActiveIndices_;

  ocs2::PinocchioInterfaceCppAd pinCppAd_;
  std::unique_ptr<ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>> robotAdPtr_;
  std::unique_ptr<ocs2::CppAdInterface> ad_;
};

/**
 * Torque stage XU tape
 *   variables = [x;u] (nx+nu)
 *   params    = [t | impactProximityScaler(1) | theta_torque_local(m)]
 *
 * Returns d l / d [x;u], caller uses head(nx) for d l / d x.
 */
class TorqueStageXUAD {
 public:
  TorqueStageXUAD(size_t nx,
                  size_t nu,
                  size_t contactIndex,
                  pinocchio::FrameIndex frameID,
                  std::vector<size_t> tauActiveIndices,
                  const ocs2::PinocchioInterfaceCppAd& pinocchioCppAd,
                  const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>& robotModelAd,
                  const std::string& modelName,
                  const std::string& modelFolder,
                  bool recompile,
                  bool verbose);

  ocs2::scalar_t value(const ocs2::vector_t& xu, const ocs2::vector_t& params) const;
  ocs2::vector_t grad_xu(const ocs2::vector_t& xu, const ocs2::vector_t& params) const;

 private:
  size_t nx_{0}, nu_{0};
  size_t contactIndex_{0};
  pinocchio::FrameIndex frameID_{0};
  std::vector<size_t> tauActiveIndices_;

  ocs2::PinocchioInterfaceCppAd pinCppAd_;
  std::unique_ptr<ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>> robotAdPtr_;
  std::unique_ptr<ocs2::CppAdInterface> ad_;
};

}  // namespace ocs2::humanoid_cost_matching
