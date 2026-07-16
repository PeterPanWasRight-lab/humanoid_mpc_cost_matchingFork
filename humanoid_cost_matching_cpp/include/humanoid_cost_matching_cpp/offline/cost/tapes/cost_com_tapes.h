#pragma once

#include <memory>
#include <string>

#include <ocs2_core/Types.h>
#include <ocs2_core/automatic_differentiation/CppAdInterface.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_pinocchio_interface/PinocchioStateInputMapping.h>

#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"

namespace ocs2::humanoid_cost_matching {

// COM/ICP error dimension: online ICPCost uses XY -> 2
// Each tape owns a writable PinocchioInterfaceCppAd (non-const Data)
// and owns a cloned robot model for AD usage
constexpr int kComErrDim = 2;

/**
 * COM theta tape:
 *   variables = theta_com (kComErrDim)
 *   params    = x (nx)
 */
class ComStageThetaAD {
 public:
  ComStageThetaAD(size_t nx,
                  const ocs2::PinocchioInterfaceCppAd& pinocchioInterfaceCppAd,
                  const ocs2::humanoid::CentroidalMpcRobotModel<ocs2::ad_scalar_t>& mpcRobotModelAd,
                  const std::string& modelName,
                  const std::string& modelFolder,
                  bool recompile,
                  bool verbose);

  ocs2::scalar_t value(const ocs2::vector_t& theta_com, const ocs2::vector_t& x_params) const;
  ocs2::vector_t grad_theta(const ocs2::vector_t& theta_com, const ocs2::vector_t& x_params) const;

 private:
  size_t nx_ = 0;
  ocs2::PinocchioInterfaceCppAd pinCppAd_;
  std::unique_ptr<ocs2::humanoid::CentroidalMpcRobotModel<ocs2::ad_scalar_t>> robotAdPtr_;
  std::unique_ptr<ocs2::CppAdInterface> ad_;
};

/**
 * COM XU tape:
 *   variables = [x; u] (nx+nu)   (u is unused but kept for interface uniformity)
 *   params    = theta_com (kComErrDim)
 *
 * returns grad wrt [x;u] (nx+nu).
 */
class ComStageXUAD {
 public:
  ComStageXUAD(size_t nx,
               size_t nu,
               const ocs2::PinocchioInterfaceCppAd& pinocchioInterfaceCppAd,
               const ocs2::humanoid::CentroidalMpcRobotModel<ocs2::ad_scalar_t>& mpcRobotModelAd,
               const std::string& modelName,
               const std::string& modelFolder,
               bool recompile,
               bool verbose);

  ocs2::scalar_t value(const ocs2::vector_t& xu, const ocs2::vector_t& theta_params) const;
  ocs2::vector_t grad_xu(const ocs2::vector_t& xu, const ocs2::vector_t& theta_params) const;

 private:
  size_t nx_ = 0;
  size_t nu_ = 0;
  ocs2::PinocchioInterfaceCppAd pinCppAd_;
  std::unique_ptr<ocs2::humanoid::CentroidalMpcRobotModel<ocs2::ad_scalar_t>> robotAdPtr_;
  std::unique_ptr<ocs2::CppAdInterface> ad_;
};

}  // namespace ocs2::humanoid_cost_matching
