#pragma once

#include <memory>
#include <string>

#include <ocs2_core/Types.h>
#include <ocs2_core/automatic_differentiation/CppAdInterface.h>

namespace ocs2::humanoid_cost_matching {

/**
 * stage cost:
 *   l = (x-xref)^T diag(theta_q) (x-xref) + (u-uref)^T diag(theta_r) (u-uref)
 * theta_qr dim = nx + nu
 * params: [t | x(nx) | u(nu) | xref(nx) | uref(nu)]
 */
class TrackingStageThetaAD {
 public:
  TrackingStageThetaAD(size_t nx, size_t nu, const std::string& modelName, const std::string& modelFolder, bool recompile, bool verbose);

  ocs2::scalar_t value(const ocs2::vector_t& theta_qr, const ocs2::vector_t& params) const;
  ocs2::vector_t grad_theta(const ocs2::vector_t& theta_qr, const ocs2::vector_t& params) const;

 private:
  size_t nx_, nu_;
  std::unique_ptr<ocs2::CppAdInterface> ad_;
};

/**
 * terminal cost:
 *   T = (x-xref)^T diag(theta_qf) (x-xref)
 * theta_qf dim = nx
 * params: [t | x(nx) | xref(nx)]
 */
class TrackingTerminalThetaAD {
 public:
  TrackingTerminalThetaAD(size_t nx, const std::string& modelName, const std::string& modelFolder, bool recompile, bool verbose);

  ocs2::scalar_t value(const ocs2::vector_t& theta_qf, const ocs2::vector_t& params) const;
  ocs2::vector_t grad_theta(const ocs2::vector_t& theta_qf, const ocs2::vector_t& params) const;

 private:
  size_t nx_;
  std::unique_ptr<ocs2::CppAdInterface> ad_;
};

/**
 * Tracking stage XU tape:
 *   ℓ = (x-xNom)^T diag(theta_q) (x-xNom) + (u-uNom)^T diag(theta_r) (u-uNom)
 *
 * variables:
 *   z = [x;u]  (nx+nu)
 *
 * params:
 *   p = [t | xNom | uNom | theta_q | theta_r]
 *   dim = 1 + nx + nu + nx + nu
 */
class TrackingStageXUAD {
 public:
  TrackingStageXUAD(size_t nx, size_t nu, const std::string& modelName, const std::string& modelFolder, bool recompile, bool verbose);

  ocs2::scalar_t value(const ocs2::vector_t& xu, const ocs2::vector_t& params) const;
  ocs2::vector_t grad_xu(const ocs2::vector_t& xu, const ocs2::vector_t& params) const;

 private:
  size_t nx_, nu_;
  std::unique_ptr<ocs2::CppAdInterface> ad_;
};

/**
 * Tracking terminal X tape:
 *   T = (x-xNom)^T diag(theta_qf) (x-xNom)
 *
 * variables:
 *   x  (nx)
 *
 * params:
 *   p = [t | xNom | theta_qf]
 *   dim = 1 + nx + nx
 */
class TrackingTerminalXAD {
 public:
  TrackingTerminalXAD(size_t nx, const std::string& modelName, const std::string& modelFolder, bool recompile, bool verbose);

  ocs2::scalar_t value(const ocs2::vector_t& x, const ocs2::vector_t& params) const;
  ocs2::vector_t grad_x(const ocs2::vector_t& x, const ocs2::vector_t& params) const;

 private:
  size_t nx_;
  std::unique_ptr<ocs2::CppAdInterface> ad_;
};

}  // namespace ocs2::humanoid_cost_matching
