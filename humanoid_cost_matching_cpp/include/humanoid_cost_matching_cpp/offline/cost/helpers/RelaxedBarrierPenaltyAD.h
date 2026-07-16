#pragma once

#include <ocs2_core/Types.h>
#include <cppad/cppad.hpp>

namespace ocs2::humanoid_cost_matching {

/** AD-safe exact replica of: ocs2::RelaxedBarrierPenalty::Config */
struct RelaxedBarrierConfig {
  double mu{1.0};
  double delta{1e-3};
};

/**
 * Exact replica of:
 * if (h > delta) return -mu*log(h);
 * else { delta_h=(h-2*delta)/delta; return mu*(-log(delta)+0.5*delta_h^2-0.5); }
 */
template <typename AD>
inline AD relaxedBarrierValueAD(const RelaxedBarrierConfig& cfg, const AD& h) {
  const AD delta = AD(cfg.delta);
  const AD mu = AD(cfg.mu);

  const AD log_branch = -mu * CppAD::log(h);

  const AD delta_h = (h - AD(2.0) * delta) / delta;
  const AD else_branch = mu * (-CppAD::log(delta) + AD(0.5) * delta_h * delta_h - AD(0.5));

  return CppAD::CondExpGt(h, delta, log_branch, else_branch);
}

/**
 * Same function but mu/delta are AD (for tapes where mu/delta are in params vector).
 * This avoids any CppAD::Value / double conversion and keeps gradients correct.
 */
template <typename AD>
inline AD relaxedBarrierValueAD(const AD& mu, const AD& delta, const AD& h) {
  const AD log_branch = -mu * CppAD::log(h);

  const AD delta_h = (h - AD(2.0) * delta) / delta;
  const AD else_branch = mu * (-CppAD::log(delta) + AD(0.5) * delta_h * delta_h - AD(0.5));

  return CppAD::CondExpGt(h, delta, log_branch, else_branch);
}

}  // namespace ocs2::humanoid_cost_matching
