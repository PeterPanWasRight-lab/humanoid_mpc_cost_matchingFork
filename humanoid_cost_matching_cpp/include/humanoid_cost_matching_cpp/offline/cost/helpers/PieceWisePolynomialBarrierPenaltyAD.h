#pragma once
#include <ocs2_core/Types.h>
#include <cppad/cppad.hpp>

namespace ocs2::humanoid_cost_matching {

/**
 * AD-safe exact replica of:
 * ocs2::PieceWisePolynomialBarrierPenalty::getValue(t, h)
 *
 * if (h <= 0):
 *   (0.5*h^2 - delta*h/2 + delta^2/6) * mu
 * else if (0 < h < delta):
 *   (-h^3/(6*delta) + 0.5*h^2 - delta*h/2 + delta^2/6) * mu
 * else:
 *   0
 *
 * Boundary behavior matches exactly:
 *  - h==0   -> first branch (non-zero)
 *  - h==delta -> else branch (0)
 */
struct PiecewiseBarrierConfig {
  ocs2::scalar_t mu{1.0};
  ocs2::scalar_t delta{1e-3};
};

template <typename AD>
inline AD piecewiseBarrierValueExact(const AD& h, const AD& mu, const AD& delta) {
  const AD half = AD(0.5);
  const AD one_over_6 = AD(1.0 / 6.0);

  // h <= 0
  const AD penalty_le0 = half * h * h - delta * h / AD(2.0) + (delta * delta) * one_over_6;
  const AD val_le0 = penalty_le0 * mu;

  // 0 < h < delta
  const AD penalty_mid = -h * h * h / (AD(6.0) * delta) + half * h * h - delta * h / AD(2.0) + (delta * delta) * one_over_6;
  const AD val_mid = penalty_mid * mu;

  const AD val_else = AD(0.0);

  // if (h < delta) -> val_mid else -> 0    (only relevant when h>0)
  const AD mid_or_else = CppAD::CondExpLt(h, delta, val_mid, val_else);

  // if (h > 0) -> mid_or_else else -> val_le0   (exactly matches h<=0)
  return CppAD::CondExpGt(h, AD(0.0), mid_or_else, val_le0);
}

}  // namespace ocs2::humanoid_cost_matching
