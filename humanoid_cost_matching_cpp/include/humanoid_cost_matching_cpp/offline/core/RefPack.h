#pragma once

#include <stdexcept>
#include <vector>

#include <ocs2_core/Types.h>
#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"  // contact_flag_t

namespace ocs2::humanoid_cost_matching {

/**
 * RefPackGrid: fixed-grid reference package used by offline Q evaluation.
 * Each knot stores quantities needed by cost/penalty terms:
 *   - xNominal (aka x_ref / desired state)
 *   - uNominal (aka u_ref / nominal input)
 *   - phase, contactFlags
 */
struct RefPackKnot {
  ocs2::scalar_t t = 0.0;
  ocs2::vector_t xNominal;  // nx
  ocs2::vector_t uNominal;  // nu
  double phase = 0.0;
  contact_flag_t contactFlags;  // feet_array_t<bool>
};

struct RefPackGrid {
  ocs2::scalar_t t0 = 0.0;
  ocs2::scalar_t dt = 0.0;
  size_t N = 0;  // horizon steps (u has N, grid has N+1 knots)

  std::vector<RefPackKnot> knots;  // size N+1

  void resize(size_t N_in) {
    N = N_in;
    knots.resize(N + 1);
  }

  size_t size() const { return knots.size(); }

  RefPackKnot& at(size_t k) {
    if (k >= knots.size()) throw std::out_of_range("[RefPackGrid] k out of range");
    return knots[k];
  }
  const RefPackKnot& at(size_t k) const {
    if (k >= knots.size()) throw std::out_of_range("[RefPackGrid] k out of range");
    return knots[k];
  }

  RefPackKnot& operator[](size_t k) { return knots[k]; }
  const RefPackKnot& operator[](size_t k) const { return knots[k]; }
};

}  // namespace ocs2::humanoid_cost_matching
