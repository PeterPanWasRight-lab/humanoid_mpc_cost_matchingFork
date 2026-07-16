#pragma once

#include <ocs2_core/Types.h>
#include "humanoid_cost_matching_cpp/offline/core/RefPack.h"
#include "humanoid_cost_matching_cpp/offline/core/ThetaLayout.h"

namespace ocs2::humanoid_cost_matching {

/**
 * ICostTerm: one pluggable cost term (stage or terminal).
 *
 * Design choice (for simplicity & clarity):
 * - term knows how to read its theta slice from the FULL theta vector (via ThetaLayout).
 * - term returns:
 *    value(t,x,u,refk,theta,layout)
 *    grad_theta_full (same dim as theta)
 *    grad_x (nx)   -- for chain: dℓ/dx^T * S
 *
 * For terms that do not depend on x, return grad_x = zeros(nx).
 * For terms that have no theta parameters, return grad_theta_full = zeros(dim_total).
 */
class ThetaLayout;  // fwd (your existing layout)

class ICostTerm {
 public:
  virtual ~ICostTerm() = default;

  virtual double value(double t,
                       const ocs2::vector_t& x,
                       const ocs2::vector_t& u,
                       const RefPackKnot& refk,
                       const ocs2::vector_t& theta,
                       const ThetaLayout& layout) const = 0;

  // full-dim gradient (same size as theta)
  virtual ocs2::vector_t grad_theta(double t,
                                    const ocs2::vector_t& x,
                                    const ocs2::vector_t& u,
                                    const RefPackKnot& refk,
                                    const ocs2::vector_t& theta,
                                    const ThetaLayout& layout) const = 0;

  // dℓ/dx (nx)
  virtual ocs2::vector_t grad_x(double t,
                                const ocs2::vector_t& x,
                                const ocs2::vector_t& u,
                                const RefPackKnot& refk,
                                const ocs2::vector_t& theta,
                                const ThetaLayout& layout) const = 0;
};

}  // namespace ocs2::humanoid_cost_matching
