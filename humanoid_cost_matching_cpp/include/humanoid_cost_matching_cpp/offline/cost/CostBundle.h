#pragma once

#include <memory>
#include <vector>

#include <ocs2_core/Types.h>
#include "humanoid_cost_matching_cpp/offline/cost/ICostTerm.h"

namespace ocs2::humanoid_cost_matching {

/**
 * CostBundle: owns stage terms + terminal terms.
 * Provides "one-line" APIs used by OfflineQEvaluator.
 */
struct CostBundle {
  std::vector<std::unique_ptr<ICostTerm>> stageTerms;
  std::vector<std::unique_ptr<ICostTerm>> terminalTerms;

  double stage_value(double t,
                     const ocs2::vector_t& x,
                     const ocs2::vector_t& u,
                     const RefPackKnot& refk,
                     const ocs2::vector_t& theta,
                     const ThetaLayout& layout) const {
    double s = 0.0;
    for (const auto& term : stageTerms) {
      s += term->value(t, x, u, refk, theta, layout);
    }
    return s;
  }

  ocs2::vector_t stage_grad_theta(double t,
                                  const ocs2::vector_t& x,
                                  const ocs2::vector_t& u,
                                  const RefPackKnot& refk,
                                  const ocs2::vector_t& theta,
                                  const ThetaLayout& layout) const {
    ocs2::vector_t g = ocs2::vector_t::Zero(theta.size());
    for (const auto& term : stageTerms) {
      g.noalias() += term->grad_theta(t, x, u, refk, theta, layout);
    }
    return g;
  }

  ocs2::vector_t stage_grad_x(double t,
                              const ocs2::vector_t& x,
                              const ocs2::vector_t& u,
                              const RefPackKnot& refk,
                              const ocs2::vector_t& theta,
                              const ThetaLayout& layout) const {
    ocs2::vector_t gx = ocs2::vector_t::Zero(x.size());
    for (const auto& term : stageTerms) {
      gx.noalias() += term->grad_x(t, x, u, refk, theta, layout);
    }
    return gx;
  }

  double terminal_value(
      double t, const ocs2::vector_t& x, const RefPackKnot& refk, const ocs2::vector_t& theta, const ThetaLayout& layout) const {
    if (terminalTerms.empty()) return 0.0;
    static const ocs2::vector_t u_empty;
    double s = 0.0;
    for (const auto& term : terminalTerms) {
      s += term->value(t, x, u_empty, refk, theta, layout);
    }
    return s;
  }

  ocs2::vector_t terminal_grad_theta(
      double t, const ocs2::vector_t& x, const RefPackKnot& refk, const ocs2::vector_t& theta, const ThetaLayout& layout) const {
    if (terminalTerms.empty()) return ocs2::vector_t::Zero(theta.size());
    static const ocs2::vector_t u_empty;
    ocs2::vector_t g = ocs2::vector_t::Zero(theta.size());
    for (const auto& term : terminalTerms) {
      g.noalias() += term->grad_theta(t, x, u_empty, refk, theta, layout);
    }
    return g;
  }

  ocs2::vector_t terminal_grad_x(
      double t, const ocs2::vector_t& x, const RefPackKnot& refk, const ocs2::vector_t& theta, const ThetaLayout& layout) const {
    if (terminalTerms.empty()) return ocs2::vector_t::Zero(x.size());
    static const ocs2::vector_t u_empty;
    ocs2::vector_t gx = ocs2::vector_t::Zero(x.size());
    for (const auto& term : terminalTerms) {
      gx.noalias() += term->grad_x(t, x, u_empty, refk, theta, layout);
    }
    return gx;
  }
};

}  // namespace ocs2::humanoid_cost_matching
