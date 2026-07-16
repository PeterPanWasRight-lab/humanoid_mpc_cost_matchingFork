#include "humanoid_cost_matching_cpp/offline/rollout/FixedGridRK4Rollout.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ocs2::humanoid_cost_matching {

FixedGridRK4Rollout::FixedGridRK4Rollout(ocs2::SystemDynamicsBase& dyn, size_t idxHlinDot, size_t idxHangDot)
    : dyn_(dyn), idxHlinDot_(idxHlinDot), idxHangDot_(idxHangDot) {}

void FixedGridRK4Rollout::setMomentumGains(const ocs2::vector_t& theta_hl, const ocs2::vector_t& theta_ha) {
  if (theta_hl.size() != 3 || theta_ha.size() != 3) {
    throw std::runtime_error("[FixedGridRK4Rollout] theta_hl/theta_ha must be size 3.");
  }
  theta_hl_ = theta_hl;
  theta_ha_ = theta_ha;
}

ocs2::scalar_array_t FixedGridRK4Rollout::splitByEvents_(ocs2::scalar_t ta, ocs2::scalar_t tb, const ocs2::ModeSchedule& ms) {
  ocs2::scalar_array_t knots;
  knots.reserve(ms.eventTimes.size() + 2);
  knots.push_back(ta);
  for (const auto& te : ms.eventTimes) {
    if (te > ta && te < tb) knots.push_back(te);
  }
  knots.push_back(tb);
  return knots;
}

FixedGridRolloutResult FixedGridRK4Rollout::run(ocs2::scalar_t t0,
                                                const ocs2::vector_t& x0,
                                                ocs2::scalar_t dt,
                                                size_t N,
                                                const ocs2::ModeSchedule& modeSchedule,
                                                const std::function<ocs2::vector_t(ocs2::scalar_t, const ocs2::vector_t&)>& u_of_t,
                                                std::vector<ocs2::matrix_t>* S_grid) const {
  FixedGridRolloutResult out;
  out.t.resize(N + 1);
  out.x.resize(N + 1);

  out.t[0] = t0;
  out.x[0] = x0;

  const int nx = static_cast<int>(x0.size());
  ocs2::matrix_t S = ocs2::matrix_t::Zero(nx, 6);  // dx/dtheta_dyn

  if (S_grid) {
    S_grid->clear();
    S_grid->reserve(N + 1);
    S_grid->push_back(S);
  }

  auto checkIdxRange = [&](const ocs2::vector_t& v) {
    const long need = static_cast<long>(idxHangDot_ + 3);
    if (v.size() < need) {
      throw std::runtime_error("[FixedGridRK4Rollout] idx out of range.");
    }
  };

  // Evaluate f_theta(x,u), A_theta(x,u), B(x,u) at given (t,x,S)
  auto eval = [&](double t, const ocs2::vector_t& x, const ocs2::matrix_t& S_in, ocs2::vector_t& xdot_out, ocs2::matrix_t& Sdot_out) {
    const ocs2::vector_t u = u_of_t(t, x);

    // nominal flow
    const ocs2::vector_t f_nom = dyn_.computeFlowMap(t, x, u);
    checkIdxRange(f_nom);

    // theta-scaled flow (only 6 momentum-dot entries)
    ocs2::vector_t f_th = f_nom;
    for (int i = 0; i < 3; ++i) f_th(static_cast<long>(idxHlinDot_ + i)) *= theta_hl_(i);
    for (int i = 0; i < 3; ++i) f_th(static_cast<long>(idxHangDot_ + i)) *= theta_ha_(i);

    // A_nom = df/dx
    auto lin = dyn_.linearApproximation(t, x, u);
    ocs2::matrix_t A = lin.dfdx;

    // row scale to match f_th definition
    for (int i = 0; i < 3; ++i) A.row(static_cast<long>(idxHlinDot_ + i)) *= theta_hl_(i);
    for (int i = 0; i < 3; ++i) A.row(static_cast<long>(idxHangDot_ + i)) *= theta_ha_(i);

    // B = df/dtheta_dyn (nx x 6), sparse
    ocs2::matrix_t B = ocs2::matrix_t::Zero(nx, 6);
    for (int i = 0; i < 3; ++i) B(static_cast<long>(idxHlinDot_ + i), i) = f_nom(static_cast<long>(idxHlinDot_ + i));
    for (int i = 0; i < 3; ++i) B(static_cast<long>(idxHangDot_ + i), 3 + i) = f_nom(static_cast<long>(idxHangDot_ + i));

    xdot_out = f_th;
    Sdot_out = A * S_in + B;
  };

  for (size_t k = 0; k < N; ++k) {
    const ocs2::scalar_t tk = t0 + static_cast<ocs2::scalar_t>(k) * dt;
    const ocs2::scalar_t tk1 = tk + dt;

    auto subKnots = splitByEvents_(tk, tk1, modeSchedule);

    ocs2::vector_t x = out.x[k];

    for (size_t si = 0; si + 1 < subKnots.size(); ++si) {
      const ocs2::scalar_t ta = subKnots[si];
      const ocs2::scalar_t tb = subKnots[si + 1];
      const ocs2::scalar_t h = tb - ta;
      if (h <= 0.0) continue;

      // Keep your original fine stepping policy (hmax)
      const double hmax = 0.01;
      const int M = std::max(1, static_cast<int>(std::ceil(h / hmax)));
      const double hs = static_cast<double>(h) / static_cast<double>(M);

      for (int m = 0; m < M; ++m) {
        const double t = static_cast<double>(ta) + static_cast<double>(m) * hs;

        // RK4 stages for x and S
        ocs2::vector_t k1x(nx), k2x(nx), k3x(nx), k4x(nx);
        ocs2::matrix_t k1S(nx, 6), k2S(nx, 6), k3S(nx, 6), k4S(nx, 6);

        eval(t, x, S, k1x, k1S);

        {
          const ocs2::vector_t x2 = x + (hs * 0.5) * k1x;
          const ocs2::matrix_t S2 = S + (hs * 0.5) * k1S;
          eval(t + hs * 0.5, x2, S2, k2x, k2S);
        }

        {
          const ocs2::vector_t x3 = x + (hs * 0.5) * k2x;
          const ocs2::matrix_t S3 = S + (hs * 0.5) * k2S;
          eval(t + hs * 0.5, x3, S3, k3x, k3S);
        }

        {
          const ocs2::vector_t x4 = x + hs * k3x;
          const ocs2::matrix_t S4 = S + hs * k3S;
          eval(t + hs, x4, S4, k4x, k4S);
        }

        // RK4 update
        x = x + (hs / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x);
        S = S + (hs / 6.0) * (k1S + 2.0 * k2S + 2.0 * k3S + k4S);
      }
    }

    out.t[k + 1] = tk1;
    out.x[k + 1] = x;  // exactly on MPC timestep grid
    if (S_grid) S_grid->push_back(S);
  }

  return out;
}

}  // namespace ocs2::humanoid_cost_matching
