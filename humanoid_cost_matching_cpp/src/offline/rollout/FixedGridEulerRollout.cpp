#include "humanoid_cost_matching_cpp/offline/rollout/FixedGridEulerRollout.h"

#include <stdexcept>

namespace ocs2::humanoid_cost_matching {

FixedGridEulerRollout::FixedGridEulerRollout(ocs2::SystemDynamicsBase& dyn, size_t idxHlinDot, size_t idxHangDot)
    : dyn_(dyn), idxHlinDot_(idxHlinDot), idxHangDot_(idxHangDot) {}

void FixedGridEulerRollout::setMomentumGains(const ocs2::vector_t& theta_hl, const ocs2::vector_t& theta_ha) {
  if (theta_hl.size() != 3 || theta_ha.size() != 3) {
    throw std::runtime_error("[FixedGridEulerRollout] theta_hl/theta_ha must be size 3.");
  }
  theta_hl_ = theta_hl;
  theta_ha_ = theta_ha;
}

ocs2::scalar_array_t FixedGridEulerRollout::splitByEvents_(ocs2::scalar_t ta, ocs2::scalar_t tb, const ocs2::ModeSchedule& ms) {
  ocs2::scalar_array_t knots;
  knots.reserve(ms.eventTimes.size() + 2);
  knots.push_back(ta);
  for (const auto& te : ms.eventTimes) {
    if (te > ta && te < tb) knots.push_back(te);
  }
  knots.push_back(tb);
  return knots;
}

FixedGridRolloutResult FixedGridEulerRollout::run(ocs2::scalar_t t0,
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

  const int nx = x0.size();
  ocs2::matrix_t S = ocs2::matrix_t::Zero(nx, 6);  // dx/dtheta_dyn

  if (S_grid) {
    S_grid->clear();
    S_grid->reserve(N + 1);
    S_grid->push_back(S);
  }

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

      const double hmax = 0.001;
      int M = std::max(1, (int)std::ceil(h / hmax));
      double hs = h / M;

      for (int m = 0; m < M; ++m) {  // set even smaller steps within [ta, tb] to improve intergration accuracy
        double t = ta + m * hs;

        const ocs2::vector_t u = u_of_t(t, x);

        // nominal flow
        const ocs2::vector_t f_nom = dyn_.computeFlowMap(t, x, u);

        // theta-scaled flow (only 6 momentum-dot entries)
        ocs2::vector_t f_th = f_nom;
        if (f_th.size() < static_cast<long>(idxHangDot_ + 3)) {
          throw std::runtime_error("[FixedGridEulerRollout] idx out of range.");
        }
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

        // Euler step
        x = x + hs * f_th;
        S = S + hs * (A * S + B);
      }
    }

    out.t[k + 1] = tk1;
    out.x[k + 1] = x;  // ensure output state is exactly on the timegrid of MPC timestep
    if (S_grid) S_grid->push_back(S);
  }

  return out;
}

}  // namespace ocs2::humanoid_cost_matching
