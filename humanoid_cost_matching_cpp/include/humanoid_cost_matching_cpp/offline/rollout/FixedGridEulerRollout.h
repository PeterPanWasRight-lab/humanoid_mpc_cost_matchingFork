#pragma once
#include <ocs2_core/Types.h>
#include <ocs2_core/dynamics/SystemDynamicsBase.h>
#include <ocs2_core/reference/ModeSchedule.h>

#include <functional>
#include <vector>

namespace ocs2::humanoid_cost_matching {

struct FixedGridRolloutResult {
  ocs2::scalar_array_t t;  // N+1
  ocs2::vector_array_t x;  // N+1
};

/**
 * Euler integrator on fixed MPC grid.
 * Within each [t_k, t_{k+1}] segment, split further at modeSchedule.eventTimes to hit switches exactly.
 *
 * Dynamics theta (v1):
 *  - elementwise gains on xdot rows of h_lin_dot (idx 0..2) and h_ang_dot (idx 3..5)
 *    f_theta = f_nom; then scale those 6 entries.
 *
 * Sensitivity recursion:
 *  S = dx / d theta_dyn, where theta_dyn = [theta_hl(3), theta_ha(3)] => dim=6
 *  Euler:
 *    x_{+} = x + h * f_theta(x,u)
 *    S_{+} = S + h * (A_theta * S + B)
 *  where:
 *    A_theta = df_theta/dx = row-scaled version of A_nom
 *    B = df_theta/dtheta_dyn is sparse on the 6 momentum-dot rows.
 */
class FixedGridEulerRollout {
 public:
  FixedGridEulerRollout(ocs2::SystemDynamicsBase& dyn, size_t idxHlinDot, size_t idxHangDot);

  void setMomentumGains(const ocs2::vector_t& theta_hl, const ocs2::vector_t& theta_ha);

  FixedGridRolloutResult run(ocs2::scalar_t t0,
                             const ocs2::vector_t& x0,
                             ocs2::scalar_t dt,
                             size_t N,
                             const ocs2::ModeSchedule& modeSchedule,
                             const std::function<ocs2::vector_t(ocs2::scalar_t, const ocs2::vector_t&)>& u_of_t,
                             std::vector<ocs2::matrix_t>* S_grid  // optional: N+1 mats (nx x 6)
  ) const;

 private:
  static ocs2::scalar_array_t splitByEvents_(ocs2::scalar_t ta, ocs2::scalar_t tb, const ocs2::ModeSchedule& ms);

  ocs2::SystemDynamicsBase& dyn_;
  size_t idxHlinDot_;
  size_t idxHangDot_;

  ocs2::vector_t theta_hl_ = ocs2::vector_t::Ones(3);
  ocs2::vector_t theta_ha_ = ocs2::vector_t::Ones(3);
};

}  // namespace ocs2::humanoid_cost_matching
