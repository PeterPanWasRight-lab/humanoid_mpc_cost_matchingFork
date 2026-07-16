#pragma once

#include <ocs2_core/Types.h>
#include <stdexcept>
#include <string>

namespace ocs2::humanoid_cost_matching {

/**
 * theta layout (v2, extensible)
 *
 * Order:
 * [ theta_hl(3), theta_ha(3), theta_q(nx), theta_r(nu), theta_qf(nx),
 * theta_base(baseDim), theta_com(comDim),
 * theta_swing(foot0,12), theta_swing(foot1,12),
 * theta_torque(leg0,6), theta_torque(leg1,6) ]
 *
 * Notes:
 * - Keep hl/ha first (dyn params first, then all cost params).
 * - base/com/torque dims are runtime (passed into make()).
 * - swing: numSwingFeet=2 by default; per-foot dim fixed to 12.
 * - torque: numTorqueLeg=2 by default; per-leg dim fixed to 6.
 */
struct ThetaLayout {
  using vector_t = ocs2::vector_t;

  size_t nx = 0;
  size_t nu = 0;

  // runtime dims for extra terms
  size_t baseDim = 0;    // e.g. 12 if you use EndEffectorKinematics errors(12)
  size_t comDim = 0;     // e.g. 2 for ICP cost
  size_t torqueDim = 0;  // total torque block dim (numTorqueLeg * 6)

  static constexpr int swing_dim_per_foot = 12;
  size_t numSwingFeet = 2;

  static constexpr int torque_dim_per_leg = 6;
  size_t numTorqueLeg = 2;

  // offsets
  size_t off_hl = 0;       // 3
  size_t off_ha = 0;       // 3
  size_t off_q = 0;        // nx
  size_t off_r = 0;        // nu
  size_t off_qf = 0;       // nx
  size_t off_base = 0;     // baseDim
  size_t off_com = 0;      // comDim
  size_t off_swing0 = 0;   // numSwingFeet * 12
  size_t off_torque0 = 0;  // numTorqueLeg * 6

  size_t off_swing(size_t foot) const {
    if (foot >= numSwingFeet) throw std::runtime_error("[ThetaLayout] swing foot index out of range");
    return off_swing0 + foot * swing_dim_per_foot;
  }

  size_t off_torque(size_t leg) const {
    if (leg >= numTorqueLeg) throw std::runtime_error("[ThetaLayout] torque leg index out of range");
    return off_torque0 + leg * torque_dim_per_leg;
  }

  static ThetaLayout make(size_t nx_, size_t nu_, size_t baseDim_, size_t comDim_, size_t numSwingFeet_ = 2, size_t numTorqueLeg_ = 2) {
    ThetaLayout L;
    L.nx = nx_;
    L.nu = nu_;
    L.baseDim = baseDim_;
    L.comDim = comDim_;
    L.numSwingFeet = numSwingFeet_;
    L.numTorqueLeg = numTorqueLeg_;

    // dyn first
    L.off_hl = 0;
    L.off_ha = L.off_hl + 3;

    // tracking
    L.off_q = L.off_ha + 3;
    L.off_r = L.off_q + L.nx;
    L.off_qf = L.off_r + L.nu;

    // extra costs
    L.off_base = L.off_qf + L.nx;
    L.off_com = L.off_base + L.baseDim;
    L.off_swing0 = L.off_com + L.comDim;
    L.off_torque0 = L.off_swing0 + L.numSwingFeet * swing_dim_per_foot;

    return L;
  }

  size_t dim_total() const {
    return 3 + 3 + nx + nu + nx + baseDim + comDim + (numSwingFeet * swing_dim_per_foot) + (numTorqueLeg * torque_dim_per_leg);
  }

  void checkSize(const vector_t& theta) const {
    if (static_cast<size_t>(theta.size()) != dim_total()) {
      throw std::runtime_error("[ThetaLayout] theta dim mismatch. got=" + std::to_string(theta.size()) +
                               " expected=" + std::to_string(dim_total()));
    }
  }

  // slices
  vector_t theta_hl(const vector_t& th) const { return th.segment(static_cast<long>(off_hl), 3); }
  vector_t theta_ha(const vector_t& th) const { return th.segment(static_cast<long>(off_ha), 3); }

  vector_t theta_q(const vector_t& th) const { return th.segment(static_cast<long>(off_q), static_cast<long>(nx)); }
  vector_t theta_r(const vector_t& th) const { return th.segment(static_cast<long>(off_r), static_cast<long>(nu)); }
  vector_t theta_qf(const vector_t& th) const { return th.segment(static_cast<long>(off_qf), static_cast<long>(nx)); }

  vector_t theta_base(const vector_t& th) const { return th.segment(static_cast<long>(off_base), static_cast<long>(baseDim)); }
  vector_t theta_com(const vector_t& th) const { return th.segment(static_cast<long>(off_com), static_cast<long>(comDim)); }

  vector_t theta_swing(const vector_t& th, size_t foot) const { return th.segment(static_cast<long>(off_swing(foot)), swing_dim_per_foot); }
  vector_t theta_torque(const vector_t& th, size_t leg) const { return th.segment(static_cast<long>(off_torque(leg)), torque_dim_per_leg); }
};

}  // namespace ocs2::humanoid_cost_matching