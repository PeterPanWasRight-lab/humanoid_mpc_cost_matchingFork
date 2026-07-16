#pragma once

#include <algorithm>
#include <stdexcept>
#include <utility>

#include <ocs2_core/Types.h>
#include <ocs2_core/control/ControllerBase.h>
#include <ocs2_core/control/ControllerType.h>

namespace ocs2::humanoid_cost_matching {

/**
 * Zero-order-hold controller over a discrete (time grid, input sequence).
 *
 * timeGrid: size N+1 (nondecreasing)
 * uSeq:     size N
 *
 * For t in [timeGrid[k], timeGrid[k+1]) -> returns uSeq[k].
 * Clamp: t <= timeGrid[0] -> uSeq[0],  t >= timeGrid[N] -> uSeq[N-1]
 *
 * Robust to rollout internal step size not matching control dt.
 */
class ZohController final : public ocs2::ControllerBase {
 public:
  ZohController() = default;

  /** Construct from explicit time grid (preferred). */
  ZohController(ocs2::scalar_array_t timeGrid, ocs2::vector_array_t uSeq);

  /** Fallback construct from (t0, dt, uSeq): generates uniform time grid. */
  ZohController(ocs2::scalar_t t0, ocs2::scalar_t dt, ocs2::vector_array_t uSeq);

  ~ZohController() override = default;

  // --- ControllerBase interface ---
  ocs2::vector_t computeInput(ocs2::scalar_t t, const ocs2::vector_t& x) override;

  void concatenate(const ocs2::ControllerBase* otherController, int index, int length) override;

  int size() const override { return static_cast<int>(uSeq_.size()); }

  ocs2::ControllerType getType() const override { return ocs2::ControllerType::FEEDFORWARD; }

  void clear() override;

  bool empty() const override { return uSeq_.empty() || timeGrid_.empty(); }

  ocs2::ControllerBase* clone() const override { return new ZohController(*this); }

  // --- helpers ---
  void setTrajectory(ocs2::scalar_array_t timeGrid, ocs2::vector_array_t uSeq);

  const ocs2::scalar_array_t& timeGrid() const { return timeGrid_; }
  const ocs2::vector_array_t& uSeq() const { return uSeq_; }

 private:
  void validateOrThrow_() const;
  size_t timeToIndex_(ocs2::scalar_t t) const;

 private:
  ocs2::scalar_array_t timeGrid_;  // size N+1
  ocs2::vector_array_t uSeq_;      // size N

  // Optimization: rollout calls are typically time-increasing
  mutable size_t lastIndex_{0};
};

}  // namespace ocs2::humanoid_cost_matching
