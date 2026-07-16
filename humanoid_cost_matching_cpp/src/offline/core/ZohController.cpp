#include "humanoid_cost_matching_cpp/offline/core/ZohController.h"

namespace ocs2::humanoid_cost_matching {

ZohController::ZohController(ocs2::scalar_array_t timeGrid, ocs2::vector_array_t uSeq)
    : timeGrid_(std::move(timeGrid)), uSeq_(std::move(uSeq)) {
  validateOrThrow_();
}

ZohController::ZohController(ocs2::scalar_t t0, ocs2::scalar_t dt, ocs2::vector_array_t uSeq) : uSeq_(std::move(uSeq)) {
  if (uSeq_.empty()) {
    throw std::runtime_error("[ZohController] uSeq is empty.");
  }
  if (dt <= 0.0) {
    throw std::runtime_error("[ZohController] dt must be positive.");
  }

  const size_t N = uSeq_.size();
  timeGrid_.resize(N + 1);
  for (size_t i = 0; i <= N; ++i) {
    timeGrid_[i] = t0 + static_cast<ocs2::scalar_t>(i) * dt;
  }
  validateOrThrow_();
}

void ZohController::setTrajectory(ocs2::scalar_array_t timeGrid, ocs2::vector_array_t uSeq) {
  timeGrid_ = std::move(timeGrid);
  uSeq_ = std::move(uSeq);
  lastIndex_ = 0;
  validateOrThrow_();
}

void ZohController::clear() {
  timeGrid_.clear();
  uSeq_.clear();
  lastIndex_ = 0;
}

void ZohController::validateOrThrow_() const {
  if (uSeq_.empty()) {
    throw std::runtime_error("[ZohController] uSeq is empty.");
  }
  if (timeGrid_.size() != uSeq_.size() + 1) {
    throw std::runtime_error("[ZohController] timeGrid size must be N+1 where uSeq size is N.");
  }
  for (size_t i = 1; i < timeGrid_.size(); ++i) {
    if (timeGrid_[i] < timeGrid_[i - 1]) {
      throw std::runtime_error("[ZohController] timeGrid must be nondecreasing.");
    }
  }
}

size_t ZohController::timeToIndex_(ocs2::scalar_t t) const {
  const size_t N = uSeq_.size();

  // Clamp outside range
  if (t <= timeGrid_.front()) {
    return 0;
  }
  if (t >= timeGrid_.back()) {
    return N - 1;
  }

  // Fast path: check last interval first
  if (lastIndex_ < N) {
    const auto tL = timeGrid_[lastIndex_];
    const auto tR = timeGrid_[lastIndex_ + 1];
    if (t >= tL && t < tR) {
      return lastIndex_;
    }
    // If moving forward, advance linearly a bit (usually small)
    if (t >= tR) {
      size_t k = lastIndex_;
      while (k + 1 < N && t >= timeGrid_[k + 1]) {
        ++k;
        if (t < timeGrid_[k + 1]) break;
      }
      return k;
    }
  }

  // General: binary search
  auto it = std::upper_bound(timeGrid_.begin(), timeGrid_.end(), t);
  size_t idx = static_cast<size_t>(std::distance(timeGrid_.begin(), it) - 1);
  if (idx >= N) idx = N - 1;
  return idx;
}

ocs2::vector_t ZohController::computeInput(ocs2::scalar_t t, const ocs2::vector_t& /*x*/) {
  const size_t k = timeToIndex_(t);
  lastIndex_ = k;
  return uSeq_[k];
}

void ZohController::concatenate(const ocs2::ControllerBase* otherController, int index, int length) {
  if (otherController == nullptr) {
    throw std::runtime_error("[ZohController] concatenate: otherController is null.");
  }
  if (length <= 0) return;

  const auto* other = dynamic_cast<const ZohController*>(otherController);
  if (!other) {
    throw std::runtime_error("[ZohController] concatenate: otherController is not a ZohController.");
  }
  if (other->empty()) return;

  const int otherN = other->size();
  if (index < 0 || index >= otherN) {
    throw std::runtime_error("[ZohController] concatenate: index out of range.");
  }
  const int end = std::min(index + length, otherN);

  // If this controller is empty, just copy the requested segment.
  if (this->empty()) {
    const int segN = end - index;
    uSeq_.assign(other->uSeq_.begin() + index, other->uSeq_.begin() + end);

    timeGrid_.resize(static_cast<size_t>(segN) + 1);
    // copy corresponding time grid segment [index, end]
    for (int i = 0; i <= segN; ++i) {
      timeGrid_[static_cast<size_t>(i)] = other->timeGrid_[static_cast<size_t>(index + i)];
    }
    lastIndex_ = 0;
    validateOrThrow_();
    return;
  }

  // Non-empty: append requested segment.
  // We try to keep timeGrid nondecreasing. If there is overlap, we allow it (nondecreasing).
  const size_t oldN = uSeq_.size();
  const int segN = end - index;

  // Append u
  uSeq_.insert(uSeq_.end(), other->uSeq_.begin() + index, other->uSeq_.begin() + end);

  // Append time grid points:
  // current timeGrid_ has size oldN+1. We keep its last point as start and then append segN new points.
  // The appended times are other->timeGrid_[index+1 ... end] (segN points).
  const size_t newN = oldN + static_cast<size_t>(segN);
  ocs2::scalar_array_t newGrid;
  newGrid.reserve(newN + 1);

  // keep existing grid
  newGrid.insert(newGrid.end(), timeGrid_.begin(), timeGrid_.end());

  // append other grid points after index
  for (int j = index + 1; j <= end; ++j) {
    newGrid.push_back(other->timeGrid_[static_cast<size_t>(j)]);
  }

  timeGrid_ = std::move(newGrid);
  lastIndex_ = 0;
  validateOrThrow_();
}

}  // namespace ocs2::humanoid_cost_matching
