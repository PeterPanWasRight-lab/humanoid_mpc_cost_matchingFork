#include "humanoid_cost_matching_cpp/offline/cost/tapes/penalty_joint_tapes.h"
#include "humanoid_cost_matching_cpp/offline/cost/helpers/PieceWisePolynomialBarrierPenaltyAD.h"

#include <stdexcept>

// Needed for CondExp* on AD type
#include <cppad/cppad.hpp>

namespace ocs2::humanoid_cost_matching {

JointStageXAD::JointStageXAD(size_t nx,
                             std::pair<ocs2::vector_t, ocs2::vector_t> positionLimits,
                             PiecewiseBarrierConfig cfg,
                             const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>& robotModelAd,
                             const std::string& modelName,
                             const std::string& modelFolder,
                             bool recompile,
                             bool verbose)
    : nx_(nx), limits_(std::move(positionLimits)), cfg_(cfg), robotAd_(&robotModelAd) {
  if (!robotAd_) throw std::runtime_error("[JointStageXAD] robotAd is null");
  if (limits_.first.size() != limits_.second.size()) throw std::runtime_error("[JointStageXAD] lower/upper limits size mismatch");
  if (!(cfg_.delta >= 0.0)) throw std::runtime_error("[JointStageXAD] delta must be >= 0 (exactly as online)");

  using ad_scalar_t = ocs2::ad_scalar_t;
  using ad_vector_t = ocs2::ad_vector_t;

  const size_t varDim = nx_;
  const size_t parDim = 0;

  auto fun = [this](const ad_vector_t& x, const ad_vector_t& /*p*/, ad_vector_t& y) {
    y.resize(1);

    const ad_vector_t qj = robotAd_->getJointAngles(x);

    // EXACT ONLINE:
    // upperBoundPositionOffset = upper - q
    // lowerBoundPositionOffset = q - lower
    const ad_vector_t upperOff = limits_.second.cast<ad_scalar_t>() - qj;
    const ad_vector_t lowerOff = qj - limits_.first.cast<ad_scalar_t>();

    const ad_scalar_t mu = ad_scalar_t(cfg_.mu);
    const ad_scalar_t delta = ad_scalar_t(cfg_.delta);

    ad_scalar_t cost = ad_scalar_t(0.0);

    for (long i = 0; i < upperOff.size(); ++i) {
      cost += piecewiseBarrierValueExact(upperOff(i), mu, delta);
      cost += piecewiseBarrierValueExact(lowerOff(i), mu, delta);
    }

    // online offset_ is currently 0.0
    y(0) = cost;
  };

  ad_.reset(new ocs2::CppAdInterface(fun, varDim, parDim, modelName, modelFolder));
  if (recompile) {
    ad_->createModels(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
  } else {
    ad_->loadModelsIfAvailable(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
  }
}

ocs2::scalar_t JointStageXAD::value(const ocs2::vector_t& x) const {
  ocs2::vector_t p;
  return ad_->getFunctionValue(x, p)(0);
}

ocs2::vector_t JointStageXAD::grad_x(const ocs2::vector_t& x) const {
  ocs2::vector_t p;
  const ocs2::matrix_t J = ad_->getJacobian(x, p);  // 1 x nx
  return J.row(0).transpose();
}

// ---------------- JointTerminalXAD ----------------

JointTerminalXAD::JointTerminalXAD(size_t nx,
                                   std::pair<ocs2::vector_t, ocs2::vector_t> positionLimits,
                                   PiecewiseBarrierConfig cfg,
                                   const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>& robotModelAd,
                                   const std::string& modelName,
                                   const std::string& modelFolder,
                                   bool recompile,
                                   bool verbose)
    : impl_(nx, std::move(positionLimits), cfg, robotModelAd, modelName, modelFolder, recompile, verbose) {}

ocs2::scalar_t JointTerminalXAD::value(const ocs2::vector_t& x) const {
  return impl_.value(x);
}
ocs2::vector_t JointTerminalXAD::grad_x(const ocs2::vector_t& x) const {
  return impl_.grad_x(x);
}

}  // namespace ocs2::humanoid_cost_matching
