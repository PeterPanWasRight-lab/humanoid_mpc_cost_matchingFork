#include "humanoid_cost_matching_cpp/offline/cost/tapes/penalty_cop_tapes.h"

#include <stdexcept>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

namespace ocs2::humanoid_cost_matching {

CoPStageXUAD::CoPStageXUAD(size_t nx,
                           size_t nu,
                           size_t contactIndex,
                           const ocs2::humanoid::ContactRectangle& contactRectangle,
                           const RelaxedBarrierConfig& barrierCfg,
                           const ocs2::PinocchioInterfaceCppAd& pinocchioAd,
                           const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>& robotModelAd,
                           std::string modelName,
                           const std::string& modelFolder,
                           bool recompile,
                           bool verbose)
    : nx_(nx),
      nu_(nu),
      contactIndex_(contactIndex),
      pinocchioAd_(&pinocchioAd),
      robotModelAd_(&robotModelAd),
      contactRectangle_(contactRectangle),
      barrierCfg_(barrierCfg) {
  if (!pinocchioAd_) throw std::runtime_error("[CoPStageXUAD] pinocchioAd is null");
  if (!robotModelAd_) throw std::runtime_error("[CoPStageXUAD] robotModelAd is null");

  using ad_scalar_t = ocs2::ad_scalar_t;
  using ad_vector_t = ocs2::ad_vector_t;
  using ad_vector3_t = Eigen::Matrix<ad_scalar_t, 3, 1>;

  const size_t varDim = nx_ + nu_;  // xu
  const size_t parDim = 6;          // [mu, delta, x_min, x_max, y_min, y_max]

  const auto* pinPtr = pinocchioAd_;
  const auto* robotPtr = robotModelAd_;
  const size_t nx_loc = nx_;
  const size_t nu_loc = nu_;
  const size_t contactIdx = contactIndex_;
  const RelaxedBarrierConfig barrierCfgLoc = barrierCfg_;

  auto fun = [pinPtr, robotPtr, nx_loc, nu_loc, contactIdx, barrierCfgLoc](const ad_vector_t& xu, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(1);

    const ad_vector_t x = xu.head(static_cast<long>(nx_loc));
    const ad_vector_t u = xu.tail(static_cast<long>(nu_loc));

    // params unpack
    // p(0), p(1) are reserved for mu/delta, but due to CG->double issue,
    // we use barrierCfgLoc (double) directly here, which is consistent with online.
    const ad_scalar_t x_min = p(2);
    const ad_scalar_t x_max = p(3);
    const ad_scalar_t y_min = p(4);
    const ad_scalar_t y_max = p(5);

    // RelaxedBarrier config (double) -> AD(cfg.xxx)
    RelaxedBarrierConfig cfg = barrierCfgLoc;

    // pinocchio model/data (copy data, same as online code)
    const auto& model = pinPtr->getModel();
    auto data = pinPtr->getData();

    // update frame placements from q(state)
    const ad_vector_t q = robotPtr->getGeneralizedCoordinates(x);
    ocs2::humanoid::updateFramePlacements(q, model, data);

    // online uses getContactFrameIndex(pinocchioInterfaceCppAd_, *mpcRobotModelPtr_, contactPointIndex_)
    const pinocchio::FrameIndex frameID = ocs2::humanoid::getContactFrameIndex(*pinPtr, *robotPtr, contactIdx);

    // rotate world->local
    const ad_vector3_t localForce = ocs2::humanoid::rotateVectorWorldToLocal(robotPtr->getContactForce(u, contactIdx), data, frameID);
    const ad_vector3_t localMoments = ocs2::humanoid::rotateVectorWorldToLocal(robotPtr->getContactMoment(u, contactIdx), data, frameID);

    // h in R^4 (replica of ContactMomentXYConstraintCppAd::constraintFunction)
    ad_vector_t h(4);
    h(0) = localMoments.x() - y_min * localForce.z();
    h(1) = -localMoments.x() + y_max * localForce.z();
    h(2) = -localMoments.y() - x_min * localForce.z();
    h(3) = localMoments.y() + x_max * localForce.z();

    // penalty sum: sum_i RelaxedBarrierPenalty(h_i)
    ad_scalar_t cost = ad_scalar_t(0.0);
    for (int i = 0; i < 4; ++i) {
      cost += relaxedBarrierValueAD<ad_scalar_t>(cfg, h(i));
    }
    y(0) = cost;
  };

  ad_.reset(new ocs2::CppAdInterface(fun, varDim, parDim, std::move(modelName), modelFolder));
  if (recompile) {
    ad_->createModels(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
  } else {
    ad_->loadModelsIfAvailable(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
  }
}

ocs2::scalar_t CoPStageXUAD::value(const ocs2::vector_t& xu, const ocs2::vector_t& params) const {
  return ad_->getFunctionValue(xu, params)(0);
}

ocs2::vector_t CoPStageXUAD::grad_xu(const ocs2::vector_t& xu, const ocs2::vector_t& params) const {
  // jac is (1 x (nx+nu))
  return ad_->getJacobian(xu, params).row(0).transpose();
}

}  // namespace ocs2::humanoid_cost_matching
