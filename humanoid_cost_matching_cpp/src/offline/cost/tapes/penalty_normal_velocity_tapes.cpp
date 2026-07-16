#include "humanoid_cost_matching_cpp/offline/cost/tapes/penalty_normal_velocity_tapes.h"

#include <stdexcept>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>

namespace ocs2::humanoid_cost_matching {

NormalVelStageXUAD::NormalVelStageXUAD(size_t nx,
                                       size_t nu,
                                       pinocchio::FrameIndex frameId,
                                       const ocs2::PinocchioInterfaceCppAd& pinocchioAd,
                                       const ocs2::humanoid::MpcRobotModelBase<ad_scalar_t>& robotModelAd,
                                       std::string modelName,
                                       const std::string& modelFolder,
                                       bool recompile,
                                       bool verbose)
    : nx_(nx), nu_(nu), frameId_(frameId), pinocchioAd_(pinocchioAd), robotModelAdPtr_(robotModelAd.clone()) {
  if (!robotModelAdPtr_) throw std::runtime_error("[NormalVelStageXUAD] robotModelAdPtr_ clone failed");

  using ad_scalar_t = ocs2::ad_scalar_t;
  using ad_vector_t = ocs2::ad_vector_t;

  const size_t varDim = nx_ + nu_;  // xu
  const size_t parDim = 2;          // [b, ax_z]

  // Use local copies for pin/robot pointers (raw pointer is copyable).
  const auto frameId_local = frameId_;
  const auto* pinPtr = &pinocchioAd_;
  auto* rmPtr = robotModelAdPtr_.get();

  auto fun = [=](const ad_vector_t& xu, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(1);

    const ad_vector_t x = xu.head(static_cast<long>(nx));
    const ad_vector_t u = xu.tail(static_cast<long>(nu));

    // unpack params
    const ad_scalar_t b = p(0);     // already includes -zVelRef and possibly -posGainZ*zPosRef
    const ad_scalar_t ax_z = p(1);  // = posGainZ or 0

    // pinocchio model/data (copy data, as online cppad constraints do)
    const auto& model = pinPtr->getModel();
    auto data = pinPtr->getData();

    // q,v from robot model
    const ad_vector_t q = rmPtr->getGeneralizedCoordinates(x);
    const ad_vector_t v = rmPtr->getGeneralizedVelocities(x, u);

    pinocchio::forwardKinematics(model, data, q, v);
    pinocchio::updateFramePlacements(model, data);

    // Position of frame in world
    const auto pos = pinocchio::updateFramePlacement(model, data, frameId_local).translation();

    // Linear velocity of frame in WORLD coords (LOCAL_WORLD_ALIGNED)
    const auto vel6 = pinocchio::getFrameVelocity(model, data, frameId_local, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED);
    const auto vlin = vel6.linear();

    // g = b + ax_z * pos_z + 1 * vel_z
    const ad_scalar_t g = b + ax_z * pos.z() + vlin.z();

    y(0) = g;  // NOTE: tape outputs g; term will build 0.5*w*g^2 with exact scaling.
  };

  ad_.reset(new ocs2::CppAdInterface(fun, varDim, parDim, std::move(modelName), modelFolder));
  if (recompile) {
    ad_->createModels(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
  } else {
    ad_->loadModelsIfAvailable(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
  }
}

ocs2::scalar_t NormalVelStageXUAD::value(const ocs2::vector_t& xu, const ocs2::vector_t& params) const {
  return ad_->getFunctionValue(xu, params)(0);
}

ocs2::vector_t NormalVelStageXUAD::grad_xu(const ocs2::vector_t& xu, const ocs2::vector_t& params) const {
  return ad_->getJacobian(xu, params).row(0).transpose();
}

}  // namespace ocs2::humanoid_cost_matching
