#include "humanoid_cost_matching_cpp/offline/cost/tapes/penalty_zero_velocity_tapes.h"
#include <stdexcept>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <ocs2_robotic_tools/common/RotationTransforms.h>  // rotationMatrixDistanceToPlane

namespace ocs2::humanoid_cost_matching {

ZeroVelocityStageXUAD::ZeroVelocityStageXUAD(size_t nx,
                                             size_t nu,
                                             pinocchio::FrameIndex frameId,
                                             double w,
                                             double posGainZ,
                                             double oriGain,
                                             const ocs2::PinocchioInterfaceCppAd& pinocchioAd,
                                             const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>& robotModelAd,
                                             std::string modelName,
                                             const std::string& modelFolder,
                                             bool recompile,
                                             bool verbose)
    : nx_(nx),
      nu_(nu),
      frameId_(frameId),
      w_(w),
      posGainZ_(posGainZ),
      oriGain_(oriGain),
      pinocchioAd_(pinocchioAd),
      robotModelAdPtr_(robotModelAd.clone()) {
  if (!robotModelAdPtr_) {
    throw std::runtime_error("[ZeroVelocityStageXUAD] robotModelAd.clone() returned null");
  }

  using ad_scalar_t = ocs2::ad_scalar_t;
  using ad_vector_t = ocs2::ad_vector_t;
  using ad_matrix_t = ocs2::ad_matrix_t;
  using ad_vector3_t = Eigen::Matrix<ad_scalar_t, 3, 1>;
  using ad_matrix3_t = Eigen::Matrix<ad_scalar_t, 3, 3>;
  using ad_vector6_t = Eigen::Matrix<ad_scalar_t, 6, 1>;
  using ad_matrix6_t = Eigen::Matrix<ad_scalar_t, 6, 6>;

  const size_t varDim = nx_ + nu_;  // xu
  const size_t parDim = 1;          // [footRefHeightZ]

  auto fun = [this](const ad_vector_t& xu, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(1);

    const ad_vector_t x = xu.head(static_cast<long>(nx_));
    const ad_vector_t u = xu.tail(static_cast<long>(nu_));

    // params
    const ad_scalar_t footRefHeight = p(0);

    // baked constants -> AD
    const ad_scalar_t w = ad_scalar_t(w_);
    const ad_scalar_t posGainZ = ad_scalar_t(posGainZ_);
    const ad_scalar_t oriGain = ad_scalar_t(oriGain_);

    // build Ax/Av/b EXACTLY like online eeZeroVelConConfig + ZeroVelocityConstraintCppAd::getValue
    ad_matrix6_t Ax = ad_matrix6_t::Zero();
    ad_matrix6_t Av = ad_matrix6_t::Identity();
    ad_vector6_t b = ad_vector6_t::Zero();

    Ax(2, 2) = posGainZ;
    Ax.block(3, 3, 3, 3).diagonal().array() = oriGain;

    // online: config.b[2] = -config.Ax(2,2) * footRefHeightZ
    b(2) = -Ax(2, 2) * footRefHeight;

    // pinocchio model/data (copy data)
    const auto& model = pinocchioAd_.getModel();
    auto data = pinocchioAd_.getData();

    // q, v
    // NOTE: getGeneralizedVelocities(x,u) is non-const => OK because robotModelAdPtr_ is a clone (writable)
    const ad_vector_t q = robotModelAdPtr_->getGeneralizedCoordinates(x);
    const ad_vector_t v = robotModelAdPtr_->getGeneralizedVelocities(x, u);

    // FK + jacobians
    pinocchio::forwardKinematics(model, data, q, v);
    pinocchio::computeJointJacobians(model, data, q);
    pinocchio::updateFramePlacements(model, data);

    // Frame jacobian in LOCAL_WORLD_ALIGNED
    const auto rf = pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED;
    ad_matrix_t J = ad_matrix_t::Zero(6, model.nv);
    pinocchio::getFrameJacobian(model, data, frameId_, rf, J);

    // EE twist
    const ad_vector6_t eeTwist = (J * v).template cast<ad_scalar_t>();

    // foot pose: position + orientation error wrt world normal (0,0,1)
    const ad_vector3_t pos = data.oMf[frameId_].translation();
    const ad_matrix3_t R = data.oMf[frameId_].rotation();

    const Eigen::Matrix<ad_scalar_t, 3, 1> n_world(ad_scalar_t(0.0), ad_scalar_t(0.0), ad_scalar_t(1.0));
    const Eigen::Matrix<ad_scalar_t, 3, 1> oriErr = ocs2::rotationMatrixDistanceToPlane(R, n_world);

    ad_vector6_t footPose;
    footPose << pos, oriErr;

    // g = b + Ax*pose + Av*twist
    const ad_vector6_t g = b + Ax * footPose + Av * eeTwist;

    // smooth penalty: 0.5*w*||g||^2
    y(0) = ad_scalar_t(0.5) * w * g.dot(g);
  };

  ad_.reset(new ocs2::CppAdInterface(fun, varDim, parDim, std::move(modelName), modelFolder));
  if (recompile) {
    ad_->createModels(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
  } else {
    ad_->loadModelsIfAvailable(ocs2::CppAdInterface::ApproximationOrder::First, verbose);
  }
}

ocs2::scalar_t ZeroVelocityStageXUAD::value(const ocs2::vector_t& xu, const ocs2::vector_t& params) const {
  return ad_->getFunctionValue(xu, params)(0);
}

ocs2::vector_t ZeroVelocityStageXUAD::grad_xu(const ocs2::vector_t& xu, const ocs2::vector_t& params) const {
  return ad_->getJacobian(xu, params).row(0).transpose();
}

}  // namespace ocs2::humanoid_cost_matching
