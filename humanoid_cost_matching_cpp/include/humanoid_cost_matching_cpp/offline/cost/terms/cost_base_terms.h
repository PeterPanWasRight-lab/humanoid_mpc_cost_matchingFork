#pragma once

#include <memory>
#include <stdexcept>
#include <string>

#include <ocs2_core/Types.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <pinocchio/multibody/fwd.hpp>

#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>
#include "humanoid_common_mpc/common/MpcRobotModelBase.h"
#include "humanoid_cost_matching_cpp/offline/core/RefPack.h"
#include "humanoid_cost_matching_cpp/offline/cost/ICostTerm.h"
#include "humanoid_cost_matching_cpp/offline/cost/tapes/cost_base_tapes.h"

namespace ocs2::humanoid_cost_matching {

/**
 * One BaseStageTerm = one link/frame.
 * theta_base is laid out as: [link0(12) | link1(12) | ...]
 *
 * taskRef(13) MUST match online:
 * EndEffectorKinematicsQuadraticCost::getReferenceCostElement(xRef,uRef,*eeKinematics)
 */
class BaseStageTerm final : public ICostTerm {
 public:
  BaseStageTerm(int nx,
                int nu,
                size_t linkIndex,
                pinocchio::FrameIndex frameID,
                const ocs2::EndEffectorKinematics<ocs2::scalar_t>& eeKinDouble,
                const ocs2::PinocchioInterfaceCppAd& pinocchioCppAd,
                const ocs2::humanoid::MpcRobotModelBase<ocs2::ad_scalar_t>& robotModelAd,
                std::string modelFolder,
                bool recompile,
                bool verbose)
      : nx_(nx), nu_(nu), linkIndex_(linkIndex), frameID_(frameID), modelFolder_(std::move(modelFolder)) {
    // clone EE kinematics (double) for reference extraction (online-consistent)
    eeKinDoublePtr_.reset(eeKinDouble.clone());
    if (!eeKinDoublePtr_) throw std::runtime_error("[BaseStageTerm] eeKinDouble.clone() returned null");

    thetaTape_.reset(new BaseStageThetaAD(nx_, nu_, frameID_, pinocchioCppAd, robotModelAd,
                                          "base_stage_theta_link" + std::to_string(linkIndex_), modelFolder_, recompile, verbose));

    xuTape_.reset(new BaseStageXUAD(nx_, nu_, frameID_, pinocchioCppAd, robotModelAd, "base_stage_xu_link" + std::to_string(linkIndex_),
                                    modelFolder_, recompile, verbose));
  }

  double value(double t,
               const ocs2::vector_t& x,
               const ocs2::vector_t& u,
               const RefPackKnot& refk,
               const ocs2::vector_t& theta,
               const ThetaLayout& layout) const override {
    const ocs2::vector_t th = thetaLocal_(theta, layout);      // 12
    const ocs2::vector_t taskRef = taskRefFromNominal_(refk);  // 13 (online-consistent)
    const ocs2::vector_t params = packParams_(t, x, u, taskRef);
    return thetaTape_->value(th, params);
  }

  ocs2::vector_t grad_theta(double t,
                            const ocs2::vector_t& x,
                            const ocs2::vector_t& u,
                            const RefPackKnot& refk,
                            const ocs2::vector_t& theta,
                            const ThetaLayout& layout) const override {
    ocs2::vector_t g_full = ocs2::vector_t::Zero(theta.size());

    const ocs2::vector_t th = thetaLocal_(theta, layout);
    const ocs2::vector_t taskRef = taskRefFromNominal_(refk);
    const ocs2::vector_t params = packParams_(t, x, u, taskRef);

    const ocs2::vector_t dldth = thetaTape_->grad_theta(th, params);  // 12

    const long off = static_cast<long>(layout.off_base) + static_cast<long>(linkIndex_) * kBaseErrDim;
    g_full.segment(off, kBaseErrDim) += dldth;
    return g_full;
  }

  ocs2::vector_t grad_x(double t,
                        const ocs2::vector_t& x,
                        const ocs2::vector_t& u,
                        const RefPackKnot& refk,
                        const ocs2::vector_t& theta,
                        const ThetaLayout& layout) const override {
    // variables z=[x;u]
    ocs2::vector_t xu(nx_ + nu_);
    xu << x, u;

    const ocs2::vector_t th = thetaLocal_(theta, layout);
    const ocs2::vector_t taskRef = taskRefFromNominal_(refk);

    // params = [t|taskRef13|theta(12)]
    ocs2::vector_t params(1 + kBaseRefDim + kBaseErrDim);
    params << t, taskRef, th;

    const ocs2::vector_t dldxu = xuTape_->grad_xu(xu, params);  // (nx+nu)
    return dldxu.head(nx_);
  }

 private:
  ocs2::vector_t thetaLocal_(const ocs2::vector_t& theta, const ThetaLayout& layout) const {
    const long off = static_cast<long>(layout.off_base) + static_cast<long>(linkIndex_) * kBaseErrDim;
    return theta.segment(off, kBaseErrDim);
  }

  ocs2::vector_t packParams_(double t, const ocs2::vector_t& x, const ocs2::vector_t& u, const ocs2::vector_t& taskRef13) const {
    ocs2::vector_t p(1 + nx_ + nu_ + kBaseRefDim);
    p << t, x, u, taskRef13;
    return p;
  }

  // --- build reference exactly like online getReferenceCostElement(xRef,uRef,*eeKinematics) ---
  ocs2::vector_t taskRefFromNominal_(const RefPackKnot& refk) const {
    // online:
    // costElement.setPosition(ee.getPosition(xRef)[0]);
    // costElement.setOrientation(ee.getOrientation(xRef)[0]);
    // costElement.setLinearVelocity(ee.getVelocity(xRef,uRef)[0]);
    // costElement.setAngularVelocity(ee.getAngularVelocity(xRef,uRef)[0]);

    const auto pos = eeKinDoublePtr_->getPosition(refk.xNominal)[0];                        // 3
    const auto ori = eeKinDoublePtr_->getOrientation(refk.xNominal)[0];                     // quaternion
    const auto lin = eeKinDoublePtr_->getVelocity(refk.xNominal, refk.uNominal)[0];         // 3
    const auto ang = eeKinDoublePtr_->getAngularVelocity(refk.xNominal, refk.uNominal)[0];  // 3

    ocs2::vector_t ts(13);
    ts << pos, ori.coeffs(), lin, ang;
    return ts;
  }

 private:
  int nx_{0}, nu_{0};
  size_t linkIndex_{0};
  pinocchio::FrameIndex frameID_{0};

  std::string modelFolder_;

  // reference extraction must use EndEffectorKinematics (double) to match online getParameters/getReferenceCostElement
  std::unique_ptr<ocs2::EndEffectorKinematics<ocs2::scalar_t>> eeKinDoublePtr_;

  std::unique_ptr<BaseStageThetaAD> thetaTape_;
  std::unique_ptr<BaseStageXUAD> xuTape_;
};

}  // namespace ocs2::humanoid_cost_matching
