#pragma once

#include <memory>
#include <string>

#include <ocs2_core/Types.h>

#include "humanoid_cost_matching_cpp/offline/core/RefPack.h"
#include "humanoid_cost_matching_cpp/offline/core/ThetaLayout.h"
#include "humanoid_cost_matching_cpp/offline/cost/ICostTerm.h"
#include "humanoid_cost_matching_cpp/offline/cost/tapes/cost_com_tapes.h"

#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"

namespace ocs2::humanoid_cost_matching {

/**
 * COM/ICP stage term:
 *  - theta_com: layout.comDim == kComErrDim
 *
 * Params:
 *   thetaTape params: x only (nx)
 *   XU   tape params: theta_com only (kComErrDim)
 */
class ComStageTerm final : public ICostTerm {
 public:
  ComStageTerm(int nx,
               int nu,
               std::string modelFolder,
               const ocs2::PinocchioInterfaceCppAd& pinocchioCppAd,
               const ocs2::humanoid::CentroidalMpcRobotModel<ocs2::ad_scalar_t>& robotModelAd,
               bool recompile = false,
               bool verbose = false)
      : nx_(nx), nu_(nu), modelFolder_(std::move(modelFolder)) {
    thetaTape_.reset(new ComStageThetaAD(static_cast<size_t>(nx_), pinocchioCppAd, robotModelAd, "cm_com_stage_theta_v2", modelFolder_,
                                         recompile, verbose));

    xuTape_.reset(new ComStageXUAD(static_cast<size_t>(nx_), static_cast<size_t>(nu_), pinocchioCppAd, robotModelAd, "cm_com_stage_xu_v2",
                                   modelFolder_, recompile, verbose));
  }

  double value(double /*t*/,
               const ocs2::vector_t& x,
               const ocs2::vector_t& /*u*/,
               const RefPackKnot& /*refk*/,
               const ocs2::vector_t& theta,
               const ThetaLayout& layout) const override {
    const ocs2::vector_t th = layout.theta_com(theta);  // (kComErrDim)

    // theta tape params: x only
    return thetaTape_->value(th, x);
  }

  ocs2::vector_t grad_theta(double /*t*/,
                            const ocs2::vector_t& x,
                            const ocs2::vector_t& /*u*/,
                            const RefPackKnot& /*refk*/,
                            const ocs2::vector_t& theta,
                            const ThetaLayout& layout) const override {
    ocs2::vector_t g = ocs2::vector_t::Zero(theta.size());

    const ocs2::vector_t th = layout.theta_com(theta);
    const ocs2::vector_t d = thetaTape_->grad_theta(th, x);  // (kComErrDim)

    g.segment(static_cast<long>(layout.off_com), static_cast<long>(layout.comDim)) += d;
    return g;
  }

  ocs2::vector_t grad_x(double /*t*/,
                        const ocs2::vector_t& x,
                        const ocs2::vector_t& u,
                        const RefPackKnot& /*refk*/,
                        const ocs2::vector_t& theta,
                        const ThetaLayout& layout) const override {
    const ocs2::vector_t th = layout.theta_com(theta);  // (kComErrDim)

    // variables=[x;u]
    ocs2::vector_t xu(nx_ + nu_);
    xu << x, u;

    // params = theta_com only
    const ocs2::vector_t g_xu = xuTape_->grad_xu(xu, th);  // (nx+nu)
    return g_xu.head(nx_);
  }

 private:
  int nx_ = 0;
  int nu_ = 0;
  std::string modelFolder_;

  std::unique_ptr<ComStageThetaAD> thetaTape_;
  std::unique_ptr<ComStageXUAD> xuTape_;
};

}  // namespace ocs2::humanoid_cost_matching
