#pragma once

#include <memory>
#include <string>

#include <ocs2_core/Types.h>
#include "humanoid_cost_matching_cpp/offline/core/RefPack.h"
#include "humanoid_cost_matching_cpp/offline/core/ThetaLayout.h"
#include "humanoid_cost_matching_cpp/offline/cost/ICostTerm.h"
#include "humanoid_cost_matching_cpp/offline/cost/tapes/cost_tracking_tapes.h"

namespace ocs2::humanoid_cost_matching {

/**
 * Tracking stage term (quadratic):
 *   ℓ = (x-xNom)^T diag(theta_q) (x-xNom) + (u-uNom)^T diag(theta_r) (u-uNom)
 */
class TrackingStageTerm final : public ICostTerm {
 public:
  TrackingStageTerm(int nx, int nu, std::string modelName, std::string modelFolder, bool recompile, bool verbose)
      : nx_(nx), nu_(nu), modelFolder_(std::move(modelFolder)) {
    // theta tape: variables=theta_qr, params=[t|x|u|xNom|uNom]
    stageThetaTape_.reset(new TrackingStageThetaAD(nx_, nu_, std::move(modelName), modelFolder_, recompile, verbose));

    // XU tape: variables=[x;u], params=[t|xNom|uNom|theta_q|theta_r]
    stageXUTape_.reset(new TrackingStageXUAD(nx_, nu_, "cm_tracking_stage_xu_v1", modelFolder_, recompile, verbose));
  }

  double value(double t,
               const ocs2::vector_t& x,
               const ocs2::vector_t& u,
               const RefPackKnot& refk,
               const ocs2::vector_t& theta,
               const ThetaLayout& layout) const override {
    // theta_qr = [theta_q, theta_r]
    const ocs2::vector_t theta_q = layout.theta_q(theta);
    const ocs2::vector_t theta_r = layout.theta_r(theta);
    ocs2::vector_t theta_qr(theta_q.size() + theta_r.size());
    theta_qr << theta_q, theta_r;

    // params = [t | x | u | xNom | uNom]
    ocs2::vector_t params(1 + nx_ + nu_ + nx_ + nu_);
    params << t, x, u, refk.xNominal, refk.uNominal;

    return stageThetaTape_->value(theta_qr, params);
  }

  ocs2::vector_t grad_theta(double t,
                            const ocs2::vector_t& x,
                            const ocs2::vector_t& u,
                            const RefPackKnot& refk,
                            const ocs2::vector_t& theta,
                            const ThetaLayout& layout) const override {
    ocs2::vector_t g_full = ocs2::vector_t::Zero(theta.size());

    const ocs2::vector_t theta_q = layout.theta_q(theta);
    const ocs2::vector_t theta_r = layout.theta_r(theta);
    ocs2::vector_t theta_qr(theta_q.size() + theta_r.size());
    theta_qr << theta_q, theta_r;

    ocs2::vector_t params(1 + nx_ + nu_ + nx_ + nu_);
    params << t, x, u, refk.xNominal, refk.uNominal;

    const ocs2::vector_t dldtheta_qr = stageThetaTape_->grad_theta(theta_qr, params);  // (nx+nu)

    g_full.segment(static_cast<long>(layout.off_q), static_cast<long>(layout.nx + layout.nu)) += dldtheta_qr;
    return g_full;
  }

  ocs2::vector_t grad_x(double t,
                        const ocs2::vector_t& x,
                        const ocs2::vector_t& u,
                        const RefPackKnot& refk,
                        const ocs2::vector_t& theta,
                        const ThetaLayout& layout) const override {
    // TapeXU: variables z=[x;u]
    ocs2::vector_t z(nx_ + nu_);
    z << x, u;

    const ocs2::vector_t theta_q = layout.theta_q(theta);
    const ocs2::vector_t theta_r = layout.theta_r(theta);

    // paramsXU = [t | xNom | uNom | theta_q | theta_r]
    ocs2::vector_t paramsXU(1 + nx_ + nu_ + nx_ + nu_);
    paramsXU << t, refk.xNominal, refk.uNominal, theta_q, theta_r;

    const ocs2::vector_t g_z = stageXUTape_->grad_xu(z, paramsXU);  // (nx+nu)
    return g_z.head(nx_);
  }

 private:
  int nx_ = 0;
  int nu_ = 0;
  std::string modelFolder_;

  std::unique_ptr<TrackingStageThetaAD> stageThetaTape_;
  std::unique_ptr<TrackingStageXUAD> stageXUTape_;
};

/**
 * Tracking terminal term:
 *   T = (x-xNom)^T diag(theta_qf) (x-xNom)
 */
class TrackingTerminalTerm final : public ICostTerm {
 public:
  TrackingTerminalTerm(int nx, std::string modelName, std::string modelFolder, bool recompile, bool verbose)
      : nx_(nx), modelFolder_(std::move(modelFolder)) {
    // theta tape: variables=theta_qf, params=[t|x|xNom]
    terminalThetaTape_.reset(new TrackingTerminalThetaAD(nx_, std::move(modelName), modelFolder_, recompile, verbose));

    // X tape: variables=x, params=[t|xNom|theta_qf]
    terminalXTape_.reset(new TrackingTerminalXAD(nx_, "cm_tracking_terminal_x_v1", modelFolder_, recompile, verbose));
  }

  double value(double t,
               const ocs2::vector_t& x,
               const ocs2::vector_t& /*u*/,
               const RefPackKnot& refk,
               const ocs2::vector_t& theta,
               const ThetaLayout& layout) const override {
    const ocs2::vector_t theta_qf = layout.theta_qf(theta);

    // paramsT = [t | x | xNom]
    ocs2::vector_t paramsT(1 + nx_ + nx_);
    paramsT << t, x, refk.xNominal;

    return terminalThetaTape_->value(theta_qf, paramsT);
  }

  ocs2::vector_t grad_theta(double t,
                            const ocs2::vector_t& x,
                            const ocs2::vector_t& /*u*/,
                            const RefPackKnot& refk,
                            const ocs2::vector_t& theta,
                            const ThetaLayout& layout) const override {
    ocs2::vector_t g_full = ocs2::vector_t::Zero(theta.size());

    const ocs2::vector_t theta_qf = layout.theta_qf(theta);

    ocs2::vector_t paramsT(1 + nx_ + nx_);
    paramsT << t, x, refk.xNominal;

    const ocs2::vector_t dTdtheta = terminalThetaTape_->grad_theta(theta_qf, paramsT);  // (nx)

    g_full.segment(static_cast<long>(layout.off_qf), static_cast<long>(layout.nx)) += dTdtheta;
    return g_full;
  }

  ocs2::vector_t grad_x(double t,
                        const ocs2::vector_t& x,
                        const ocs2::vector_t& /*u*/,
                        const RefPackKnot& refk,
                        const ocs2::vector_t& theta,
                        const ThetaLayout& layout) const override {
    const ocs2::vector_t theta_qf = layout.theta_qf(theta);

    // paramsX = [t | xNom | theta_qf]
    ocs2::vector_t paramsX(1 + nx_ + nx_);
    paramsX << t, refk.xNominal, theta_qf;

    return terminalXTape_->grad_x(x, paramsX);
  }

 private:
  int nx_ = 0;
  std::string modelFolder_;

  std::unique_ptr<TrackingTerminalThetaAD> terminalThetaTape_;
  std::unique_ptr<TrackingTerminalXAD> terminalXTape_;
};

}  // namespace ocs2::humanoid_cost_matching
