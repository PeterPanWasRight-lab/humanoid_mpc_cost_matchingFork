/******************************************************************************
Copyright (c) 2025, Manuel Yves Galliker. All rights reserved.
Copyright (c) 2024, 1X Technologies. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************************************************************/

#include "humanoid_centroidal_mpc/dynamics/CentroidalDynamicsAD.h"

#include <cstdlib>   // std::getenv
#include <iostream>  // std::cerr

namespace ocs2::humanoid {

CentroidalDynamicsAD::CentroidalDynamicsAD(const PinocchioInterface& pinocchioInterface,
                                           const CentroidalModelInfo& info,
                                           const std::string& modelName,
                                           const ModelSettings& modelSettings)
    : pinocchioCentroidalDynamicsAd_(pinocchioInterface,
                                     info,
                                     modelName,
                                     modelSettings.modelFolderCppAd,
                                     modelSettings.recompileLibrariesCppAd,
                                     modelSettings.verboseCppAd),
      costMatchingEnable_(modelSettings.costMatchingEnable),
      theta_hl_(modelSettings.theta_hl),
      theta_ha_(modelSettings.theta_ha) {
  // Load push windows (hardcoded profiles) from environment variables
  loadPushProfileFromEnv();

  // Optional: one-line ctor print to confirm it is used
  std::cerr << "[CentroidalDynamicsAD] pushEnable=" << (pushEnable_ ? "true" : "false") << " profile=" << pushProfileName_
            << " windows=" << pushWindows_.size() << std::endl;
  std::cerr << "costMatchingEnable: " << costMatchingEnable_;
  std::cerr << " theta_hl: " << theta_hl_.transpose() << " theta_ha: " << theta_ha_.transpose() << std::endl;
  std::cerr.flush();
}

void CentroidalDynamicsAD::loadPushProfileFromEnv() {
  // Enable flag: WB_PUSH_ENABLE=0 to disable, otherwise enabled by default
  pushEnable_ = true;
  if (const char* en = std::getenv("WB_PUSH_ENABLE"); en != nullptr) {
    // treat "0" as disable, anything else as enable
    pushEnable_ = (std::string(en) != "0");
  }

  pushWindows_.clear();
  if (!pushEnable_) return;

  addProfile();
}

// push train + small yaw torque)
void CentroidalDynamicsAD::addProfile() {
  pushWindows_.clear();
  // lateral push train
  pushWindows_.push_back(PushWindow{12.00, 0.10, Eigen::Vector3d(0.0, +18.0, 0.0), Eigen::Vector3d(0.0, 0.0, 0.0)});
  pushWindows_.push_back(PushWindow{12.75, 0.10, Eigen::Vector3d(0.0, +18.0, 0.0), Eigen::Vector3d(0.0, 0.0, 0.0)});
  pushWindows_.push_back(PushWindow{13.55, 0.10, Eigen::Vector3d(0.0, +18.0, 0.0), Eigen::Vector3d(0.0, 0.0, 0.0)});

  // small yaw torque pulse (plausible body twist during contact)
  pushWindows_.push_back(PushWindow{13.05, 0.06, Eigen::Vector3d(0.0, 0.0, 0.0), Eigen::Vector3d(0.0, 0.0, +8.0)});
}

void CentroidalDynamicsAD::applyPushWindows(scalar_t time, vector_t& xdot) const {
  if (!pushEnable_ || pushWindows_.empty()) return;

  for (const auto& w : pushWindows_) {
    if (time >= w.t0 && time <= (w.t0 + w.dur)) {
      // xdot[0:3] = \dot h_lin, xdot[3:6] = \dot h_ang
      xdot.segment<3>(0) += w.forceW;
      xdot.segment<3>(3) += w.torqueW;
    }
  }
}

void CentroidalDynamicsAD::applyPushWindowsToLinF(scalar_t time, VectorFunctionLinearApproximation& lin) const {
  if (!pushEnable_ || pushWindows_.empty()) return;

  for (const auto& w : pushWindows_) {
    if (time >= w.t0 && time <= (w.t0 + w.dur)) {
      lin.f.segment<3>(0) += w.forceW;
      lin.f.segment<3>(3) += w.torqueW;
    }
  }
}

vector_t CentroidalDynamicsAD::computeFlowMap(scalar_t time, const vector_t& state, const vector_t& input, const PreComputation& preComp) {
  vector_t xdot = pinocchioCentroidalDynamicsAd_.getValue(time, state, input);

  // cost-matching scaling
  if (costMatchingEnable_) {
    xdot.segment<3>(0) = theta_hl_.cwiseProduct(xdot.segment<3>(0));
    xdot.segment<3>(3) = theta_ha_.cwiseProduct(xdot.segment<3>(3));
  }
  // external push: apply AFTER scaling so it is NOT affected by theta
  applyPushWindows(time, xdot);

  return xdot;
}

VectorFunctionLinearApproximation CentroidalDynamicsAD::linearApproximation(scalar_t time,
                                                                            const vector_t& state,
                                                                            const vector_t& input,
                                                                            const PreComputation& preComp) {
  auto lin = pinocchioCentroidalDynamicsAd_.getLinearApproximation(time, state, input);

  // cost-matching scaling
  if (costMatchingEnable_) {
    lin.f.segment<3>(0) = theta_hl_.cwiseProduct(lin.f.segment<3>(0));
    lin.f.segment<3>(3) = theta_ha_.cwiseProduct(lin.f.segment<3>(3));

    // A = dfdx ： scale corresponding rows
    lin.dfdx.row(0) *= theta_hl_[0];
    lin.dfdx.row(1) *= theta_hl_[1];
    lin.dfdx.row(2) *= theta_hl_[2];

    lin.dfdx.row(3) *= theta_ha_[0];
    lin.dfdx.row(4) *= theta_ha_[1];
    lin.dfdx.row(5) *= theta_ha_[2];

    // B = dfdu : scale corresponding rows
    lin.dfdu.row(0) *= theta_hl_[0];
    lin.dfdu.row(1) *= theta_hl_[1];
    lin.dfdu.row(2) *= theta_hl_[2];

    lin.dfdu.row(3) *= theta_ha_[0];
    lin.dfdu.row(4) *= theta_ha_[1];
    lin.dfdu.row(5) *= theta_ha_[2];
  }

  // external push: time-only bias -> only affects lin.f (A/B unchanged)
  applyPushWindowsToLinF(time, lin);

  return lin;
}

}  // namespace ocs2::humanoid