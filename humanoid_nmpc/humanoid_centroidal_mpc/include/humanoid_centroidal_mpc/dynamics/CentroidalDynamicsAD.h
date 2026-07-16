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

#pragma once

#include <ocs2_centroidal_model/PinocchioCentroidalDynamicsAD.h>
#include <ocs2_core/dynamics/SystemDynamicsBase.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include <Eigen/Core>
#include <string>
#include <vector>

#include "humanoid_common_mpc/common/ModelSettings.h"

namespace ocs2::humanoid {

class CentroidalDynamicsAD final : public SystemDynamicsBase {
 public:
  CentroidalDynamicsAD(const PinocchioInterface& pinocchioInterface,
                       const CentroidalModelInfo& info,
                       const std::string& modelName,
                       const ModelSettings& modelSettings);

  ~CentroidalDynamicsAD() override = default;
  CentroidalDynamicsAD* clone() const override { return new CentroidalDynamicsAD(*this); }

  vector_t computeFlowMap(scalar_t time, const vector_t& state, const vector_t& input, const PreComputation& preComp) override;
  VectorFunctionLinearApproximation linearApproximation(scalar_t time,
                                                        const vector_t& state,
                                                        const vector_t& input,
                                                        const PreComputation& preComp) override;

 private:
  CentroidalDynamicsAD(const CentroidalDynamicsAD& rhs) = default;
  struct PushWindow {
    scalar_t t0{0.0};
    scalar_t dur{0.0};
    Eigen::Vector3d forceW{0.0, 0.0, 0.0};   // add to \dot h_lin  (xdot[0:3])
    Eigen::Vector3d torqueW{0.0, 0.0, 0.0};  // add to \dot h_ang  (xdot[3:6])
  };

  // helpers
  void loadPushProfileFromEnv();
  void addProfileA();
  void addProfileB();
  void applyPushWindows(scalar_t time, vector_t& xdot) const;
  void applyPushWindowsToLinF(scalar_t time, VectorFunctionLinearApproximation& lin) const;

  PinocchioCentroidalDynamicsAD pinocchioCentroidalDynamicsAd_;

  // for cost-matching
  bool costMatchingEnable_{false};
  Eigen::Vector3d theta_hl_{1.0, 1.0, 1.0};
  Eigen::Vector3d theta_ha_{1.0, 1.0, 1.0};

  // push windows (dummy sim disturbance)
  bool pushEnable_{false};
  std::vector<PushWindow> pushWindows_;
};

}  // namespace ocs2::humanoid