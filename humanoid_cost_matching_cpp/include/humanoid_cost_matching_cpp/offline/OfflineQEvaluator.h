#pragma once

#include <Eigen/Dense>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <ocs2_core/Types.h>
#include <ocs2_core/reference/ModeSchedule.h>
#include <ocs2_core/reference/TargetTrajectories.h>
#include <ocs2_oc/oc_problem/OptimalControlProblem.h>
#include <ocs2_oc/rollout/TimeTriggeredRollout.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>

#include <humanoid_centroidal_mpc/CentroidalMpcInterface.h>
#include <humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h>
#include "humanoid_centroidal_mpc/common/CentroidalMpcRobotModel.h"

#include <ocs2_centroidal_model/AccessHelperFunctions.h>                   // centroidal_model::getGeneralizedCoordinates
#include <ocs2_centroidal_model/CentroidalModelPinocchioMapping.h>         // CentroidalModelPinocchioMappingCppAd
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematicsCppAd.h>  // PinocchioEndEffectorKinematicsCppAd
#include <ocs2_robotic_tools/end_effector/EndEffectorKinematics.h>         // EndEffectorKinematics<scalar_t>

#include <humanoid_common_mpc/cost/ExternalTorqueQuadraticCostAD.h>  // for Config + loadConfigFromFile

#include "humanoid_cost_matching_cpp/offline/core/ThetaLayout.h"
#include "humanoid_cost_matching_cpp/offline/cost/CostBundle.h"
#include "humanoid_cost_matching_cpp/offline/rollout/FixedGridRK4Rollout.h"

namespace ocs2::humanoid_cost_matching {

class ZohController;

struct OfflineSample {
  ocs2::scalar_t t0;
  ocs2::scalar_t dt;
  ocs2::vector_t x0;                // nx
  ocs2::vector_array_t xMeasSeq;    // N+1, each nx (optional)
  ocs2::vector_array_t uSeq;        // N, each nu
  size_t initMode;                  // from obs.mode[k]
  ocs2::TargetTrajectories target;  // 3 knots
  ocs2::scalar_t horizon;           // e.g. N*dt
  ocs2::scalar_array_t timeGrid;    // size N+1 (optional, preferred)
};

struct QAndGrad {
  ocs2::scalar_t Q = 0.0;
  ocs2::vector_t grad;  // dim = layout.dim_total()
};

class OfflineQEvaluator {
 public:
  OfflineQEvaluator(std::string taskFile, std::string urdfFile, std::string referenceFile);

  // rollout + integrate online costs
  ocs2::scalar_t evaluateQ(const OfflineSample& sample);

  // fixed grid + sensitivity + CostBundle
  QAndGrad evaluateQAndGradFixedGrid(const OfflineSample& sample, const ocs2::vector_t& theta);

 private:
  void updateReference_(const OfflineSample& sample);

  ocs2::scalar_t accumulateCost_(const ocs2::scalar_array_t& timeTrajectory,
                                 const ocs2::vector_array_t& stateTrajectory,
                                 ZohController& ctrl);

 private:
  std::string taskFile_, urdfFile_, referenceFile_;

  std::unique_ptr<ocs2::humanoid::CentroidalMpcInterface> interface_;
  std::shared_ptr<ocs2::humanoid::SwitchedModelReferenceManager> refMgr_;

  const ocs2::OptimalControlProblem* problem_{nullptr};
  ocs2::RolloutBase* rollout_{nullptr};

  ocs2::ModeSchedule modeSchedule_;
  ocs2::TargetTrajectories targetTrajModified_;

  // caches / handles for offline terms
  const ocs2::PinocchioInterface* pinocchio_ = nullptr;
  const ocs2::PinocchioInterfaceCppAd* pinocchio_ad_ = nullptr;
  std::unique_ptr<ocs2::PinocchioInterfaceCppAd> pinocchio_ad_holder_;
  const ocs2::humanoid::CentroidalMpcRobotModel<ocs2::scalar_t>* robot_model_ = nullptr;
  const ocs2::humanoid::CentroidalMpcRobotModel<ocs2::ad_scalar_t>* robot_model_ad_ = nullptr;

  // for base term (task-space kinematics)
  CentroidalModelInfoCppAd infoCppAd_;
  std::unique_ptr<ocs2::CentroidalModelPinocchioMappingCppAd> pinocchioMappingCppAd_;
  std::unique_ptr<ocs2::EndEffectorKinematics<ocs2::scalar_t>> baseEeKin_;

  size_t torqueActiveJointCount_{0};

  // where to store cppad models
  std::string offlineCppAdFolder_;

  // unified cost bundle (tracking/com/swing/... all live here)
  std::unique_ptr<CostBundle> costBundle_;
};

}  // namespace ocs2::humanoid_cost_matching
