#include "humanoid_cost_matching_cpp/offline/OfflineQEvaluator.h"

#include "humanoid_common_mpc/common/ModelSettings.h"
#include "humanoid_common_mpc/pinocchio_model/DynamicsHelperFunctions.h"

#include "humanoid_cost_matching_cpp/offline/core/RefPack.h"
#include "humanoid_cost_matching_cpp/offline/core/ThetaLayout.h"
#include "humanoid_cost_matching_cpp/offline/core/ZohController.h"

#include "humanoid_cost_matching_cpp/offline/cost/CostBundle.h"
#include "humanoid_cost_matching_cpp/offline/cost/terms/cost_base_terms.h"
#include "humanoid_cost_matching_cpp/offline/cost/terms/cost_com_terms.h"
#include "humanoid_cost_matching_cpp/offline/cost/terms/cost_swing_terms.h"
#include "humanoid_cost_matching_cpp/offline/cost/terms/cost_torque_terms.h"
#include "humanoid_cost_matching_cpp/offline/cost/terms/cost_tracking_terms.h"
#include "humanoid_cost_matching_cpp/offline/cost/terms/penalty_collision_terms.h"
#include "humanoid_cost_matching_cpp/offline/cost/terms/penalty_cop_terms.h"
#include "humanoid_cost_matching_cpp/offline/cost/terms/penalty_joint_terms.h"
#include "humanoid_cost_matching_cpp/offline/cost/terms/penalty_normal_velocity_terms.h"
#include "humanoid_cost_matching_cpp/offline/cost/terms/penalty_zero_velocity_terms.h"

#include <ocs2_core/misc/LoadData.h>
#include <boost/property_tree/info_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include "humanoid_common_mpc/contact/ContactRectangle.h"
#include "humanoid_cost_matching_cpp/offline/cost/helpers/PieceWisePolynomialBarrierPenaltyAD.h"

#include "humanoid_cost_matching_cpp/offline/rollout/FixedGridRK4Rollout.h"

#include <ocs2_core/ComputationRequest.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace ocs2::humanoid_cost_matching {

OfflineQEvaluator::OfflineQEvaluator(std::string taskFile, std::string urdfFile, std::string referenceFile)
    : taskFile_(std::move(taskFile)), urdfFile_(std::move(urdfFile)), referenceFile_(std::move(referenceFile)) {
  // setupOCP=true: build costs/constraints/dynamics/rollout just like online
  interface_ = std::make_unique<ocs2::humanoid::CentroidalMpcInterface>(taskFile_, urdfFile_, referenceFile_, true);

  // modelFolder: separate offline models
  offlineCppAdFolder_ = interface_->modelSettings().modelFolderCppAd + "/cost_matching_offline_q";

  // Reference manager
  auto baseRef = interface_->getReferenceManagerPtr();
  refMgr_ = std::dynamic_pointer_cast<ocs2::humanoid::SwitchedModelReferenceManager>(baseRef);
  if (!refMgr_) {
    throw std::runtime_error("[OfflineQEvaluator] Failed to cast ReferenceManagerInterface -> SwitchedModelReferenceManager");
  }

  problem_ = &interface_->getOptimalControlProblem();
  rollout_ = &interface_->getRollout();
  if (!problem_) throw std::runtime_error("[OfflineQEvaluator] problem_ is null.");
  if (!rollout_) throw std::runtime_error("[OfflineQEvaluator] rollout_ is null.");

  // pinocchio + robot model handles (needed by frame IDs, etc.)
  pinocchio_ = &interface_->getPinocchioInterface();
  robot_model_ = &interface_->getMpcRobotModel();
  robot_model_ad_ = &interface_->getMpcRobotModelAD();

  if (!pinocchio_) throw std::runtime_error("[OfflineQEvaluator] pinocchio_ is null.");
  if (!robot_model_) throw std::runtime_error("[OfflineQEvaluator] robot_model_ is null.");
  if (!robot_model_ad_) throw std::runtime_error("[OfflineQEvaluator] robot_model_ad_ is null.");

  // Create ONE owned PinocchioInterfaceCppAd instance for building terms.
  // (Each tape will copy it internally anyway, but we still need a concrete object to pass in.)
  pinocchio_ad_holder_ = std::make_unique<ocs2::PinocchioInterfaceCppAd>(interface_->getPinocchioInterface().toCppAd());

  if (!pinocchio_ad_holder_) throw std::runtime_error("[OfflineQEvaluator] pinocchio_ad_holder_ is null.");

  // ---------------- build cost bundle once ----------------
  costBundle_ = std::make_unique<CostBundle>();

  const bool recompile = false;  // set true once if no models exist
  const bool verbose = false;

  const int nx = static_cast<int>(robot_model_->getStateDim());
  const int nu = static_cast<int>(robot_model_->getInputDim());

  // tracking stage/terminal terms
  costBundle_->stageTerms.emplace_back(
      std::make_unique<TrackingStageTerm>(nx, nu, "cm_tracking_stage_theta_v1", offlineCppAdFolder_, recompile, verbose));
  costBundle_->terminalTerms.emplace_back(
      std::make_unique<TrackingTerminalTerm>(nx, "cm_tracking_terminal_theta_v1", offlineCppAdFolder_, recompile, verbose));

  // com stage term
  costBundle_->stageTerms.emplace_back(
      std::make_unique<ComStageTerm>(nx, nu, offlineCppAdFolder_, *pinocchio_ad_holder_, *robot_model_ad_, recompile, verbose));

  // swing stage term: two feet, frames from model (frame IDs come from NON-AD model)
  const auto& model = pinocchio_->getModel();
  const auto fidL = model.getFrameId("foot_l_contact");
  const auto fidR = model.getFrameId("foot_r_contact");

  costBundle_->stageTerms.emplace_back(std::make_unique<SwingStageTerm>(nx, nu,
                                                                        /*contactIndex=*/0,
                                                                        /*frameID=*/fidL, *pinocchio_ad_holder_, *robot_model_ad_,
                                                                        offlineCppAdFolder_, recompile, verbose, refMgr_.get()));

  costBundle_->stageTerms.emplace_back(std::make_unique<SwingStageTerm>(nx, nu,
                                                                        /*contactIndex=*/1,
                                                                        /*frameID=*/fidR, *pinocchio_ad_holder_, *robot_model_ad_,
                                                                        offlineCppAdFolder_, recompile, verbose, refMgr_.get()));

  // ---------------- base stage term (task-space kinematics) ----------------
  // Build the same objects as online:
  //   infoCppAd = centroidalModelInfo_.toCppAd()
  //   CentroidalModelPinocchioMappingCppAd pinocchioMappingCppAd(infoCppAd)
  //   velocityUpdateCallback uses centroidal_model::getGeneralizedCoordinates + updateCentroidalDynamics
  //
  // Then build PinocchioEndEffectorKinematicsCppAd for the single link in task.info (mid360_link)
  {
    // keep infoCppAd alive (member)
    infoCppAd_ = interface_->getCentroidalModelInfo().toCppAd();
    pinocchioMappingCppAd_ = std::make_unique<ocs2::CentroidalModelPinocchioMappingCppAd>(infoCppAd_);

    // link_name from task.info (task_space_costs.torso.link_name)
    const std::string costName = "torso";
    const std::string linkName = "mid360_link";

    const auto frameID = pinocchio_->getModel().getFrameId(linkName);

    // velocity update callback: EXACTLY as online
    auto velocityUpdateCallback = [this](const ocs2::ad_vector_t& state, ocs2::PinocchioInterfaceCppAd& pinocchioInterfaceAd) {
      const ocs2::ad_vector_t q = ocs2::centroidal_model::getGeneralizedCoordinates(state, infoCppAd_);
      ocs2::updateCentroidalDynamics(pinocchioInterfaceAd, infoCppAd_, q);
    };

    // create EndEffectorKinematics<double> object used to build taskRef (online-consistent reference extraction)
    baseEeKin_.reset(new ocs2::PinocchioEndEffectorKinematicsCppAd(
        *pinocchio_, *pinocchioMappingCppAd_, std::vector<std::string>{linkName}, nx, nu, velocityUpdateCallback, linkName,
        interface_->modelSettings().modelFolderCppAd, interface_->modelSettings().recompileLibrariesCppAd,
        interface_->modelSettings().verboseCppAd));

    if (!baseEeKin_) throw std::runtime_error("[OfflineQEvaluator] baseEeKin_ is null");

    // add BaseStageTerm (only one link => linkIndex=0, baseDim=12)
    costBundle_->stageTerms.emplace_back(std::make_unique<BaseStageTerm>(nx, nu,
                                                                         /*linkIndex=*/0,
                                                                         /*frameID=*/frameID,
                                                                         /*eeKinDouble=*/*baseEeKin_,
                                                                         /*pinocchioCppAd=*/*pinocchio_ad_holder_,
                                                                         /*robotModelAd=*/*robot_model_ad_,
                                                                         /*modelFolder=*/offlineCppAdFolder_, recompile, verbose));
  }

  // ---------------- torque stage terms (external torque quadratic) ----------------
  {
    const auto& ms = interface_->modelSettings();

    // load configs exactly like HumanoidCostConstraintFactory
    ocs2::humanoid::ExternalTorqueQuadraticCostAD::Config cfgL =
        ocs2::humanoid::ExternalTorqueQuadraticCostAD::loadConfigFromFile(taskFile_, "left_leg_torque_cost.", verbose);
    ocs2::humanoid::ExternalTorqueQuadraticCostAD::Config cfgR =
        ocs2::humanoid::ExternalTorqueQuadraticCostAD::loadConfigFromFile(taskFile_, "right_leg_torque_cost.", verbose);

    if (cfgL.activeJointNames.size() != static_cast<size_t>(cfgL.weights.size()))
      throw std::runtime_error("[OfflineQEvaluator] left torque config: activeJointNames and weights size mismatch");
    if (cfgR.activeJointNames.size() != static_cast<size_t>(cfgR.weights.size()))
      throw std::runtime_error("[OfflineQEvaluator] right torque config: activeJointNames and weights size mismatch");
    if (cfgL.activeJointNames.size() != cfgR.activeJointNames.size())
      throw std::runtime_error("[OfflineQEvaluator] left/right torque activeJointNames size mismatch (expect same m)");

    torqueActiveJointCount_ = cfgL.activeJointNames.size();
    if (torqueActiveJointCount_ != ThetaLayout::torque_dim_per_leg) {
      throw std::runtime_error("[OfflineQEvaluator] torqueActiveJointCount (" + std::to_string(torqueActiveJointCount_) +
                               ") from taskFile must match ThetaLayout::torque_dim_per_leg (" +
                               std::to_string(ThetaLayout::torque_dim_per_leg) + ")");
    }

    const size_t m = torqueActiveJointCount_;
    if (m == 0) throw std::runtime_error("[OfflineQEvaluator] torque activeJointNames empty");

    // precompute tau indices: tauExt[6 + jointIndex(name)] exactly as online
    std::vector<size_t> tauActiveIndices;
    tauActiveIndices.reserve(m);
    for (size_t i = 0; i < m; ++i) {
      const auto& jn = cfgL.activeJointNames[i];
      tauActiveIndices.push_back(6 + robot_model_ad_->getJointIndex(jn));
    }

    // frame IDs: online uses modelSettings.contactNames[endEffectorIndex]
    const auto fid0 = pinocchio_->getModel().getFrameId(ms.contactNames[0]);
    const auto fid1 = pinocchio_->getModel().getFrameId(ms.contactNames[1]);

    // NOTE: modelName must be unique per contact (avoid cg/so overwrite)
    costBundle_->stageTerms.emplace_back(std::make_unique<TorqueStageTerm>(nx, nu,
                                                                           /*contactIndex=*/0,
                                                                           /*frameID=*/fid0, tauActiveIndices, *pinocchio_ad_holder_,
                                                                           *robot_model_ad_, offlineCppAdFolder_, recompile, verbose,
                                                                           refMgr_.get()));

    costBundle_->stageTerms.emplace_back(std::make_unique<TorqueStageTerm>(nx, nu,
                                                                           /*contactIndex=*/1,
                                                                           /*frameID=*/fid1, tauActiveIndices, *pinocchio_ad_holder_,
                                                                           *robot_model_ad_, offlineCppAdFolder_, recompile, verbose,
                                                                           refMgr_.get()));
  }
  // ---------------- joint limits penalty (stage + terminal) terms ----------------
  {
    boost::property_tree::ptree pt;
    boost::property_tree::read_info(taskFile_, pt);

    PiecewiseBarrierConfig cfg;
    loadData::loadPtreeValue(pt, cfg.mu, "jointLimits.mu", verbose);
    loadData::loadPtreeValue(pt, cfg.delta, "jointLimits.delta", verbose);

    // same as online readPinocchioJointLimits(): tail to skip universe/root
    const auto& model = pinocchio_->getModel();
    const auto& ms = interface_->modelSettings();
    const size_t jointCount = ms.mpcModelJointNames.size();
    if (jointCount == 0) throw std::runtime_error("[OfflineQEvaluator] mpcModelJointNames is empty");

    ocs2::vector_t upper = model.upperPositionLimit.tail(jointCount);
    ocs2::vector_t lower = model.lowerPositionLimit.tail(jointCount);

    // stage
    costBundle_->stageTerms.emplace_back(
        std::make_unique<JointStageTerm>(nx, std::make_pair(lower, upper), cfg, *robot_model_ad_, offlineCppAdFolder_, recompile, verbose));

    // terminal
    costBundle_->terminalTerms.emplace_back(std::make_unique<JointTerminalTerm>(nx, std::make_pair(lower, upper), cfg, *robot_model_ad_,
                                                                                offlineCppAdFolder_, recompile, verbose));
  }
  // ---------------- collision constraint penalty (stage + terminal) terms ----------------
  {
    boost::property_tree::ptree pt;
    boost::property_tree::read_info(taskFile_, pt);

    // mu/delta (exact keys from task.info)
    PiecewiseBarrierConfig penCfg;
    loadData::loadPtreeValue(pt, penCfg.mu, "collision_constraint.mu", verbose);
    loadData::loadPtreeValue(pt, penCfg.delta, "collision_constraint.delta", verbose);

    // frames + radii (exact keys from task.info)
    FootCollisionConfigOffline cfg;
    loadData::loadPtreeValue(pt, cfg.leftAnkleFrame, "collision_constraint.foot.leftAnkleFrame", verbose);
    loadData::loadPtreeValue(pt, cfg.rightAnkleFrame, "collision_constraint.foot.rightAnkleFrame", verbose);
    loadData::loadPtreeValue(pt, cfg.footCollisionSphereRadius, "collision_constraint.foot.footCollisionSphereRadius", verbose);

    loadData::loadPtreeValue(pt, cfg.leftKneeFrame, "collision_constraint.knee.leftKneeFrame", verbose);
    loadData::loadPtreeValue(pt, cfg.rightKneeFrame, "collision_constraint.knee.rightKneeFrame", verbose);
    loadData::loadPtreeValue(pt, cfg.kneeCollisionSphereRadius, "collision_constraint.knee.kneeCollisionSphereRadius", verbose);

    // stage
    costBundle_->stageTerms.emplace_back(std::make_unique<CollisionStageTerm>(nx, cfg, penCfg, *pinocchio_ad_holder_, *robot_model_ad_,
                                                                              offlineCppAdFolder_, recompile, verbose, refMgr_.get()));

    // terminal
    costBundle_->terminalTerms.emplace_back(std::make_unique<CollisionTerminalTerm>(
        nx, cfg, penCfg, *pinocchio_ad_holder_, *robot_model_ad_, offlineCppAdFolder_, recompile, verbose, refMgr_.get()));
  }
  // ---------------- FrictionForceConeConstraint is U-only, so no penalty term needed ----------------
  // ---------------- CoP (ContactMomentXY) penalty stage term ----------------
  {
    // load mu/delta from task.info: "contacts.contactMomentXYSoftConstraint."
    boost::property_tree::ptree pt;
    boost::property_tree::read_info(taskFile_, pt);
    const std::string prefix = "contacts.contactMomentXYSoftConstraint.";

    RelaxedBarrierConfig cfg;
    loadData::loadPtreeValue(pt, cfg.mu, prefix + "mu", verbose);
    loadData::loadPtreeValue(pt, cfg.delta, prefix + "delta", verbose);

    // per-foot contact rectangle from task.info (same as online factory)
    const auto rect0 = ocs2::humanoid::ContactRectangle::loadContactRectangle(taskFile_, robot_model_->modelSettings, 0);
    const auto rect1 = ocs2::humanoid::ContactRectangle::loadContactRectangle(taskFile_, robot_model_->modelSettings, 1);

    costBundle_->stageTerms.emplace_back(std::make_unique<CoPStageTerm>(
        nx, nu,
        /*contactIndex=*/0, rect0, cfg, *pinocchio_ad_holder_, *robot_model_ad_, offlineCppAdFolder_, recompile, verbose, refMgr_.get(),
        /*modelNamePrefix=*/"pen_contactCoP"));

    costBundle_->stageTerms.emplace_back(std::make_unique<CoPStageTerm>(
        nx, nu,
        /*contactIndex=*/1, rect1, cfg, *pinocchio_ad_holder_, *robot_model_ad_, offlineCppAdFolder_, recompile, verbose, refMgr_.get(),
        /*modelNamePrefix=*/"pen_contactCoP"));
  }
  // ---------------- ZeroVelocity (stance foot) penalty stage term ----------------
  {
    double w_task = 1.0;  // default weight defined only in cost-macthing code, not in online factory

    // gains (match online)
    const double posGainZ = robot_model_->modelSettings.footConstraintConfig.positionErrorGain_z;
    const double oriGain = robot_model_->modelSettings.footConstraintConfig.orientationErrorGain;

    // frame ids
    const auto& modelAd = pinocchio_ad_holder_->getModel();
    const auto fid0 = modelAd.getFrameId(robot_model_->modelSettings.contactNames6DoF[0]);
    const auto fid1 = modelAd.getFrameId(robot_model_->modelSettings.contactNames6DoF[1]);

    costBundle_->stageTerms.emplace_back(std::make_unique<ocs2::humanoid_cost_matching::ZeroVelocityStageTerm>(
        nx, nu,
        /*contactIndex=*/0, fid0,
        /*w_task=*/w_task,
        /*posGainZ=*/posGainZ,
        /*oriGain=*/oriGain, *pinocchio_ad_holder_, *robot_model_ad_, offlineCppAdFolder_, recompile, verbose, refMgr_.get(),
        /*modelNamePrefix=*/"pen_zeroVel"));

    costBundle_->stageTerms.emplace_back(std::make_unique<ocs2::humanoid_cost_matching::ZeroVelocityStageTerm>(
        nx, nu,
        /*contactIndex=*/1, fid1,
        /*w_task=*/w_task,
        /*posGainZ=*/posGainZ,
        /*oriGain=*/oriGain, *pinocchio_ad_holder_, *robot_model_ad_, offlineCppAdFolder_, recompile, verbose, refMgr_.get(),
        /*modelNamePrefix=*/"pen_zeroVel"));
  }
  // ---------------- Normal velocity penalty stage term ----------------
  {
    const double posGainZ = interface_->modelSettings().footConstraintConfig.positionErrorGain_z;
    double w = 1.0;  // default weight defined only in cost-macthing code, not in online factory

    const auto& model = pinocchio_->getModel();
    const auto& ms = interface_->modelSettings();
    const pinocchio::FrameIndex fid0 = model.getFrameId(ms.contactNames6DoF[0]);
    const pinocchio::FrameIndex fid1 = model.getFrameId(ms.contactNames6DoF[1]);

    costBundle_->stageTerms.emplace_back(std::make_unique<ocs2::humanoid_cost_matching::NormalVelStageTerm>(
        nx, nu, /*footIndex=*/0, fid0, *pinocchio_ad_holder_, *robot_model_ad_, offlineCppAdFolder_, recompile, verbose, refMgr_.get(), w,
        posGainZ));

    costBundle_->stageTerms.emplace_back(std::make_unique<ocs2::humanoid_cost_matching::NormalVelStageTerm>(
        nx, nu, /*footIndex=*/1, fid1, *pinocchio_ad_holder_, *robot_model_ad_, offlineCppAdFolder_, recompile, verbose, refMgr_.get(), w,
        posGainZ));
  }
}

/**
 * Update reference-related internal states to match online MPC:
 * - mode schedule
 * - swing trajectory planner
 * - (optional) adapt targetTrajectories to ground height
 *
 * IMPORTANT: modifyReferences() may "legally" modify the passed TargetTrajectories (e.g., ground height adapt).
 * We therefore keep a modified copy for subsequent cost evaluation to avoid mismatch.
 */
void OfflineQEvaluator::updateReference_(const OfflineSample& sample) {
  if (!refMgr_) throw std::runtime_error("[OfflineQEvaluator] refMgr_ is null.");

  const ocs2::scalar_t tf = sample.t0 + sample.horizon;

  // updateReferencesOffline will:
  // 1) generate modeSchedule based on gaitSchedule
  // 2) adapt targetTraj to ground height (if enabled in code)
  // 3) swingTrajectoryPlanner.update(modeSchedule, terrainHeight)
  // and write back to referenceManager's internal modeSchedule_
  //
  // NOTE: we keep a local modified copy so we can pass the *effective* target to cost evaluation.
  targetTrajModified_ = sample.target;

  refMgr_->updateReferencesOffline(sample.t0, tf, sample.x0, sample.initMode, targetTrajModified_, modeSchedule_);
}

ocs2::scalar_t OfflineQEvaluator::evaluateQ(const OfflineSample& sample) {
  updateReference_(sample);

  const size_t N = sample.uSeq.size();
  if (N == 0) throw std::runtime_error("[OfflineQEvaluator] uSeq is empty.");

  const ocs2::scalar_t tf = sample.t0 + sample.horizon;

  // ZOH controller from logged u_seq
  //
  // IMPORTANT: rollout integrator timeStep may be smaller and not aligned with sample.dt.
  // Therefore ZohController MUST be time-grid based (preferred), i.e. it maps any time t
  // to the correct discrete index k using a provided timeGrid (N+1).
  //
  // If you have the time grid (recommended): pass sample.timeGrid (N+1 timestamps).
  // Otherwise fallback to (t0, dt) mapping (less accurate when integrator step != dt).
  ZohController ctrl;
  if (!sample.timeGrid.empty()) {
    // sample.timeGrid should be size N+1, e.g. ref_times or obs_time[k:k+N+1]
    ctrl.setTrajectory(sample.timeGrid, sample.uSeq);
  } else {
    // Fallback: not recommended, but keeps pipeline running if you haven't wired timeGrid yet.
    ctrl = ZohController(sample.t0, sample.dt, sample.uSeq);
  }

  // outputs from rollout
  ocs2::scalar_array_t timeTrajectory;
  ocs2::size_array_t postEventIndicesStock;
  ocs2::vector_array_t stateTrajectory;
  ocs2::vector_array_t inputTrajectory;

  // Rollout from t0 to tf using the provided controller and the modeSchedule (hybrid).
  // This will internally integrate with rolloutSettings_ (can be smaller dt than control dt),
  // and query ctrl.computeInput(t, x) whenever needed.
  //
  // postEventIndicesStock: indices of post-event states (mode switches). We don't use it in A-stage.
  rollout_->run(sample.t0, sample.x0, tf, &ctrl, modeSchedule_, timeTrajectory, postEventIndicesStock, stateTrajectory, inputTrajectory);

  // --- debug printouts (rollout statistics) ---
  if (timeTrajectory.empty()) throw std::runtime_error("[OfflineQEvaluator] rollout returned empty timeTrajectory.");
  if (stateTrajectory.empty()) throw std::runtime_error("[OfflineQEvaluator] rollout returned empty stateTrajectory.");
  const auto t_front = timeTrajectory.front();
  const auto t_back = timeTrajectory.back();

  ocs2::scalar_t dt_min = 1e9, dt_max = 0.0, dt_sum = 0.0;
  for (size_t i = 0; i + 1 < timeTrajectory.size(); ++i) {
    const ocs2::scalar_t dti = timeTrajectory[i + 1] - timeTrajectory[i];
    if (dti > 0.0) {
      dt_min = std::min(dt_min, dti);
      dt_max = std::max(dt_max, dti);
      dt_sum += dti;
    }
  }
  const ocs2::scalar_t dt_mean = (timeTrajectory.size() > 1) ? (dt_sum / static_cast<ocs2::scalar_t>(timeTrajectory.size() - 1)) : 0.0;

  std::cerr << "\n######################### OfflineQEvaluator rollout stats #################################\n"
            << "  desired: t0=" << sample.t0 << " tf=" << tf << " horizon=" << sample.horizon << "\n"
            << "  got:     t_front=" << t_front << " t_back=" << t_back << "\n"
            << "  sizes:   time=" << timeTrajectory.size() << " state=" << stateTrajectory.size() << " input=" << inputTrajectory.size()
            << " (may differ by OCS2 version)\n"
            << "  dt:      min=" << dt_min << " mean=" << dt_mean << " max=" << dt_max << "  (sample.dt=" << sample.dt << ")\n"
            << std::endl;

  // accumulate Q (stage + terminal)
  return accumulateCost_(timeTrajectory, stateTrajectory, ctrl);
}

ocs2::scalar_t OfflineQEvaluator::accumulateCost_(const ocs2::scalar_array_t& timeTrajectory,
                                                  const ocs2::vector_array_t& stateTrajectory,
                                                  ZohController& ctrl) {
  if (!problem_) throw std::runtime_error("[OfflineQEvaluator] problem_ is null.");
  if (!problem_->preComputationPtr) throw std::runtime_error("[OfflineQEvaluator] preComputationPtr is null.");
  if (timeTrajectory.size() != stateTrajectory.size()) throw std::runtime_error("[OfflineQEvaluator] time/state trajectory size mismatch.");

  auto& preComp = *problem_->preComputationPtr;

  ocs2::scalar_t Q = 0.0;

  // --- stage costs ---
  // Use trapezoidal integration over variable rollout steps:
  // sum_i 0.5*(l_i + l_{i+1})*dt_i  (robust across rollout internal dt)
  for (size_t i = 0; i + 1 < timeTrajectory.size(); ++i) {
    const ocs2::scalar_t t0 = timeTrajectory[i];
    const ocs2::scalar_t t1 = timeTrajectory[i + 1];
    const ocs2::scalar_t dt = t1 - t0;
    if (dt <= 0.0) continue;

    const ocs2::vector_t& x0 = stateTrajectory[i];
    const ocs2::vector_t& x1 = stateTrajectory[i + 1];

    // ZOH input at the left endpoint (consistent with rollout dynamics using ctrl)
    const ocs2::vector_t u0 = ctrl.computeInput(t0, x0);
    const ocs2::vector_t u1 = ctrl.computeInput(t1, x1);  // optional, improves integration smoothness

    // precomp + cost at t0
    preComp.request(ocs2::Request::Cost, t0, x0, u0);
    const ocs2::scalar_t l0 = problem_->costPtr->getValue(t0, x0, u0, targetTrajModified_, preComp);

    // precomp + cost at t1
    preComp.request(ocs2::Request::Cost, t1, x1, u1);
    const ocs2::scalar_t l1 = problem_->costPtr->getValue(t1, x1, u1, targetTrajModified_, preComp);

    Q += 0.5 * (l0 + l1) * dt;
  }

  // --- terminal cost ---
  const ocs2::scalar_t tN = timeTrajectory.back();
  const ocs2::vector_t& xN = stateTrajectory.back();

  // NOTE: some OCS2 versions expect a valid u; use zeros(nu) when needed
  preComp.request(ocs2::Request::Cost, tN, xN, ocs2::vector_t::Zero(static_cast<long>(robot_model_->getInputDim())));
  Q += problem_->finalCostPtr->getValue(tN, xN, targetTrajModified_, preComp);

  return Q;
}

QAndGrad OfflineQEvaluator::evaluateQAndGradFixedGrid(const OfflineSample& sample, const ocs2::vector_t& theta) {
  // reference + mode schedule
  updateReference_(sample);

  const size_t N = sample.uSeq.size();
  if (N == 0) throw std::runtime_error("[OfflineQEvaluator] uSeq is empty.");

  const int nx = static_cast<int>(sample.x0.size());
  const int nu = static_cast<int>(sample.uSeq.front().size());

  // ---- ThetaLayout(v2) ----
  const size_t baseDim = 12;  // base: single link "mid360_link" => 12 weights
  const size_t comDim = 2;    // ICP xy
  const size_t numSwingFeet = 2;
  const size_t numTorqueLegs = 2;

  const ThetaLayout layout = ThetaLayout::make(nx, nu, baseDim, comDim, numSwingFeet, numTorqueLegs);
  layout.checkSize(theta);

  // controller (ZOH)
  ZohController ctrl;
  if (!sample.timeGrid.empty())
    ctrl.setTrajectory(sample.timeGrid, sample.uSeq);
  else
    ctrl = ZohController(sample.t0, sample.dt, sample.uSeq);

  // x in not needed in ZOH, but we need to pass it to get correct u(t) from ctrl.computeInput(t, x)
  auto u_of_t = [&](ocs2::scalar_t t, const ocs2::vector_t& x) -> ocs2::vector_t { return ctrl.computeInput(t, x); };

  // rollout on fixed grid with sensitivity
  constexpr size_t idx_hlin_dot = 0;
  constexpr size_t idx_hang_dot = 3;

  FixedGridRK4Rollout integ(*problem_->dynamicsPtr, idx_hlin_dot, idx_hang_dot);
  integ.setMomentumGains(layout.theta_hl(theta), layout.theta_ha(theta));

  std::vector<ocs2::matrix_t> S_grid;  // (N+1) * (nx x 6)
  const FixedGridRolloutResult roll = integ.run(sample.t0, sample.x0, sample.dt, N, modeSchedule_, u_of_t, &S_grid);

  // // ---- DEBUG: validate sensitivity S_N by finite difference on xN ----
  // const double eps = 1e-6;
  // auto run_xN = [&](const ocs2::vector_t& th_hl, const ocs2::vector_t& th_ha) {
  //   FixedGridRK4Rollout integ2(*problem_->dynamicsPtr, 0, 3);
  //   integ2.setMomentumGains(th_hl, th_ha);
  //   std::vector<ocs2::matrix_t> S_dummy;
  //   auto roll2 = integ2.run(sample.t0, sample.x0, sample.dt, N, modeSchedule_, u_of_t, &S_dummy);
  //   return roll2.x.back();
  // };

  // const ocs2::vector_t hl0 = layout.theta_hl(theta);
  // const ocs2::vector_t ha0 = layout.theta_ha(theta);
  // const ocs2::vector_t xN0 = roll.x.back();
  // const ocs2::matrix_t SN  = S_grid.back(); // nx x 6

  // for (int i = 0; i < 6; ++i) {
  //   ocs2::vector_t hlp = hl0, hlg = hl0;
  //   ocs2::vector_t hap = ha0, hag = ha0;
  //   if (i < 3) { hlp(i) += eps; hlg(i) -= eps; }
  //   else { hap(i-3) += eps; hag(i-3) -= eps; }

  //   const ocs2::vector_t xNp = run_xN(hlp, hap);
  //   const ocs2::vector_t xNm = run_xN(hlg, hag);
  //   const ocs2::vector_t Sfd = (xNp - xNm) / (2.0 * eps);

  //   const double err = (SN.col(i) - Sfd).norm() / std::max(1.0, Sfd.norm());
  //   std::cerr << "[S-check] col " << i
  //             << "  rel_err=" << err
  //             << "  |SN|=" << SN.col(i).norm()
  //             << "  |Sfd|=" << Sfd.norm()
  //             << std::endl;
  // }

  // build RefPackGrid on fixed grid from sample
  RefPackGrid refGrid;
  refGrid.t0 = sample.t0;
  refGrid.dt = sample.dt;
  refGrid.resize(N + 1);

  for (size_t k = 0; k <= N; ++k) {
    const ocs2::scalar_t tk = roll.t[k];
    const auto& x_meas_k = sample.xMeasSeq.empty() ? sample.x0 : sample.xMeasSeq[k];

    auto& rk = refGrid[k];
    rk.t = tk;
    rk.phase = static_cast<double>(refMgr_->getPhaseVariable(tk));
    rk.contactFlags = refMgr_->getContactFlags(tk);

    rk.xNominal = refMgr_->getDesiredState(targetTrajModified_, x_meas_k, tk);
    rk.uNominal = weightCompensatingInput(*pinocchio_, rk.contactFlags, *robot_model_);
  }

  // ---- accumulate Q and grad ----
  QAndGrad out;
  out.Q = 0.0;
  out.grad = ocs2::vector_t::Zero(static_cast<long>(layout.dim_total()));

  // ---- stage sum ----
  for (size_t k = 0; k < N; ++k) {
    const ocs2::scalar_t tk = roll.t[k];
    const ocs2::vector_t& xk = roll.x[k];
    const ocs2::vector_t uk = ctrl.computeInput(tk, xk);
    const RefPackKnot& refk = refGrid[k];

    // value
    const double lk = costBundle_->stage_value(tk, xk, uk, refk, theta, layout);
    out.Q += lk * sample.dt;

    // direct grad wrt theta_cost (full-dim)
    out.grad.noalias() += costBundle_->stage_grad_theta(tk, xk, uk, refk, theta, layout) * sample.dt;

    // chain for dyn theta via S: (dℓ/dx)^T * S
    const ocs2::vector_t dldx = costBundle_->stage_grad_x(tk, xk, uk, refk, theta, layout);  // (nx)
    const ocs2::matrix_t& Sk = S_grid[k];                                                    // (nx x 6)
    const ocs2::vector_t dldtheta_dyn = Sk.transpose() * dldx;                               // (6)
    out.grad.segment(static_cast<long>(layout.off_hl), 6) += dldtheta_dyn * sample.dt;
  }

  // ---- terminal ----
  {
    const ocs2::scalar_t tN = roll.t.back();
    const ocs2::vector_t& xN = roll.x.back();
    const RefPackKnot& refkN = refGrid[N];

    // value
    out.Q += costBundle_->terminal_value(tN, xN, refkN, theta, layout);

    // direct grad wrt theta_cost
    out.grad.noalias() += costBundle_->terminal_grad_theta(tN, xN, refkN, theta, layout);

    // chain dyn theta
    const ocs2::vector_t dTdx = costBundle_->terminal_grad_x(tN, xN, refkN, theta, layout);  // (nx)
    const ocs2::matrix_t& SN = S_grid.back();                                                // (nx x 6)
    const ocs2::vector_t dTdtheta_dyn = SN.transpose() * dTdx;                               // (6)
    out.grad.segment(static_cast<long>(layout.off_hl), 6) += dTdtheta_dyn;
  }

  return out;
}

}  // namespace ocs2::humanoid_cost_matching
