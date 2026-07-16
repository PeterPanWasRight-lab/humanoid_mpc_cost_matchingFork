#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <vector>
#include "humanoid_cost_matching_cpp/offline/OfflineQEvaluator.h"

using namespace ocs2;
using namespace ocs2::humanoid_cost_matching;

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  const std::string taskFile =
      "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_centroidal_mpc/config/mpc/task_cost_matching.info";
  const std::string urdfFile = "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_description/urdf/g1_29dof.urdf";
  const std::string referenceFile =
      "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/robot_models/unitree_g1/g1_centroidal_mpc/config/command/reference.info";

  RCLCPP_INFO(rclcpp::get_logger("offline_q_eval_main"), "taskFile: %s", taskFile.c_str());
  RCLCPP_INFO(rclcpp::get_logger("offline_q_eval_main"), "urdfFile: %s", urdfFile.c_str());
  RCLCPP_INFO(rclcpp::get_logger("offline_q_eval_main"), "referenceFile: %s", referenceFile.c_str());

  OfflineQEvaluator eval(taskFile, urdfFile, referenceFile);

  OfflineSample sample;
  sample.t0 = 0.0;
  sample.dt = 0.02;
  sample.horizon = 60 * sample.dt;
  sample.initMode = 3;  // from obs.mode[k] in real use

  // Dummy dims (replace by actual dims)
  const int stateDim = 35;
  const int inputDim = 35;
  sample.x0 = vector_t::Zero(stateDim);

  // xMeasSeq length N+1=61
  sample.xMeasSeq.resize(61);
  for (auto& xMeas : sample.xMeasSeq) xMeas = vector_t::Zero(stateDim);

  // uSeq length N=60
  sample.uSeq.resize(60);
  for (auto& u : sample.uSeq) u = vector_t::Zero(inputDim);

  // TargetTrajectories: 3 knots (you logged)
  sample.target.timeTrajectory = {0.0, 0.7 * sample.horizon, sample.horizon};
  sample.target.stateTrajectory = {vector_t::Zero(stateDim), vector_t::Zero(stateDim), vector_t::Zero(stateDim)};
  sample.target.inputTrajectory = {vector_t::Zero(inputDim), vector_t::Zero(inputDim), vector_t::Zero(inputDim)};

  const auto Q = eval.evaluateQ(sample);
  std::cout << "Q = " << Q << "\n";
  return 0;
}
