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

#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/multi_array_dimension.hpp>

#include <ocs2_core/ComputationRequest.h>
#include <ocs2_ros2_interfaces/mpc/MPC_ROS_Interface.h>
#include <ocs2_ros2_interfaces/synchronized_module/RosReferenceManager.h>
#include <ocs2_sqp/SqpMpc.h>
#include <ocs2_ros2_msgs/msg/mpc_flattened_controller.hpp>
#include <ocs2_ros2_msgs/msg/mpc_observation.hpp>

#include <humanoid_centroidal_mpc/CentroidalMpcInterface.h>
#include <humanoid_centroidal_mpc/command/CentroidalMpcTargetTrajectoriesCalculator.h>
#include "humanoid_common_mpc_ros2/ros_comm/Ros2ProceduralMpcMotionManager.h"

#include "humanoid_common_mpc/reference_manager/SwitchedModelReferenceManager.h"
// #include "humanoid_cost_matching_cpp/refpack_logger.hpp"

using namespace ocs2;
using namespace ocs2::humanoid;

static std_msgs::msg::MultiArrayLayout makeLayout1D(int dim0, const char* label0) {
  std_msgs::msg::MultiArrayLayout layout;
  layout.dim.resize(1);
  layout.dim[0].label = label0;
  layout.dim[0].size = static_cast<uint32_t>(dim0);
  layout.dim[0].stride = static_cast<uint32_t>(dim0);
  layout.data_offset = 0;
  return layout;
}

static std_msgs::msg::MultiArrayLayout makeLayout2D(int dim0, const char* label0, int dim1, const char* label1) {
  std_msgs::msg::MultiArrayLayout layout;
  layout.dim.resize(2);
  layout.dim[0].label = label0;
  layout.dim[0].size = static_cast<uint32_t>(dim0);
  layout.dim[0].stride = static_cast<uint32_t>(dim0 * dim1);
  layout.dim[1].label = label1;
  layout.dim[1].size = static_cast<uint32_t>(dim1);
  layout.dim[1].stride = static_cast<uint32_t>(dim1);
  layout.data_offset = 0;
  return layout;
}

int main(int argc, char** argv) {
  std::vector<std::string> programArgs;
  programArgs = rclcpp::remove_ros_arguments(argc, argv);
  if (programArgs.size() < 5) {
    throw std::runtime_error("No robot name, config folder, target command file, or description name specified. Aborting.");
  }

  const std::string robotName(argv[1]);
  const std::string taskFile(argv[2]);
  const std::string referenceFile(argv[3]);
  const std::string urdfFile(argv[4]);
  const std::string gaitFile(argv[5]);

  rclcpp::init(argc, argv);

  // Robot interface
  CentroidalMpcInterface interface(taskFile, urdfFile, referenceFile, true);

  // MPC
  SqpMpc mpc(interface.mpcSettings(), interface.sqpSettings(), interface.getOptimalControlProblem(), interface.getInitializer());

  // Launch MPC ROS node
  rclcpp::Node::SharedPtr nodeHandle = std::make_shared<rclcpp::Node>(robotName + "_mpc");

  auto qos = rclcpp::QoS(1);
  qos.best_effort();

  // Reference and motion management for Procedural MPC
  CentroidalMpcTargetTrajectoriesCalculator mpcTargetTrajectoriesCalculator(
      referenceFile, interface.getMpcRobotModel(), interface.getPinocchioInterface(), interface.getCentroidalModelInfo(),
      interface.mpcSettings().timeHorizon_);
  ProceduralMpcMotionManager::VelocityTargetToTargetTrajectories targetTrajectoriesFunc =
      [&mpcTargetTrajectoriesCalculator](const vector4_t& velocityTarget, scalar_t initTime, scalar_t finalTime,
                                         const vector_t& initState) mutable {
        return mpcTargetTrajectoriesCalculator.commandedVelocityToTargetTrajectories(velocityTarget, initTime, initState);
      };

  auto switchedRefMgrPtr = interface.getSwitchedModelReferenceManagerPtr();
  if (!switchedRefMgrPtr) {
    throw std::runtime_error("interface.getSwitchedModelReferenceManagerPtr() returned null.");
  }

  auto ros2ProceduralMpcMotionManager = std::make_shared<Ros2ProceduralMpcMotionManager>(
      gaitFile, referenceFile, switchedRefMgrPtr, interface.getMpcRobotModel(), targetTrajectoriesFunc);

  ros2ProceduralMpcMotionManager->subscribe(nodeHandle, qos);

  mpc.getSolverPtr()->setReferenceManager(interface.getReferenceManagerPtr());
  mpc.getSolverPtr()->addSynchronizedModule(ros2ProceduralMpcMotionManager);

  MPC_ROS_Interface mpcNode(mpc, robotName);

  // ===== Begin for cost-matching logger =====
  // ---- Publish MPC total cost from /<robot>/mpc_policy.performance_indices.cost ----
  const std::string policyTopic = "/" + robotName + "/mpc_policy";
  const std::string costTopic = "/" + robotName + "/mpc_total_cost";

  auto totalCostPub = nodeHandle->create_publisher<std_msgs::msg::Float32>(costTopic, 10);

  auto policySub = nodeHandle->create_subscription<ocs2_ros2_msgs::msg::MpcFlattenedController>(
      policyTopic, qos, [totalCostPub](const ocs2_ros2_msgs::msg::MpcFlattenedController::SharedPtr msg) {
        std_msgs::msg::Float32 out;
        out.data = msg->performance_indices.cost;
        totalCostPub->publish(out);
      });

  RCLCPP_INFO(nodeHandle->get_logger(), "[CostMatching] republish %s -> %s", policyTopic.c_str(), costTopic.c_str());

  // ---- lightweight plan target trajectories publisher (avoid logging /mpc_policy in python) ----
  const std::string planKnotTimeTopic = "/" + robotName + "/plan_target_knots_time";
  const std::string planKnotStateTopic = "/" + robotName + "/plan_target_knots_state";
  const std::string planKnotInputTopic = "/" + robotName + "/plan_target_knots_input";

  auto planTimePub = nodeHandle->create_publisher<std_msgs::msg::Float64MultiArray>(planKnotTimeTopic, 10);
  auto planStatePub = nodeHandle->create_publisher<std_msgs::msg::Float64MultiArray>(planKnotStateTopic, 10);
  auto planInputPub = nodeHandle->create_publisher<std_msgs::msg::Float64MultiArray>(planKnotInputTopic, 10);

  // dims inferred from /mpc_policy once
  auto dimsMtx = std::make_shared<std::mutex>();
  auto nxPtr = std::make_shared<int>(-1);
  auto nuPtr = std::make_shared<int>(-1);

  RCLCPP_INFO(nodeHandle->get_logger(), "[CostMatching] publish lightweight target knots:");
  RCLCPP_INFO(nodeHandle->get_logger(), "  %s  (Float64MultiArray 3)", planKnotTimeTopic.c_str());
  RCLCPP_INFO(nodeHandle->get_logger(), "  %s (Float64MultiArray 3xnx)", planKnotStateTopic.c_str());
  RCLCPP_INFO(nodeHandle->get_logger(), "  %s (Float64MultiArray 3xnu)", planKnotInputTopic.c_str());

  // ---- measured stage cost publisher ----
  const std::string obsTopic = "/" + robotName + "/mpc_observation";
  const std::string measTopic = "/" + robotName + "/measured_stage_cost";

  auto measCostPub = nodeHandle->create_publisher<std_msgs::msg::Float32>(measTopic, rclcpp::QoS(10));

  // pointers to OCP pieces
  const auto* stageCostPtr = interface.getOptimalControlProblem().costPtr.get();
  const auto refMgrPtr = interface.getReferenceManagerPtr();  // keep (unused below), avoid touching refMgr in callbacks

  // ---- Cache target trajectories from /mpc_policy (thread-safe snapshot) ----
  auto targetTrajMtx = std::make_shared<std::mutex>();
  auto cachedTargetTraj = std::make_shared<std::shared_ptr<ocs2::TargetTrajectories>>(nullptr);

  auto policySub2 = nodeHandle->create_subscription<ocs2_ros2_msgs::msg::MpcFlattenedController>(
      policyTopic, qos,
      [targetTrajMtx, cachedTargetTraj, planTimePub, planStatePub, planInputPub, dimsMtx, nxPtr,
       nuPtr](const ocs2_ros2_msgs::msg::MpcFlattenedController::SharedPtr msg) {
        // Build TargetTrajectories from plan_target_trajectories (typically 3 knots)
        const auto& pt = msg->plan_target_trajectories;

        const size_t M = pt.time_trajectory.size();
        if (M == 0 || pt.state_trajectory.size() == 0 || pt.input_trajectory.size() == 0) {
          return;
        }

        const size_t Mm = std::min(M, std::min(pt.state_trajectory.size(), pt.input_trajectory.size()));
        if (Mm == 0) return;

        // infer dims once
        int nx = -1;
        int nu = -1;
        {
          std::lock_guard<std::mutex> lk(*dimsMtx);
          if (*nxPtr < 0 && !pt.state_trajectory.empty()) {
            *nxPtr = static_cast<int>(pt.state_trajectory[0].value.size());
          }
          if (*nuPtr < 0 && !pt.input_trajectory.empty()) {
            *nuPtr = static_cast<int>(pt.input_trajectory[0].value.size());
          }
          nx = *nxPtr;
          nu = *nuPtr;
        }

        // ----- publish lightweight knots (always 3) -----
        const double NaN = std::numeric_limits<double>::quiet_NaN();
        const size_t K = std::min<size_t>(3, Mm);

        // time (3,)
        std_msgs::msg::Float64MultiArray tmsg;
        tmsg.layout = makeLayout1D(3, "knot");
        tmsg.data.assign(3, NaN);
        for (size_t i = 0; i < K; ++i) {
          tmsg.data[i] = static_cast<double>(pt.time_trajectory[i]);
        }
        planTimePub->publish(tmsg);

        // state (3,nx)
        if (nx > 0) {
          std_msgs::msg::Float64MultiArray xmsg;
          xmsg.layout = makeLayout2D(3, "knot", nx, "x");
          xmsg.data.assign(static_cast<size_t>(3 * nx), NaN);
          for (size_t i = 0; i < K; ++i) {
            const auto& xs = pt.state_trajectory[i].value;
            const size_t dx = std::min<size_t>(static_cast<size_t>(nx), xs.size());
            for (size_t j = 0; j < dx; ++j) {
              xmsg.data[i * static_cast<size_t>(nx) + j] = static_cast<double>(xs[j]);
            }
          }
          planStatePub->publish(xmsg);
        }

        // input (3,nu)
        if (nu > 0) {
          std_msgs::msg::Float64MultiArray umsg;
          umsg.layout = makeLayout2D(3, "knot", nu, "u");
          umsg.data.assign(static_cast<size_t>(3 * nu), NaN);
          for (size_t i = 0; i < K; ++i) {
            const auto& us = pt.input_trajectory[i].value;
            const size_t du = std::min<size_t>(static_cast<size_t>(nu), us.size());
            for (size_t j = 0; j < du; ++j) {
              umsg.data[i * static_cast<size_t>(nu) + j] = static_cast<double>(us[j]);
            }
          }
          planInputPub->publish(umsg);
        }

        // ----- also build cached TargetTrajectories for measured_stage_cost -----
        std::vector<ocs2::scalar_t> t;
        t.reserve(Mm);
        std::vector<ocs2::vector_t> x;
        x.reserve(Mm);
        std::vector<ocs2::vector_t> u;
        u.reserve(Mm);

        for (size_t i = 0; i < Mm; ++i) {
          t.push_back(static_cast<ocs2::scalar_t>(pt.time_trajectory[i]));

          const auto& xs = pt.state_trajectory[i].value;
          ocs2::vector_t xi(xs.size());
          for (size_t k = 0; k < xs.size(); ++k) xi(static_cast<int>(k)) = static_cast<ocs2::scalar_t>(xs[k]);
          x.push_back(std::move(xi));

          const auto& usv = pt.input_trajectory[i].value;
          ocs2::vector_t ui(usv.size());
          for (size_t k = 0; k < usv.size(); ++k) ui(static_cast<int>(k)) = static_cast<ocs2::scalar_t>(usv[k]);
          u.push_back(std::move(ui));
        }

        auto traj = std::make_shared<ocs2::TargetTrajectories>(t, x, u);
        {
          std::lock_guard<std::mutex> lk(*targetTrajMtx);
          *cachedTargetTraj = std::move(traj);
        }
      });

  std::shared_ptr<ocs2::PreComputation> preCompPtr;
  if (interface.getOptimalControlProblem().preComputationPtr) {
    preCompPtr.reset(interface.getOptimalControlProblem().preComputationPtr->clone());
  } else {
    preCompPtr = std::make_shared<ocs2::PreComputation>();
  }

  std::function<void(ocs2_ros2_msgs::msg::MpcObservation::SharedPtr)> obsCb =
      [measCostPub, stageCostPtr, preCompPtr, targetTrajMtx, cachedTargetTraj](ocs2_ros2_msgs::msg::MpcObservation::SharedPtr msg) {
        const ocs2::scalar_t t = static_cast<ocs2::scalar_t>(msg->time);

        ocs2::vector_t x(msg->state.value.size());
        for (size_t i = 0; i < msg->state.value.size(); ++i) {
          x(static_cast<int>(i)) = static_cast<ocs2::scalar_t>(msg->state.value[i]);
        }

        ocs2::vector_t u(msg->input.value.size());
        for (size_t i = 0; i < msg->input.value.size(); ++i) {
          u(static_cast<int>(i)) = static_cast<ocs2::scalar_t>(msg->input.value[i]);
        }

        // ---- Use cached target trajectories from /mpc_policy (avoid refMgr race) ----
        std::shared_ptr<ocs2::TargetTrajectories> targetTrajPtr;
        {
          std::lock_guard<std::mutex> lk(*targetTrajMtx);
          targetTrajPtr = *cachedTargetTraj;
        }
        if (!targetTrajPtr || targetTrajPtr->empty()) {
          // Policy not received yet
          return;
        }

        // precomputation for cost terms
        preCompPtr->request(ocs2::Request::Cost, t, x, u);

        // measured stage cost
        const ocs2::scalar_t l = stageCostPtr->getValue(t, x, u, *targetTrajPtr, *preCompPtr);
        if (!std::isfinite(static_cast<double>(l))) {
          return;
        }

        std_msgs::msg::Float32 out;
        out.data = static_cast<float>(l);
        measCostPub->publish(out);
      };

  auto obsSub = nodeHandle->create_subscription<ocs2_ros2_msgs::msg::MpcObservation>(obsTopic, rclcpp::QoS(10).best_effort(), obsCb);

  RCLCPP_INFO(nodeHandle->get_logger(), "[CostMatching] publish %s (Float32) from %s", measTopic.c_str(), obsTopic.c_str());

  // ===== End for cost-matching logger =====

  mpcNode.launchNodes(nodeHandle, qos);

  return 0;
}
