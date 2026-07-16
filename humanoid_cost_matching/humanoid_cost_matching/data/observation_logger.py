#!/usr/bin/env python3
import os
import time
import argparse
from dataclasses import dataclass
from typing import Optional, List

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
    qos_profile_sensor_data,
)

# --- Hardcoded Imports for Specific Message Types ---
from ocs2_ros2_msgs.msg import MpcObservation
from humanoid_mpc_msgs.msg import WalkingVelocityCommand
from std_msgs.msg import Float32
from std_msgs.msg import Float64MultiArray


@dataclass
class LogBuffer:
    t: List[float]
    state: List[np.ndarray]
    u: List[np.ndarray]
    mode: List[int]
    mpc_total_cost: List[float]
    measured_stage_cost: List[float]
    cmd_vel: List[np.ndarray]

    # Plan target trajectories (fixed 3 knots per step)
    target_time: List[np.ndarray]  # (3,)
    target_state: List[np.ndarray]  # (3, stateDim)
    target_input: List[np.ndarray]  # (3, inputDim)

    def __init__(self):
        self.t = []
        self.state = []
        self.u = []
        self.mode = []
        self.mpc_total_cost = []
        self.measured_stage_cost = []
        self.cmd_vel = []

        self.target_time = []
        self.target_state = []
        self.target_input = []


class MPCObservationLogger(Node):
    """
    Logs:
      - /g1/mpc_observation
      - /g1/mpc_total_cost
      - /g1/measured_stage_cost
      - /g1/plan_target_knots_time
      - /g1/plan_target_knots_state
      - /g1/plan_target_knots_input
      - /humanoid/walking_velocity_command
    """

    def __init__(
        self,
        out_dir: str,
        max_steps: int,
        run_id: str = "",
    ):
        super().__init__("mpc_observation_logger")

        self.out_dir = out_dir
        self.max_steps = max_steps
        self.run_id = run_id

        # --- Hardcoded Configurations ---
        self.robot_name = "g1"
        self.cmd_dim = 4

        # Fixed Topics
        self.obs_topic = f"/{self.robot_name}/mpc_observation"
        self.mpc_total_cost_topic = f"/{self.robot_name}/mpc_total_cost"
        self.measured_stage_cost_topic = f"/{self.robot_name}/measured_stage_cost"
        self.cmd_topic = "/humanoid/walking_velocity_command"

        # Lightweight plan target topics
        self.plan_time_topic = f"/{self.robot_name}/plan_target_knots_time"
        self.plan_state_topic = f"/{self.robot_name}/plan_target_knots_state"
        self.plan_input_topic = f"/{self.robot_name}/plan_target_knots_input"

        os.makedirs(self.out_dir, exist_ok=True)

        self.qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        # Buffers for "latest" scalar topics
        self.last_mpc_total_cost = np.nan
        self.last_measured_stage_cost = np.nan

        # Command cache
        self.last_cmd = np.zeros(self.cmd_dim, dtype=np.float64)
        self.has_cmd = False

        # --- Plan target caches ---
        self.knot_count = 3
        self._state_dim = None
        self._input_dim = None

        self.last_target_time = np.full((self.knot_count,), np.nan, dtype=np.float64)
        self.last_target_state = None
        self.last_target_input = None
        self.got_total_cost = False
        self.got_measured_cost = False
        self.got_plan_time = False
        self.got_plan_state = False
        self.got_plan_input = False

        self.buf = LogBuffer()

        # --- Subscriptions ---
        self.sub_obs = self.create_subscription(
            MpcObservation, self.obs_topic, self._on_obs, qos_profile_sensor_data
        )
        self.sub_total_cost = self.create_subscription(
            Float32, self.mpc_total_cost_topic, self._on_total_cost, self.qos
        )
        self.sub_meas_cost = self.create_subscription(
            Float32,
            self.measured_stage_cost_topic,
            self._on_measured_stage_cost,
            self.qos,
        )

        # Lightweight plan knots
        self.sub_plan_time = self.create_subscription(
            Float64MultiArray, self.plan_time_topic, self._on_plan_time, self.qos
        )
        self.sub_plan_state = self.create_subscription(
            Float64MultiArray, self.plan_state_topic, self._on_plan_state, self.qos
        )
        self.sub_plan_input = self.create_subscription(
            Float64MultiArray, self.plan_input_topic, self._on_plan_input, self.qos
        )

        self.sub_cmd = self.create_subscription(
            WalkingVelocityCommand, self.cmd_topic, self._on_cmd, self.qos
        )

        self.start_wall = time.time()
        self.get_logger().info("Logger started.")
        self.get_logger().info(f"  Output Dir:       {self.out_dir}")
        self.get_logger().info(
            f"  Run ID:           {self.run_id if self.run_id else 'None (using timestamp)'}"
        )
        self.get_logger().info(f"  Max Steps:        {self.max_steps}")
        self.get_logger().info(f"  obs:              {self.obs_topic}")
        self.get_logger().info(f"  total cost:       {self.mpc_total_cost_topic}")
        self.get_logger().info(f"  measured stage:   {self.measured_stage_cost_topic}")
        self.get_logger().info(f"  plan time:        {self.plan_time_topic}")
        self.get_logger().info(f"  plan state:       {self.plan_state_topic}")
        self.get_logger().info(f"  plan input:       {self.plan_input_topic}")
        self.get_logger().info(f"  cmd topic:        {self.cmd_topic}")
        self.get_logger().info("Waiting for messages...")

    # ----------------------- callbacks -----------------------

    def _on_total_cost(self, msg: Float32):
        self.last_mpc_total_cost = float(msg.data)
        self.got_total_cost = True

    def _on_measured_stage_cost(self, msg: Float32):
        self.last_measured_stage_cost = float(msg.data)
        self.got_measured_cost = True

    def _on_cmd(self, msg: WalkingVelocityCommand):
        """
        Directly parse WalkingVelocityCommand.
        Vector: [vx, vy, pelvis_height, wz]
        """
        pelvis = float(getattr(msg, "desired_pelvis_height", np.nan))

        v = np.array(
            [
                float(msg.linear_velocity_x),
                float(msg.linear_velocity_y),
                pelvis,
                float(msg.angular_velocity_z),
            ],
            dtype=np.float64,
        )
        self.last_cmd[:] = v
        self.has_cmd = True

    def _on_plan_time(self, msg: Float64MultiArray):
        d = np.asarray(msg.data, dtype=np.float64)
        if d.size >= self.knot_count:
            self.last_target_time[:] = d[: self.knot_count]
        else:
            self.last_target_time[:] = np.nan
            self.last_target_time[: d.size] = d
        self.got_plan_time = True

    def _on_plan_state(self, msg: Float64MultiArray):
        d = np.asarray(msg.data, dtype=np.float64)
        # Expect layout 2D: (3,nx). If nx unknown, infer.
        if self._state_dim is None:
            if d.size % self.knot_count == 0:
                self._state_dim = int(d.size // self.knot_count)
        if self._state_dim is None or self._state_dim <= 0:
            return
        if self.last_target_state is None or self.last_target_state.shape != (
            self.knot_count,
            self._state_dim,
        ):
            self.last_target_state = np.full(
                (self.knot_count, self._state_dim), np.nan, dtype=np.float64
            )

        need = self.knot_count * self._state_dim
        if d.size < need:
            self.last_target_state[:] = np.nan
            self.last_target_state.flat[: d.size] = d
        else:
            self.last_target_state[:] = d[:need].reshape(
                (self.knot_count, self._state_dim)
            )
        self.got_plan_state = True

    def _on_plan_input(self, msg: Float64MultiArray):
        d = np.asarray(msg.data, dtype=np.float64)
        # Expect layout 2D: (3,nu). If nu unknown, infer.
        if self._input_dim is None:
            if d.size % self.knot_count == 0:
                self._input_dim = int(d.size // self.knot_count)
        if self._input_dim is None or self._input_dim <= 0:
            return
        if self.last_target_input is None or self.last_target_input.shape != (
            self.knot_count,
            self._input_dim,
        ):
            self.last_target_input = np.full(
                (self.knot_count, self._input_dim), np.nan, dtype=np.float64
            )

        need = self.knot_count * self._input_dim
        if d.size < need:
            self.last_target_input[:] = np.nan
            self.last_target_input.flat[: d.size] = d
        else:
            self.last_target_input[:] = d[:need].reshape(
                (self.knot_count, self._input_dim)
            )
        self.got_plan_input = True

    def _on_obs(self, msg: MpcObservation):
        if (not self.got_total_cost) or (not self.got_measured_cost):
            return
        if (
            (not self.got_plan_time)
            or (not self.got_plan_state)
            or (not self.got_plan_input)
        ):
            return
        t = float(msg.time)
        s = np.asarray(msg.state.value, dtype=np.float64)
        u = np.asarray(msg.input.value, dtype=np.float64)
        mode_int = self._parse_mode(msg.mode)

        # Init dims from obs if not set
        if self._state_dim is None:
            self._state_dim = int(s.shape[0])
        if self._input_dim is None:
            self._input_dim = int(u.shape[0])

        # Ensure target caches allocated with correct dims
        if self.last_target_state is None:
            self.last_target_state = np.full(
                (self.knot_count, self._state_dim), np.nan, dtype=np.float64
            )
        if self.last_target_input is None:
            self.last_target_input = np.full(
                (self.knot_count, self._input_dim), np.nan, dtype=np.float64
            )

        self.buf.t.append(t)
        self.buf.state.append(s)
        self.buf.u.append(u)
        self.buf.mode.append(mode_int)

        self.buf.mpc_total_cost.append(self.last_mpc_total_cost)
        self.buf.measured_stage_cost.append(self.last_measured_stage_cost)
        self.buf.cmd_vel.append(self.last_cmd.copy())

        # per-obs target traj snapshot (light topics, safe)
        self.buf.target_time.append(self.last_target_time.copy())
        self.buf.target_state.append(self.last_target_state.copy())
        self.buf.target_input.append(self.last_target_input.copy())

        n = len(self.buf.t)
        if n % 200 == 0:
            elapsed = time.time() - self.start_wall
            self.get_logger().info(
                f"Logged {n} steps ({elapsed:.1f}s). mode={mode_int}, t={t:.3f}, "
                f"meas_l={self.last_measured_stage_cost:.4g}"
            )

        if self.max_steps > 0 and n >= self.max_steps:
            self.get_logger().info(
                f"Reached max_steps={self.max_steps}. Saving and shutting down."
            )
            self.save_and_shutdown()

    # ----------------------- utils -----------------------

    def _parse_mode(self, mode_raw) -> int:
        if isinstance(mode_raw, (bytes, bytearray)):
            return int(mode_raw[0]) if len(mode_raw) > 0 else 0
        if isinstance(mode_raw, str):
            return ord(mode_raw[0]) if len(mode_raw) > 0 else 0
        try:
            return int(mode_raw)
        except Exception:
            return 0

    # ----------------------- save -----------------------

    def save_and_shutdown(self):
        path = self._save_npz()
        self.get_logger().info(f"Saved: {path}")
        rclpy.shutdown()

    def _save_npz(self) -> str:
        if self.run_id:
            filename = f"obs_{self.run_id}.npz"
        else:
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            filename = f"obs_{ts}.npz"

        path = os.path.join(self.out_dir, filename)

        t = np.asarray(self.buf.t, dtype=np.float64)

        state = (
            np.stack(self.buf.state, axis=0)
            if len(self.buf.state) > 0
            else np.zeros((0, 0), dtype=np.float64)
        )
        u = (
            np.stack(self.buf.u, axis=0)
            if len(self.buf.u) > 0
            else np.zeros((0, 0), dtype=np.float64)
        )
        mode = np.asarray(self.buf.mode, dtype=np.int32)

        cmd_vel_stack = (
            np.stack(self.buf.cmd_vel, axis=0)
            if len(self.buf.cmd_vel) > 0
            else np.zeros((0, 0), dtype=np.float64)
        )

        target_time = (
            np.stack(self.buf.target_time, axis=0)
            if len(self.buf.target_time) > 0
            else np.zeros((0, self.knot_count), dtype=np.float64)
        )
        target_state = (
            np.stack(self.buf.target_state, axis=0)
            if len(self.buf.target_state) > 0
            else np.zeros((0, self.knot_count, 0), dtype=np.float64)
        )
        target_input = (
            np.stack(self.buf.target_input, axis=0)
            if len(self.buf.target_input) > 0
            else np.zeros((0, self.knot_count, 0), dtype=np.float64)
        )

        save_dict = {
            "time": t,
            "state": state,
            "input": u,
            "mode": mode,
            "mpc_total_cost": np.asarray(self.buf.mpc_total_cost, dtype=np.float64),
            "measured_stage_cost": np.asarray(
                self.buf.measured_stage_cost, dtype=np.float64
            ),
            "cmd_vel": cmd_vel_stack,
            "target_timeTrajectory": target_time,
            "target_stateTrajectory": target_state,
            "target_inputTrajectory": target_input,
        }

        save_dict["meta_topics"] = np.array(
            [
                self.obs_topic,
                self.mpc_total_cost_topic,
                self.measured_stage_cost_topic,
                self.plan_time_topic,
                self.plan_state_topic,
                self.plan_input_topic,
                self.cmd_topic,
            ],
            dtype=object,
        )

        np.savez_compressed(path, **save_dict)
        return path


def main():
    parser = argparse.ArgumentParser()

    default_out_dir = (
        "/wb_humanoid_mpc_ws/src/wb_humanoid_mpc/humanoid_cost_matching/logs/runs/"
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default=default_out_dir,
        help="Base output directory for npz logs",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=5000,
        help="Stop after this many observations",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="",
        help="If provided, appends this ID to the out_dir path AND uses it for the filename (obs_<run_id>.npz)",
    )

    args = parser.parse_args()

    rclpy.init()
    node = MPCObservationLogger(
        out_dir=args.out_dir, max_steps=args.max_steps, run_id=args.run_id
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C received. Saving and exiting.")
        node.save_and_shutdown()


if __name__ == "__main__":
    main()
