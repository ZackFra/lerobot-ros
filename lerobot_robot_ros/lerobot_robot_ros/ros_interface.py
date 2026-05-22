# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import threading
import time

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import GripperCommand
from lerobot.utils.errors import DeviceNotConnectedError
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import Executor, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.publisher import Publisher
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .config import ActionType, GripperActionType, ROS2InterfaceConfig
from .moveit_servo import MoveIt2Servo

logger = logging.getLogger(__name__)

# JointTrajectoryController needs a non-zero segment time to track goals reliably (esp. Gazebo).
_TRAJ_TIME_250MS = Duration(sec=0, nanosec=250_000_000)
# Gripper JTC often benefits from a slightly longer segment to settle the last millimetre in Gazebo.
_GRIPPER_TRAJ_TIME = Duration(sec=0, nanosec=400_000_000)


def _clip_joint_positions(
    joint_positions: list[float],
    min_positions: list[float],
    max_positions: list[float],
) -> list[float]:
    return [
        max(lo, min(hi, pos))
        for pos, lo, hi in zip(joint_positions, min_positions, max_positions, strict=True)
    ]


class ROS2Interface:
    """Class to interface with a MoveIt2 manipulator.

    This class supports both JointGroupPositionController and JointTrajectoryController
    from ros2_control for arm control, depending on the configuration:

    - ActionType.JOINT_POSITION:
      Uses JointGroupPositionController.
      Publishes Float64MultiArray messages to '/position_controller/commands'

    - ActionType.JOINT_TRAJECTORY:
      Uses JointTrajectoryController.
      Publishes JointTrajectory messages to '/arm_controller/joint_trajectory'

    The gripper control also supports both trajectory and action-based control
    via the gripper_action_type configuration option.
    """

    def __init__(self, config: ROS2InterfaceConfig, action_type: ActionType):
        self.config = config
        self.action_type = action_type
        self.robot_node: Node | None = None
        self.pos_cmd_pub: Publisher | None = None
        self.traj_cmd_pub: Publisher | None = None
        self.gripper_action_client: ActionClient | None = None
        self.gripper_traj_pub: Publisher | None = None
        self.gripper_pos_cmd_pub: Publisher | None = None
        self.executor: Executor | None = None
        self.moveit2_servo: MoveIt2Servo | None = None
        self.executor_thread: threading.Thread | None = None
        self.is_connected = False
        self._last_joint_state: dict[str, dict[str, float]] | None = None
        self._filtered_jog_vel: list[float] | None = None
        self._leader_joint_step: dict[str, float] = {}

    def connect(self) -> None:
        if not rclpy.ok():
            rclpy.init()

        self._filtered_jog_vel = None
        self._leader_joint_step = {}
        self.robot_node = Node("moveit2_interface_node", namespace=self.config.namespace)
        if self.action_type == ActionType.JOINT_POSITION:
            self.pos_cmd_pub = self.robot_node.create_publisher(
                Float64MultiArray, self.config.arm_commands_topic, 10
            )
            self.gripper_pos_cmd_pub = self.robot_node.create_publisher(
                Float64MultiArray, self.config.gripper_commands_topic, 10
            )
        elif self.action_type == ActionType.JOINT_TRAJECTORY:
            self.traj_cmd_pub = self.robot_node.create_publisher(
                JointTrajectory, "/arm_controller/joint_trajectory", 10
            )
        elif self.action_type == ActionType.CARTESIAN_VELOCITY:
            self.moveit2_servo = MoveIt2Servo(
                node=self.robot_node,
                frame_id=self.config.base_link,
                callback_group=ReentrantCallbackGroup(),
            )
        elif self.action_type == ActionType.JOINT_JOG:
            self.moveit2_servo = MoveIt2Servo(
                node=self.robot_node,
                frame_id=self.config.base_link,
                callback_group=ReentrantCallbackGroup(),
            )

        if self.action_type == ActionType.JOINT_POSITION:
            pass  # gripper uses gripper_pos_cmd_pub
        elif self.action_type == ActionType.JOINT_JOG:
            if self.config.gripper_use_forward_commands:
                self.gripper_pos_cmd_pub = self.robot_node.create_publisher(
                    Float64MultiArray, self.config.gripper_commands_topic, 10
                )
            elif self.config.gripper_action_type == GripperActionType.TRAJECTORY:
                self.gripper_traj_pub = self.robot_node.create_publisher(
                    JointTrajectory, "/gripper_controller/joint_trajectory", 10
                )
        elif self.config.gripper_action_type == GripperActionType.TRAJECTORY:
            self.gripper_traj_pub = self.robot_node.create_publisher(
                JointTrajectory, "/gripper_controller/joint_trajectory", 10
            )
        else:
            self.gripper_action_client = ActionClient(
                self.robot_node,
                GripperCommand,
                "/gripper_controller/gripper_cmd",
                callback_group=ReentrantCallbackGroup(),
            )
            self._goal_msg = GripperCommand.Goal()

        self.joint_state_sub = self.robot_node.create_subscription(
            JointState,
            "joint_states",
            self._joint_state_callback,
            10,
        )

        # Create and start the executor in a separate thread
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.robot_node)
        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()
        time.sleep(3)  # Give some time to connect to services and receive messages

        if self.action_type == ActionType.JOINT_JOG and self.moveit2_servo is not None:
            if not self.moveit2_servo.enable(
                mode="joint_jog", wait_for_server_timeout_sec=10.0
            ):
                logger.error(
                    "MoveIt Servo joint_jog mode failed to start — "
                    "check servo_node is running and /servo_node/switch_command_type is available."
                )
            else:
                logger.info("MoveIt Servo joint_jog mode ready.")

        self.is_connected = True

    def set_leader_joint_steps(self, steps: dict[str, float]) -> None:
        """Per-frame leader joint deltas (rad) for open-loop joint jog (e.g. shoulder_pan)."""
        self._leader_joint_step = steps

    def send_joint_position_command(self, joint_positions: list[float], unnormalize: bool = True) -> None:
        """
        Send a command to the robot's joints.
        Args:
            joint_positions (list[float]): The target positions for the joints.
            unnormalize (bool): Whether to unnormalize the joint positions based on the robot's configuration.
        """
        if not self.robot_node:
            raise DeviceNotConnectedError("ROS2Interface is not connected. You need to call `connect()`.")

        if unnormalize:
            if self.config.min_joint_positions is None or self.config.max_joint_positions is None:
                raise ValueError(
                    "Joint position normalization requires min and max joint positions to be set."
                )
            joint_positions = _clip_joint_positions(
                joint_positions,
                self.config.min_joint_positions,
                self.config.max_joint_positions,
            )

        if len(joint_positions) != len(self.config.arm_joint_names):
            raise ValueError(
                f"Expected {len(self.config.arm_joint_names)} joint positions, but got {len(joint_positions)}."
            )

        if self.action_type == ActionType.JOINT_JOG:
            if self.moveit2_servo is None:
                raise DeviceNotConnectedError("MoveIt Servo is not initialized.")
            if self._last_joint_state is None:
                raise ValueError("Joint state is not available yet.")
            if (
                self.config.min_joint_positions is not None
                and self.config.max_joint_positions is not None
            ):
                joint_positions = _clip_joint_positions(
                    joint_positions,
                    self.config.min_joint_positions,
                    self.config.max_joint_positions,
                )
            n = len(self.config.arm_joint_names)
            if self._filtered_jog_vel is None or len(self._filtered_jog_vel) != n:
                self._filtered_jog_vel = [0.0] * n
            err_scale = self.config.joint_jog_unitless_error_rad
            per_joint_err_scale = self.config.joint_jog_joint_error_scale
            default_alpha = self.config.joint_jog_velocity_filter_alpha
            per_joint_alpha = self.config.joint_jog_joint_velocity_filter_alpha
            limit_eps_by_joint = {"shoulder_lift": 0.006}
            default_limit_eps = 0.003
            velocities = []
            for i, (joint, goal) in enumerate(
                zip(self.config.arm_joint_names, joint_positions, strict=True)
            ):
                present = self._last_joint_state["position"].get(joint, 0.0)
                err = goal - present
                if (
                    self.config.min_joint_positions is not None
                    and self.config.max_joint_positions is not None
                ):
                    lo = self.config.min_joint_positions[i]
                    hi = self.config.max_joint_positions[i]
                    limit_eps = limit_eps_by_joint.get(joint, default_limit_eps)
                    if present >= hi - limit_eps and err > 0.0:
                        err = 0.0
                    elif present <= lo + limit_eps and err < 0.0:
                        err = 0.0
                joint_err_scale = err_scale * per_joint_err_scale.get(joint, 1.0)
                if joint_err_scale > 0.0:
                    cmd = max(-1.0, min(1.0, err / joint_err_scale))
                else:
                    cmd = 0.0
                alpha = per_joint_alpha.get(joint, default_alpha)
                if alpha < 1.0:
                    cmd = alpha * cmd + (1.0 - alpha) * self._filtered_jog_vel[i]
                self._filtered_jog_vel[i] = cmd
                velocities.append(cmd)
            # Servo must already be in joint_jog mode (enabled at connect).
            self.moveit2_servo.joint_jog(
                self.config.arm_joint_names, velocities, enable_if_disabled=False
            )
        elif self.action_type == ActionType.JOINT_TRAJECTORY:
            if self.traj_cmd_pub is None:
                raise DeviceNotConnectedError("Trajectory command publisher is not initialized.")
            msg = JointTrajectory()
            msg.joint_names = self.config.arm_joint_names
            point = JointTrajectoryPoint()
            point.positions = joint_positions
            point.time_from_start = _TRAJ_TIME_250MS
            msg.points = [point]
            self.traj_cmd_pub.publish(msg)
        else:
            if self.pos_cmd_pub is None:
                raise DeviceNotConnectedError("Position command publisher is not initialized.")
            msg = Float64MultiArray()
            msg.data = joint_positions
            self.pos_cmd_pub.publish(msg)

    def servo(self, linear, angular, normalize: bool = True) -> None:
        if not self.moveit2_servo:
            raise DeviceNotConnectedError("ROS2Interface is not connected. You need to call `connect()`.")

        if normalize:
            linear = [v * self.config.max_linear_velocity for v in linear]
            angular = [v * self.config.max_angular_velocity for v in angular]
        self.moveit2_servo.servo(linear=linear, angular=angular)

    def send_gripper_command(self, position: float, unnormalize: bool = True) -> bool:
        """
        Send a command to the gripper to move to a specific position.
        Args:
            position (float): The target position for the gripper (0=open, 1=closed).
        Returns:
            bool: True if the command was sent successfully, False otherwise.
        """
        if not self.robot_node:
            raise DeviceNotConnectedError("ROS2Interface is not connected. You need to call `connect()`.")

        if unnormalize:
            # Map normalized position (0=open, 1=closed) to actual gripper joint position
            open_pos = self.config.gripper_open_position
            closed_pos = self.config.gripper_close_position
            p = max(0.0, min(1.0, float(position)))
            gripper_goal = open_pos + p * (closed_pos - open_pos)
        else:
            gripper_goal = float(position)

        if self.action_type == ActionType.JOINT_POSITION or (
            self.action_type == ActionType.JOINT_JOG and self.gripper_pos_cmd_pub is not None
        ):
            if self.gripper_pos_cmd_pub is None:
                raise DeviceNotConnectedError("Gripper command publisher is not initialized.")
            msg = Float64MultiArray()
            msg.data = [float(gripper_goal)]
            self.gripper_pos_cmd_pub.publish(msg)
            return True
        if self.config.gripper_action_type == GripperActionType.TRAJECTORY:
            if self.gripper_traj_pub is None:
                raise DeviceNotConnectedError("Gripper command publisher is not initialized.")
            msg = JointTrajectory()
            msg.joint_names = [self.config.gripper_joint_name]
            point = JointTrajectoryPoint()
            point.positions = [float(gripper_goal)]
            point.time_from_start = _GRIPPER_TRAJ_TIME
            msg.points = [point]
            self.gripper_traj_pub.publish(msg)
            return True
        else:
            if not self.gripper_action_client:
                raise DeviceNotConnectedError("Gripper action client is not initialized.")

            if not self.gripper_action_client.wait_for_server(timeout_sec=1.0):
                logger.error("Gripper action server not available")
                return False

            self._goal_msg.command.position = float(gripper_goal)
            if not (resp := self.gripper_action_client.send_goal(self._goal_msg)):
                logger.error("Failed to send gripper command")
                return False
            result = resp.result  # type: ignore  # ROS2 types available at runtime
            if result.reached_goal:
                return True
            logger.error(
                f"Gripper did not reach goal. stalled: {result.stalled}, "
                f"effort: {result.effort}, position: {result.position}"
            )
            return False

    @property
    def joint_state(self) -> dict[str, dict[str, float]] | None:
        """Get the last received joint state."""
        return self._last_joint_state

    def _joint_state_callback(self, msg: "JointState") -> None:
        self._last_joint_state = self._last_joint_state or {}
        positions = {}
        velocities = {}
        name_to_index = {name: i for i, name in enumerate(msg.name)}
        for joint_name in self.config.arm_joint_names:
            idx = name_to_index.get(joint_name)
            if idx is None:
                raise ValueError(f"Joint '{joint_name}' not found in joint state.")
            positions[joint_name] = msg.position[idx]
            velocities[joint_name] = msg.velocity[idx]

        if self.config.gripper_joint_name:
            idx = name_to_index.get(self.config.gripper_joint_name)
            if idx is None:
                raise ValueError(
                    f"Gripper joint '{self.config.gripper_joint_name}' not found in joint state."
                )
            positions[self.config.gripper_joint_name] = msg.position[idx]
            velocities[self.config.gripper_joint_name] = msg.velocity[idx]

        self._last_joint_state["position"] = positions
        self._last_joint_state["velocity"] = velocities

    def disconnect(self):
        if self.joint_state_sub:
            self.joint_state_sub.destroy()
            self.joint_state_sub = None
        if self.pos_cmd_pub:
            self.pos_cmd_pub.destroy()
            self.pos_cmd_pub = None
        if self.traj_cmd_pub:
            self.traj_cmd_pub.destroy()
            self.traj_cmd_pub = None
        if self.gripper_action_client:
            self.gripper_action_client.destroy()
            self.gripper_action_client = None
        if self.gripper_traj_pub:
            self.gripper_traj_pub.destroy()
            self.gripper_traj_pub = None
        if self.gripper_pos_cmd_pub:
            self.gripper_pos_cmd_pub.destroy()
            self.gripper_pos_cmd_pub = None
        if self.robot_node:
            self.robot_node.destroy_node()
            self.robot_node = None
        if self.moveit2_servo:
            self.moveit2_servo = None
        self._filtered_jog_vel = None
        self._leader_joint_step = {}

        if self.executor:
            self.executor.shutdown()
            self.executor = None
        if self.executor_thread:
            self.executor_thread.join()
            self.executor_thread = None

        self.is_connected = False
