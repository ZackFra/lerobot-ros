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
import math
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config import ActionType, ROS2Config, SO101ROSConfig
from .ros_interface import ROS2Interface

logger = logging.getLogger(__name__)


class ROS2Robot(Robot):
    config_class = ROS2Config
    name = "ros2"

    def __init__(self, config: ROS2Config):
        super().__init__(config)
        self.config = config
        self.ros2_interface = ROS2Interface(config.ros2_interface, config.action_type)
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        all_joint_names = self.config.ros2_interface.arm_joint_names.copy()
        if self.config.ros2_interface.gripper_joint_name:
            all_joint_names.append(self.config.ros2_interface.gripper_joint_name)
        motor_state_ft = {f"{motor}.pos": float for motor in all_joint_names}
        return {**motor_state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        if self.config.action_type == ActionType.CARTESIAN_VELOCITY:
            return {
                "linear_x.vel": float,
                "linear_y.vel": float,
                "linear_z.vel": float,
                "angular_x.vel": float,
                "angular_y.vel": float,
                "angular_z.vel": float,
                "gripper.pos": float,
            }
        elif self.config.action_type in (
            ActionType.JOINT_POSITION,
            ActionType.JOINT_TRAJECTORY,
            ActionType.JOINT_JOG,
        ):
            return {f"{joint}.pos": float for joint in self.config.ros2_interface.arm_joint_names} | {
                "gripper.pos": float
            }
        else:
            raise ValueError(f"Unsupported action type: {self.config.action_type}")

    @property
    def is_connected(self) -> bool:
        return self.ros2_interface.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        for cam in self.cameras.values():
            cam.connect()
        self.ros2_interface.connect()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass  # robot must be calibrated before running LeRobot

    def configure(self) -> None:
        pass  # robot must be configured before running LeRobot

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs_dict: dict[str, Any] = {}
        joint_state = self.ros2_interface.joint_state
        if joint_state is None:
            raise ValueError("Joint state is not available yet.")
        obs_dict.update({f"{joint}.pos": pos for joint, pos in joint_state["position"].items()})

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            try:
                obs_dict[cam_key] = cam.async_read(timeout_ms=300)
            except Exception as e:
                logger.error(f"Failed to read camera {cam_key}: {e}")
                obs_dict[cam_key] = None
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Args:
            action (dict[str, float]): The goal positions for the motors or pressed_keys dict.

        Raises:
            DeviceNotConnectedError: if robot is not connected.

        Returns:
            dict[str, float]: The action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.config.action_type == ActionType.CARTESIAN_VELOCITY:
            if self.config.max_relative_target is not None:
                # We don't have the current velocity of the arm, so set it to 0.0
                # Effectively the goal velocity gets clipped by max_relative_target
                goal_present_vel = {key: (act, 0.0) for key, act in action.items()}
                action = ensure_safe_goal_position(goal_present_vel, self.config.max_relative_target)

            linear_vel = (
                action["linear_x.vel"],
                action["linear_y.vel"],
                action["linear_z.vel"],
            )
            angular_vel = (
                action["angular_x.vel"],
                action["angular_y.vel"],
                action["angular_z.vel"],
            )
            self.ros2_interface.servo(linear=linear_vel, angular=angular_vel)
        elif self.config.action_type in (
            ActionType.JOINT_POSITION,
            ActionType.JOINT_TRAJECTORY,
            ActionType.JOINT_JOG,
        ):
            if self.config.max_relative_target is not None:
                goal_present_pos = {}
                joint_state = self.ros2_interface.joint_state
                if joint_state is None:
                    raise ValueError("Joint state is not available yet.")
                gripper_name = self.config.ros2_interface.gripper_joint_name

                for key, goal in action.items():
                    if key == "gripper.pos":
                        continue
                    joint_name = key.removesuffix(".pos")
                    present_pos = joint_state["position"].get(joint_name, 0.0)
                    goal_present_pos[key] = (goal, present_pos)
                action = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

            joint_positions = [action[joint + ".pos"] for joint in self.config.ros2_interface.arm_joint_names]
            self.ros2_interface.send_joint_position_command(joint_positions)

        gripper_pos = action["gripper.pos"]
        self.ros2_interface.send_gripper_command(gripper_pos)
        return action

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        for cam in self.cameras.values():
            cam.disconnect()
        self.ros2_interface.disconnect()

        logger.info(f"{self} disconnected.")


class SO101ROS(ROS2Robot):
    def __init__(self, config: ROS2Config):
        super().__init__(config)
        self._leader_home: dict[str, float] | None = None
        self._follower_home: dict[str, float] | None = None
        self._leader_prev_frame: dict[str, float] | None = None
        self._filtered_goals: dict[str, float] | None = None

    def connect(self, calibrate: bool = True) -> None:
        self._leader_home = None
        self._follower_home = None
        self._leader_prev_frame = None
        self._filtered_goals = None
        super().connect(calibrate=calibrate)

    def disconnect(self):
        self._leader_home = None
        self._follower_home = None
        self._leader_prev_frame = None
        self._filtered_goals = None
        super().disconnect()

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not isinstance(self.config, SO101ROSConfig):
            return super().send_action(action)

        action = dict(action)
        arm_joints = self.config.ros2_interface.arm_joint_names

        if self.config.convert_so101_leader_units:
            for joint in arm_joints:
                action[f"{joint}.pos"] = math.radians(action[f"{joint}.pos"])

        leader_now = {joint: action[f"{joint}.pos"] for joint in arm_joints}
        leader_steps: dict[str, float] = {}
        if self._leader_prev_frame is not None:
            for joint in arm_joints:
                leader_steps[joint] = leader_now[joint] - self._leader_prev_frame[joint]
        self._leader_prev_frame = leader_now
        if self.config.action_type == ActionType.JOINT_JOG:
            self.ros2_interface.set_leader_joint_steps(leader_steps)

        if self.config.mirror_leader_delta:
            joint_state = self.ros2_interface.joint_state
            if joint_state is None:
                raise ValueError("Joint state is not available yet.")
            if self._leader_home is None:
                self._leader_home = {j: action[f"{j}.pos"] for j in arm_joints}
                self._follower_home = {
                    j: joint_state["position"][j] for j in arm_joints
                }
                logger.info("SO-101 delta teleop: captured leader/follower home pose.")
                ri = self.config.ros2_interface
                if ri.min_joint_positions and ri.max_joint_positions:
                    margin = 0.12
                    for joint, lo, hi in zip(
                        arm_joints,
                        ri.min_joint_positions,
                        ri.max_joint_positions,
                        strict=True,
                    ):
                        pos = self._follower_home[joint]
                        if pos <= lo + margin or pos >= hi - margin:
                            logger.warning(
                                "Teleop home: follower %s at %.3f rad (limits [%.2f, %.2f]). "
                                "Restart teleop with arm near mid-range for full travel.",
                                joint,
                                pos,
                                lo,
                                hi,
                            )
            delta_scales = self.config.mirror_leader_joint_delta_scale
            for joint in arm_joints:
                delta = action[f"{joint}.pos"] - self._leader_home[joint]
                delta *= delta_scales.get(joint, 1.0)
                action[f"{joint}.pos"] = self._follower_home[joint] + delta

        ri = self.config.ros2_interface
        if ri.min_joint_positions is not None and ri.max_joint_positions is not None:
            for joint, lo, hi in zip(
                arm_joints, ri.min_joint_positions, ri.max_joint_positions, strict=True
            ):
                key = f"{joint}.pos"
                action[key] = max(lo, min(hi, action[key]))

        alpha = self.config.command_goal_filter_alpha
        if alpha is not None and 0.0 < alpha < 1.0:
            if self._filtered_goals is None:
                joint_state = self.ros2_interface.joint_state
                self._filtered_goals = {
                    joint: (
                        joint_state["position"][joint]
                        if joint_state is not None
                        else action[f"{joint}.pos"]
                    )
                    for joint in arm_joints
                }
            for joint in arm_joints:
                key = f"{joint}.pos"
                goal = action[key]
                self._filtered_goals[joint] = (
                    alpha * goal + (1.0 - alpha) * self._filtered_goals[joint]
                )
                action[key] = self._filtered_goals[joint]

        raw = float(action["gripper.pos"])
        t = self.config.gripper_leader_safe_close_raw
        o = self.config.gripper_leader_open_raw
        if t is not None and o > t:
            if raw <= t:
                p = 1.0
            elif raw >= o:
                p = 0.0
            else:
                p = (o - raw) / (o - t)
        else:
            g = max(0.0, min(1.0, raw / 100.0))
            p = 1.0 - g
        action["gripper.pos"] = max(0.0, min(1.0, p))
        return super().send_action(action)


class AnninAR4(ROS2Robot):
    pass
