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
from dataclasses import dataclass, field
from enum import Enum

from lerobot.cameras import CameraConfig
from lerobot.robots import RobotConfig


class ActionType(Enum):
    CARTESIAN_VELOCITY = "cartesian_velocity"
    JOINT_POSITION = "joint_position"
    JOINT_TRAJECTORY = "joint_trajectory"
    # Leader joint goals → MoveIt Servo joint jog (requires servo_node in bringup).
    JOINT_JOG = "joint_jog"


class GripperActionType(Enum):
    TRAJECTORY = "trajectory"  # Use JointTrajectoryController for gripper
    ACTION = "action"  # Use GripperActionClient


@dataclass
class ROS2InterfaceConfig:
    # Namespace used by ros2_control / MoveIt2 nodes
    namespace: str = ""

    arm_joint_names: list[str] = field(
        default_factory=lambda: [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ]
    )
    gripper_joint_name: str = "gripper_joint"

    # Base link name for computing end effector pose / velocity
    # Only applicable for cartesian control
    base_link: str = "base_link"

    # Only applicable if velocity control is used.
    max_linear_velocity: float = 0.10
    max_angular_velocity: float = 0.25  # rad/s

    # Only applicable if position control is used.
    min_joint_positions: list[float] | None = None
    max_joint_positions: list[float] | None = None

    gripper_open_position: float = 0.0
    gripper_close_position: float = 1.0

    gripper_action_type: GripperActionType = GripperActionType.TRAJECTORY

    # Used with forward_command_controller (hardware); see so101_controllers_hw.yaml.
    arm_commands_topic: str = "/arm_controller/commands"
    gripper_commands_topic: str = "/gripper_controller/commands"
    # When true (SO-101 HW), gripper uses Float64MultiArray on gripper_commands_topic.
    gripper_use_forward_commands: bool = False

    # Joint jog (MoveIt Servo unitless): |goal − present| >= this rad → |cmd| = 1.0.
    # Smaller → "snappier" tracking (servo runs at full speed for small errors).
    joint_jog_unitless_error_rad: float = 0.02
    # Per-joint multiplier on error scale (>1 = softer if a joint hunts).
    joint_jog_joint_error_scale: dict[str, float] = field(default_factory=dict)
    # EMA on outgoing unitless cmd (1.0 = off; Servo accel smoothing handles smoothness).
    joint_jog_velocity_filter_alpha: float = 1.0
    joint_jog_joint_velocity_filter_alpha: dict[str, float] = field(default_factory=dict)


@dataclass
class ROS2Config(RobotConfig):
    # Action type for controlling the robot. Can be 'cartesian_velocity' or 'joint_position'.
    action_type: ActionType = ActionType.JOINT_POSITION

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a list that is the same length as
    # the number of motors in your follower arms.
    max_relative_target: int | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # ROS2 interface configuration
    ros2_interface: ROS2InterfaceConfig = field(default_factory=ROS2InterfaceConfig)


@RobotConfig.register_subclass("annin_ar4_mk1")
@dataclass
class AnninAR4Config(ROS2Config):
    """Annin Robotics AR4 robot configuration - extends ROS2Config with
    AR4-specific settings
    """

    action_type: ActionType = ActionType.CARTESIAN_VELOCITY

    ros2_interface: ROS2InterfaceConfig = field(
        default_factory=lambda: ROS2InterfaceConfig(
            gripper_joint_name="gripper_jaw1_joint",
            base_link="base_link",
            min_joint_positions=[-2.9671, -0.7330, -1.5533, -2.8798, -1.8326, -2.7053],
            max_joint_positions=[2.9671, 1.5708, 0.9076, 2.8798, 1.8326, 2.7053],
            gripper_open_position=0.014,
            gripper_close_position=0.0,
            gripper_action_type=GripperActionType.ACTION,
        ),
    )


@RobotConfig.register_subclass("so101_ros")
@dataclass
class SO101ROSConfig(ROS2Config):
    """Configuration for the ROS 2 SO-101: URDF joints match Hugging Face LeRobot (`so101_leader` / `so101_follower`).

    **Gripper “full close” tuning (no guesswork):**
    1. Drive the gripper in sim until the tips look right (touching, no mesh overlap).
    2. Read the actual angle:
       ``ros2 topic echo /joint_states --once``
       and find ``gripper_joint`` in ``name`` / ``position`` (radians).
    3. Set ``ros2_interface.gripper_close_position`` to that value (URDF ``lower`` / ros2_control ``min``
       can be slightly more negative for margin).

    **HW teleop (smooth + collision):** ``--robot.action_type=joint_jog`` → MoveIt Servo
    (``/servo_node/delta_joint_cmds`` → ``/arm_controller/commands``). Use ``command_goal_filter_alpha``
    and ``max_relative_target`` to soften motion; keep ``servo_node`` from bringup. For direct
    ros2_control without collision checking, use ``joint_position`` instead.

    **Leader NORM vs joint_states:** teleop ``gripper.pos`` is LeRobot’s 0–100-style bus normalization,
    not degrees of ``gripper_joint``. Use ``gripper_leader_safe_close_raw`` to snap “this tight or tighter”
    to ``gripper_close_position``.
    """

    action_type: ActionType = ActionType.JOINT_TRAJECTORY

    # Optional EMA on arm goals (rad) before Servo. None = off (recommended; add only if needed).
    command_goal_filter_alpha: float | None = None
    # Max rad per arm joint per teleop frame. None = off. Never applied to gripper.pos.
    max_relative_target: float | None = None

    # With `teleop.type=so101_leader` and default `teleop.use_degrees=true`, arm `.pos` values are **degrees**
    # (see `SOLeaderConfig` / `SOLeader` in lerobot). Set this true so ROS gets radians. If you set
    # `teleop.use_degrees=false`, arm joints use normalized −100..100 instead — do not use this flag; you
    # would need a different mapping to joint angles.
    convert_so101_leader_units: bool = True
    # When true, follower tracks RELATIVE leader motion from teleop start ("deltas").
    # When false (default), follower SNAPS to leader's absolute calibrated joint angles.
    mirror_leader_delta: bool = False
    # Scale leader delta per joint (e.g. shoulder_lift range vs follower URDF).
    mirror_leader_joint_delta_scale: dict[str, float] = field(
        default_factory=lambda: {"shoulder_lift": 1.15}
    )
    # Leader gripper reading (0–100 style) at the safe visual close; raw <= this → commanded close = 1.0.
    gripper_leader_safe_close_raw: float | None = 6.39
    # Leader reading treated as “fully open” for linear interpolation (open → p=0).
    gripper_leader_open_raw: float = 100.0

    # Feetech ``SOFollower.configure()`` runs in ``so101_controller_hw.launch.py`` before ros2_control
    # opens the USB port (not here — teleop connects after the driver holds the bus).

    ros2_interface: ROS2InterfaceConfig = field(
        default_factory=lambda: ROS2InterfaceConfig(
            arm_joint_names=[
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
            ],
            gripper_joint_name="gripper_joint",
            base_link="base",
            min_joint_positions=[-1.91986, -2.35, -1.74533, -1.74533, -2.79253],
            max_joint_positions=[1.91986, 1.74533, 1.85, 1.74533, 2.79253],
            gripper_open_position=1.74533,
            gripper_close_position=-0.191776,
            gripper_use_forward_commands=True,
            joint_jog_unitless_error_rad=0.035,
            joint_jog_velocity_filter_alpha=1.0,
            joint_jog_joint_error_scale={
                "shoulder_pan": 1.6,
                "shoulder_lift": 1.5,
                "elbow_flex": 1.3,
                "wrist_flex": 2.0,
                "wrist_roll": 2.0,
            },
        ),
    )
