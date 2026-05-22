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

from control_msgs.msg import JointJog
from geometry_msgs.msg import TwistStamped
from moveit_msgs.msg import ServoStatus
from moveit_msgs.srv import ServoCommandType
from rclpy import qos
from rclpy.callback_groups import CallbackGroup
from rclpy.node import Node
from std_srvs.srv import SetBool

logger = logging.getLogger(__name__)


class MoveIt2Servo:
    """MoveIt Servo client: cartesian twist (AR4-style) or joint jog (SO-101 leader teleop)."""

    def __init__(
        self,
        node: "Node",
        frame_id: str,
        callback_group: "CallbackGroup",
    ):
        self._node = node
        self._frame_id = frame_id
        self._enabled = False
        qos_profile = qos.QoSProfile(
            durability=qos.QoSDurabilityPolicy.VOLATILE,
            reliability=qos.QoSReliabilityPolicy.RELIABLE,
            history=qos.QoSHistoryPolicy.KEEP_ALL,
        )

        self._twist_pub = node.create_publisher(
            TwistStamped,
            "/servo_node/delta_twist_cmds",
            qos_profile,
            callback_group=callback_group,
        )
        self._joint_jog_pub = node.create_publisher(
            JointJog,
            "/servo_node/delta_joint_cmds",
            qos_profile,
            callback_group=callback_group,
        )
        self._pause_srv = node.create_client(
            SetBool, "/servo_node/pause_servo", callback_group=callback_group
        )
        self._cmd_type_srv = node.create_client(
            ServoCommandType, "/servo_node/switch_command_type", callback_group=callback_group
        )
        self._twist_msg = TwistStamped()
        self._joint_jog_msg = JointJog()
        self._unpause_req = SetBool.Request(data=False)
        self._pause_req = SetBool.Request(data=True)
        self._twist_type_req = ServoCommandType.Request(
            command_type=ServoCommandType.Request.TWIST
        )
        self._joint_jog_type_req = ServoCommandType.Request(
            command_type=ServoCommandType.Request.JOINT_JOG  # 0
        )
        self._status_sub = node.create_subscription(
            ServoStatus,
            "/servo_node/status",
            self._status_callback,
            10,
            callback_group=callback_group,
        )
        self._last_status_code: int | None = None

    def _status_callback(self, msg) -> None:
        if msg.code != self._last_status_code:
            self._last_status_code = msg.code
            if msg.code != 0:
                logger.warning("MoveIt Servo status: code=%s %s", msg.code, msg.message)

    def _switch_command_type(self, request: ServoCommandType.Request) -> bool:
        if not self._cmd_type_srv.wait_for_service(timeout_sec=2.0):
            logger.warning("Servo switch_command_type service not available.")
            return False
        result = self._cmd_type_srv.call(request)
        if not result or not result.success:
            logger.error("MoveIt Servo switch_command_type failed.")
            return False
        return True

    def enable(self, mode: str = "twist", wait_for_server_timeout_sec: float = 2.0) -> bool:
        if not self._pause_srv.wait_for_service(timeout_sec=wait_for_server_timeout_sec):
            logger.warning("Pause service not available.")
            return False
        result = self._pause_srv.call(self._unpause_req)
        if not result or not result.success:
            logger.error("Enable (unpause) failed: %s", getattr(result, "message", ""))
            self._enabled = False
            return False
        type_req = self._joint_jog_type_req if mode == "joint_jog" else self._twist_type_req
        if not self._switch_command_type(type_req):
            self._enabled = False
            return False
        logger.info("MoveIt Servo enabled (%s).", mode)
        self._enabled = True
        return True

    def disable(self, wait_for_server_timeout_sec: float = 1.0) -> bool:
        if not self._pause_srv.wait_for_service(timeout_sec=wait_for_server_timeout_sec):
            logger.warning("Pause service not available.")
            return False
        result = self._pause_srv.call(self._pause_req)
        self._enabled = not (result and result.success)
        return bool(result and result.success)

    def servo(self, linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0), enable_if_disabled=True):
        if not self._enabled and enable_if_disabled and not self.enable(mode="twist"):
            logger.warning("Dropping servo command because MoveIt2 Servo is not enabled.")
            return

        self._twist_msg.header.frame_id = self._frame_id
        self._twist_msg.header.stamp = self._node.get_clock().now().to_msg()
        self._twist_msg.twist.linear.x = float(linear[0])
        self._twist_msg.twist.linear.y = float(linear[1])
        self._twist_msg.twist.linear.z = float(linear[2])
        self._twist_msg.twist.angular.x = float(angular[0])
        self._twist_msg.twist.angular.y = float(angular[1])
        self._twist_msg.twist.angular.z = float(angular[2])
        self._twist_pub.publish(self._twist_msg)

    def joint_jog(
        self,
        joint_names: list[str],
        velocities: list[float],
        enable_if_disabled: bool = True,
    ) -> None:
        """Publish joint jog commands (unitless velocities in [-1, 1] for SO-101 teleop)."""
        if not self._enabled and enable_if_disabled and not self.enable(mode="joint_jog"):
            logger.warning("Dropping joint jog because MoveIt2 Servo is not enabled.")
            return

        self._joint_jog_msg.header.stamp = self._node.get_clock().now().to_msg()
        self._joint_jog_msg.header.frame_id = self._frame_id
        self._joint_jog_msg.joint_names = list(joint_names)
        self._joint_jog_msg.displacements = []
        self._joint_jog_msg.velocities = [float(v) for v in velocities]
        self._joint_jog_pub.publish(self._joint_jog_msg)
