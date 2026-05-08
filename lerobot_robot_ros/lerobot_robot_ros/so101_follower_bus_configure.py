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

"""Feetech register setup matching ``lerobot.robots.so_follower.SOFollower.configure``.

Use when the SO-101 follower is driven by ROS 2 / ros2_control instead of the native
``so101_follower`` driver: the stack does not call ``SOFollower.configure``, so PID and
gripper protection registers stay at factory defaults.

**Serial port:** only one client can open the bus. If ``ros2_control`` already holds
``/dev/ttyUSB*``, stop the robot launch (or skip on-connect configure) and run::

    python -m lerobot_robot_ros.so101_follower_bus_configure /dev/ttyUSB0

once before starting the stack, or rely on your hardware interface to set registers.
"""

from __future__ import annotations

import argparse
import logging

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

logger = logging.getLogger(__name__)


def so101_follower_motors_map() -> dict[str, Motor]:
    """Same layout as ``lerobot.robots.so_follower.SOFollower`` (IDs 1-6, sts3215)."""
    norm_body = MotorNormMode.DEGREES
    return {
        "shoulder_pan": Motor(1, "sts3215", norm_body),
        "shoulder_lift": Motor(2, "sts3215", norm_body),
        "elbow_flex": Motor(3, "sts3215", norm_body),
        "wrist_flex": Motor(4, "sts3215", norm_body),
        "wrist_roll": Motor(5, "sts3215", norm_body),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    }


def make_so101_follower_bus(
    port: str,
    calibration: dict[str, MotorCalibration] | None = None,
) -> FeetechMotorsBus:
    return FeetechMotorsBus(port=port, motors=so101_follower_motors_map(), calibration=calibration)


def apply_sofollower_configure(bus: FeetechMotorsBus) -> None:
    """Mirror ``SOFollower.configure()`` (torque off → ``configure_motors`` → mode/PID/gripper limits)."""
    with bus.torque_disabled():
        bus.configure_motors()
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            bus.write("P_Coefficient", motor, 16)
            bus.write("I_Coefficient", motor, 0)
            bus.write("D_Coefficient", motor, 32)
            if motor == "gripper":
                bus.write("Max_Torque_Limit", motor, 500)
                bus.write("Protection_Current", motor, 250)
                bus.write("Overload_Torque", motor, 25)


def configure_so101_follower_serial(
    port: str,
    *,
    calibration: dict[str, MotorCalibration] | None = None,
) -> None:
    """Open ``port``, apply :func:`apply_sofollower_configure`, then close without disabling torque again."""
    bus = make_so101_follower_bus(port, calibration)
    bus.connect()
    try:
        apply_sofollower_configure(bus)
        logger.info("Applied SOFollower.configure-equivalent settings on %s", port)
    finally:
        bus.disconnect(disable_torque=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Apply SOFollower.configure() Feetech settings (PID, gripper torque limits)."
    )
    parser.add_argument("port", help="Serial device, e.g. /dev/ttyUSB0")
    args = parser.parse_args()
    configure_so101_follower_serial(args.port)
    print(f"Configured Feetech motors on {args.port}")


if __name__ == "__main__":
    main()
