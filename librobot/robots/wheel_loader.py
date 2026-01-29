"""Wheel loader robot definition."""

import numpy as np

from librobot.robots.base import BaseRobot, RobotConfig
from librobot.utils.registry import register_robot


@register_robot("wheel_loader", aliases=["loader"])
class WheelLoaderRobot(BaseRobot):
    """Wheel loader robot definition.

    Action space (6D):
    - throttle: [-1, 1] (negative = brake)
    - steering: [-1, 1] (left/right)
    - boom: [-1, 1] (up/down)
    - bucket: [-1, 1] (tilt)
    - brake: [0, 1] (on/off)
    - gear: [-1, 1] (forward/reverse)

    State space (22D):
    - position: [x, y, z] (3D)
    - orientation: [roll, pitch, yaw] (3D)
    - linear_velocity: [vx, vy, vz] (3D)
    - angular_velocity: [wx, wy, wz] (3D)
    - throttle: scalar
    - steering: scalar
    - boom_angle: scalar
    - bucket_angle: scalar
    - brake: scalar
    - gear: scalar
    - engine_rpm: scalar
    - wheel_speed: [fl, fr, rl, rr] (4D)
    """

    def __init__(self):
        """Initialize wheel loader robot."""
        config = RobotConfig(
            name="wheel_loader",
            action_dim=6,
            state_dim=22,
            action_names=[
                "throttle",
                "steering",
                "boom",
                "bucket",
                "brake",
                "gear",
            ],
            state_names=[
                "pos_x", "pos_y", "pos_z",
                "roll", "pitch", "yaw",
                "vel_x", "vel_y", "vel_z",
                "ang_vel_x", "ang_vel_y", "ang_vel_z",
                "throttle",
                "steering",
                "boom_angle",
                "bucket_angle",
                "brake",
                "gear",
                "engine_rpm",
                "wheel_speed_fl", "wheel_speed_fr", "wheel_speed_rl", "wheel_speed_rr",
            ],
            action_ranges=np.array([
                [-1.0, 1.0],  # throttle
                [-1.0, 1.0],  # steering
                [-1.0, 1.0],  # boom
                [-1.0, 1.0],  # bucket
                [0.0, 1.0],   # brake
                [-1.0, 1.0],  # gear
            ]),
            control_frequency=10.0,
            description="Wheel loader with 6D action space and 22D state space",
        )
        super().__init__(config)
