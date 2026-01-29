"""SO100 arm robot definition."""

import numpy as np

from librobot.robots.base import BaseRobot, RobotConfig
from librobot.utils.registry import register_robot


@register_robot("so100_arm", aliases=["so100"])
class SO100ArmRobot(BaseRobot):
    """SO100 robotic arm definition.

    Action space (6D):
    - joint_1 through joint_6: Joint angles/velocities

    State space (12D):
    - joint_positions: [j1, j2, j3, j4, j5, j6] (6D)
    - joint_velocities: [v1, v2, v3, v4, v5, v6] (6D)
    """

    def __init__(self):
        """Initialize SO100 arm robot."""
        config = RobotConfig(
            name="so100_arm",
            action_dim=6,
            state_dim=12,
            action_names=[
                f"joint_{i}" for i in range(1, 7)
            ],
            state_names=[
                *[f"joint_pos_{i}" for i in range(1, 7)],
                *[f"joint_vel_{i}" for i in range(1, 7)],
            ],
            action_ranges=np.array([
                [-np.pi, np.pi],  # joint 1
                [-np.pi, np.pi],  # joint 2
                [-np.pi, np.pi],  # joint 3
                [-np.pi, np.pi],  # joint 4
                [-np.pi, np.pi],  # joint 5
                [-np.pi, np.pi],  # joint 6
            ]),
            control_frequency=20.0,
            description="SO100 6-DOF robotic arm",
        )
        super().__init__(config)
