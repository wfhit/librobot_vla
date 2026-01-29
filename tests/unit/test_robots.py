"""Test robot definitions."""

import numpy as np
import pytest

from librobot.robots import SO100ArmRobot, WheelLoaderRobot


def test_wheel_loader_robot():
    """Test wheel loader robot definition."""
    robot = WheelLoaderRobot()

    assert robot.name == "wheel_loader"
    assert robot.action_dim == 6
    assert robot.state_dim == 22
    assert len(robot.action_names) == 6
    assert len(robot.state_names) == 22


def test_so100_arm_robot():
    """Test SO100 arm robot definition."""
    robot = SO100ArmRobot()

    assert robot.name == "so100_arm"
    assert robot.action_dim == 6
    assert robot.state_dim == 12
    assert len(robot.action_names) == 6
    assert len(robot.state_names) == 12


def test_robot_normalize_denormalize_action():
    """Test action normalization/denormalization."""
    robot = WheelLoaderRobot()

    # Test action at min range
    action = np.array([-1.0, -1.0, -1.0, -1.0, 0.0, -1.0])
    normalized = robot.normalize_action(action)
    assert np.allclose(normalized, [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])

    # Test denormalization
    denormalized = robot.denormalize_action(normalized)
    assert np.allclose(denormalized, action, atol=1e-6)

    # Test action at max range
    action = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    normalized = robot.normalize_action(action)
    assert np.allclose(normalized, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # Test denormalization
    denormalized = robot.denormalize_action(normalized)
    assert np.allclose(denormalized, action, atol=1e-6)
