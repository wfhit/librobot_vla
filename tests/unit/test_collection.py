"""
Unit tests for the data collection module.

Tests teleoperation interfaces, recording utilities, converters, and DataCollector.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from librobot.collection import (
    CameraRecorder,
    DataBuffer,
    DataCollector,
    EpisodeRecorder,
    KeyboardTeleop,
    LeaderFollowerTeleop,
    LeRobotConverter,
    MocapTeleop,
    SpaceMouseTeleop,
    VRTeleop,
    create_teleop,
    list_teleoperation,
)


class TestTeleoperation:
    """Test suite for teleoperation interfaces."""

    def test_keyboard_teleop_initialization(self):
        """Test keyboard teleoperation initialization."""
        teleop = KeyboardTeleop(action_dim=7)
        assert teleop.action_dim == 7
        assert not teleop.is_connected

    def test_keyboard_teleop_connect(self):
        """Test keyboard teleoperation connection."""
        teleop = KeyboardTeleop()
        result = teleop.connect()
        assert result is True
        assert teleop.is_connected

    def test_keyboard_teleop_get_action(self):
        """Test getting action from keyboard."""
        teleop = KeyboardTeleop(action_dim=7)
        teleop.connect()
        action = teleop.get_action()
        assert isinstance(action, np.ndarray)
        assert action.shape == (7,)

    def test_keyboard_teleop_update_action(self):
        """Test updating action from key press."""
        teleop = KeyboardTeleop(action_dim=7)
        teleop.connect()

        # Press 'w' key (forward)
        teleop.update_action_from_key("w", pressed=True)
        action = teleop.get_action()
        assert action[0] != 0

        # Release 'w' key
        teleop.update_action_from_key("w", pressed=False)
        action = teleop.get_action()
        assert action[0] == 0

    def test_spacemouse_teleop_initialization(self):
        """Test SpaceMouse teleoperation initialization."""
        teleop = SpaceMouseTeleop(action_dim=7)
        assert teleop.action_dim == 7
        assert not teleop.is_connected

    def test_spacemouse_teleop_get_action(self):
        """Test getting action from SpaceMouse."""
        teleop = SpaceMouseTeleop(action_dim=7)
        action = teleop.get_action()
        assert isinstance(action, np.ndarray)
        assert action.shape == (7,)

    def test_vr_teleop_initialization(self):
        """Test VR teleoperation initialization."""
        teleop = VRTeleop(action_dim=7)
        assert teleop.action_dim == 7
        assert not teleop.is_connected

    def test_mocap_teleop_initialization(self):
        """Test motion capture teleoperation initialization."""
        teleop = MocapTeleop(action_dim=7)
        assert teleop.action_dim == 7
        assert not teleop.is_connected

    def test_mocap_teleop_calibrate(self):
        """Test motion capture calibration."""
        teleop = MocapTeleop(action_dim=7)
        teleop.connect()
        result = teleop.calibrate()
        assert result is True

    def test_leader_follower_teleop_initialization(self):
        """Test leader-follower teleoperation initialization."""
        teleop = LeaderFollowerTeleop(action_dim=7)
        assert teleop.action_dim == 7
        assert not teleop.is_connected

    def test_leader_follower_teleop_with_robot(self):
        """Test leader-follower with mock robot."""
        mock_robot = Mock()
        mock_robot.is_connected = True
        mock_robot.get_state.return_value = {"joint_positions": np.zeros(7)}

        teleop = LeaderFollowerTeleop(action_dim=7, leader_robot=mock_robot)
        teleop.connect()

        action = teleop.get_action()
        assert isinstance(action, np.ndarray)
        assert action.shape == (7,)

    def test_teleop_registry(self):
        """Test teleoperation registry."""
        teleops = list_teleoperation()
        assert "KeyboardTeleop" in teleops or "keyboard" in teleops

    def test_create_teleop(self):
        """Test creating teleoperation from registry."""
        teleop = create_teleop("keyboard", action_dim=7)
        assert isinstance(teleop, KeyboardTeleop)
        assert teleop.action_dim == 7

    def test_teleop_context_manager(self):
        """Test teleoperation as context manager."""
        with KeyboardTeleop() as teleop:
            assert teleop.is_connected

        assert not teleop.is_connected


class TestDataBuffer:
    """Test suite for DataBuffer."""

    def test_buffer_initialization(self):
        """Test buffer initialization."""
        buffer = DataBuffer(max_size=100)
        assert buffer.max_size == 100
        assert len(buffer) == 0

    def test_create_stream(self):
        """Test creating data stream."""
        buffer = DataBuffer()
        buffer.create_stream("images", dtype=np.uint8)
        assert "images" in buffer

    def test_append_data(self):
        """Test appending data to stream."""
        buffer = DataBuffer()
        buffer.append("actions", np.array([1, 2, 3]))
        assert buffer.get_stream_length("actions") == 1

    def test_get_stream(self):
        """Test getting stream data."""
        buffer = DataBuffer()
        buffer.append("actions", np.array([1, 2, 3]))
        buffer.append("actions", np.array([4, 5, 6]))
        data = buffer.get_stream("actions")
        assert len(data) == 2

    def test_get_last(self):
        """Test getting last n items."""
        buffer = DataBuffer()
        for i in range(10):
            buffer.append("values", i)

        last_3 = buffer.get_last("values", 3)
        assert len(last_3) == 3
        assert last_3 == [7, 8, 9]

    def test_clear_stream(self):
        """Test clearing a stream."""
        buffer = DataBuffer()
        buffer.append("actions", np.array([1, 2, 3]))
        buffer.clear_stream("actions")
        assert buffer.get_stream_length("actions") == 0

    def test_clear_all(self):
        """Test clearing all streams."""
        buffer = DataBuffer()
        buffer.append("actions", np.array([1, 2, 3]))
        buffer.append("images", np.zeros((10, 10)))
        buffer.clear_all()
        assert buffer.get_stream_length("actions") == 0
        assert buffer.get_stream_length("images") == 0

    def test_list_streams(self):
        """Test listing all streams."""
        buffer = DataBuffer()
        buffer.create_stream("stream1")
        buffer.create_stream("stream2")
        streams = buffer.list_streams()
        assert "stream1" in streams
        assert "stream2" in streams


class TestCameraRecorder:
    """Test suite for CameraRecorder."""

    def test_camera_recorder_initialization(self):
        """Test camera recorder initialization."""
        recorder = CameraRecorder(camera_names=["cam1", "cam2"])
        assert recorder.camera_names == ["cam1", "cam2"]
        assert recorder.resolution == (640, 480)
        assert recorder.fps == 30

    def test_setup_cameras(self):
        """Test camera setup."""
        recorder = CameraRecorder(camera_names=["cam1"])
        result = recorder.setup_cameras()
        assert result is True

    def test_start_stop_recording(self):
        """Test starting and stopping recording."""
        recorder = CameraRecorder(camera_names=["cam1"])
        recorder.start_recording()
        assert recorder._is_recording

        recorder.stop_recording()
        assert not recorder._is_recording

    def test_capture_frame(self):
        """Test capturing frames."""
        recorder = CameraRecorder(camera_names=["cam1", "cam2"])
        recorder.start_recording()
        frames = recorder.capture_frame()
        assert isinstance(frames, dict)
        assert "cam1" in frames
        assert "cam2" in frames

    def test_camera_context_manager(self):
        """Test camera recorder as context manager."""
        with CameraRecorder(camera_names=["cam1"]) as recorder:
            assert len(recorder._cameras) > 0


class TestEpisodeRecorder:
    """Test suite for EpisodeRecorder."""

    def test_episode_recorder_initialization(self):
        """Test episode recorder initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = EpisodeRecorder(output_dir=tmpdir, format="lerobot")
            assert recorder.output_dir == Path(tmpdir)
            assert recorder.format == "lerobot"

    def test_start_episode(self):
        """Test starting episode recording."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = EpisodeRecorder(output_dir=tmpdir)
            result = recorder.start_episode(task_name="test_task")
            assert result is True
            assert recorder.is_recording

    def test_stop_episode(self):
        """Test stopping episode recording."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = EpisodeRecorder(output_dir=tmpdir)
            recorder.start_episode(task_name="test_task")
            episode_data = recorder.stop_episode()
            assert isinstance(episode_data, dict)
            assert "metadata" in episode_data
            assert not recorder.is_recording

    def test_add_timestep(self):
        """Test adding timestep data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = EpisodeRecorder(output_dir=tmpdir)
            recorder.start_episode()

            recorder.add_timestep(
                action=np.array([1, 2, 3]),
                proprioception=np.array([4, 5, 6]),
            )

            assert recorder._buffer.get_stream_length("action") == 1

    def test_save_episode(self):
        """Test saving episode to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = EpisodeRecorder(output_dir=tmpdir, format="lerobot")
            recorder.start_episode(task_name="test_task")

            recorder.add_timestep(
                action=np.array([1, 2, 3]),
                proprioception=np.array([4, 5, 6]),
            )

            saved_path = recorder.save_episode()
            assert saved_path is not None
            assert saved_path.exists()


class TestConverters:
    """Test suite for data format converters."""

    def test_lerobot_converter_initialization(self):
        """Test LeRobot converter initialization."""
        converter = LeRobotConverter()
        assert converter.format_name == "lerobot"

    def test_lerobot_converter_write_read(self):
        """Test writing and reading episodes with LeRobot converter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = LeRobotConverter()

            # Create test episode data
            episode_data = {
                "metadata": {"episode_idx": 0, "task_name": "test"},
                "action": np.array([[1, 2, 3], [4, 5, 6]]),
                "timestamp": np.array([0.0, 0.1]),
            }

            # Write episode
            converter.write_episode(tmpdir, episode_data)

            # Read episode back
            read_data = converter.read_episode(tmpdir, episode_idx=0)

            assert "metadata" in read_data
            assert "action" in read_data
            assert np.allclose(read_data["action"], episode_data["action"])

    def test_lerobot_converter_validate(self):
        """Test LeRobot converter validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = LeRobotConverter()

            # Empty directory should be invalid
            assert not converter.validate_dataset(tmpdir)

            # Write an episode
            episode_data = {
                "metadata": {"episode_idx": 0},
                "action": np.array([[1, 2, 3]]),
            }
            converter.write_episode(tmpdir, episode_data)

            # Now should be valid
            assert converter.validate_dataset(tmpdir)

    def test_lerobot_converter_metadata(self):
        """Test LeRobot converter metadata operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            converter = LeRobotConverter()

            # Set metadata
            metadata = {"dataset_name": "test", "num_episodes": 10}
            converter.set_metadata(tmpdir, metadata)

            # Get metadata
            read_metadata = converter.get_metadata(tmpdir)
            assert read_metadata["dataset_name"] == "test"


class TestDataCollector:
    """Test suite for DataCollector."""

    def test_data_collector_initialization(self):
        """Test data collector initialization."""
        mock_robot = Mock()
        mock_teleop = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DataCollector(
                robot=mock_robot,
                teleop=mock_teleop,
                cameras=["cam1"],
                fps=30,
                output_dir=tmpdir,
            )

            assert collector.robot == mock_robot
            assert collector.fps == 30
            assert collector.cameras == ["cam1"]

    def test_data_collector_with_teleop_name(self):
        """Test data collector with teleoperation name."""
        mock_robot = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DataCollector(
                robot=mock_robot,
                teleop="keyboard",
                fps=30,
                output_dir=tmpdir,
            )

            assert isinstance(collector.teleop, KeyboardTeleop)

    def test_data_collector_get_status(self):
        """Test getting data collector status."""
        mock_robot = Mock()
        mock_teleop = Mock()

        with tempfile.TemporaryDirectory() as tmpdir:
            collector = DataCollector(
                robot=mock_robot,
                teleop=mock_teleop,
                output_dir=tmpdir,
            )

            status = collector.get_status()
            assert isinstance(status, dict)
            assert "robot" in status
            assert "teleop" in status
            assert "fps" in status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
