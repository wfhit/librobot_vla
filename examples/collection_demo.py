"""Example demonstrating the data collection module."""

import numpy as np

from librobot.collection import (
    CameraRecorder,
    DataCollector,
    EpisodeRecorder,
    KeyboardTeleop,
    SpaceMouseTeleop,
    create_teleop,
    list_teleoperation,
)


def example_1_basic_teleoperation():
    """Example 1: Basic teleoperation interface usage."""
    print("Example 1: Basic Teleoperation Interface")
    print("-" * 60)

    # Create keyboard teleoperation
    teleop = KeyboardTeleop(action_dim=7, speed_scale=0.1)

    # Connect the device
    teleop.connect()
    print(f"Teleoperation connected: {teleop.is_connected}")

    # Get action
    action = teleop.get_action()
    print(f"Initial action: {action}")

    # Simulate key press
    teleop.update_action_from_key("w", pressed=True)
    action = teleop.get_action()
    print(f"Action after pressing 'w': {action}")

    # Get status
    status = teleop.get_status()
    print(f"Status: {status}")

    # Disconnect
    teleop.disconnect()
    print()


def example_2_using_registry():
    """Example 2: Using the teleoperation registry."""
    print("Example 2: Using Teleoperation Registry")
    print("-" * 60)

    # List available teleoperation interfaces
    available = list_teleoperation()
    print(f"Available teleoperation interfaces: {available}")

    # Create teleoperation by name
    teleop = create_teleop("keyboard", action_dim=7)
    print(f"Created: {type(teleop).__name__}")
    print()


def example_3_episode_recording():
    """Example 3: Recording episodes."""
    print("Example 3: Episode Recording")
    print("-" * 60)

    # Create episode recorder
    recorder = EpisodeRecorder(output_dir="./example_data", format="lerobot")

    # Start episode
    recorder.start_episode(task_name="pick_and_place", instruction="Pick up the block")
    print(f"Recording: {recorder.is_recording}")

    # Add some timesteps
    for i in range(5):
        recorder.add_timestep(
            action=np.random.randn(7),
            proprioception=np.random.randn(14),
            images={"camera1": np.zeros((480, 640, 3), dtype=np.uint8)},
        )

    # Stop and save episode
    episode_data = recorder.stop_episode()
    saved_path = recorder.save_episode(episode_data)
    print(f"Episode saved to: {saved_path}")
    print()


def example_4_camera_recording():
    """Example 4: Camera recording."""
    print("Example 4: Camera Recording")
    print("-" * 60)

    # Create camera recorder with multiple cameras
    camera_recorder = CameraRecorder(
        camera_names=["wrist", "front"],
        resolution=(640, 480),
        fps=30,
    )

    # Setup cameras
    camera_recorder.setup_cameras()
    print(f"Cameras initialized: {camera_recorder._cameras.keys()}")

    # Start recording
    camera_recorder.start_recording()

    # Capture frames
    frames = camera_recorder.capture_frame()
    print(f"Captured frames from: {frames.keys()}")

    # Stop recording
    camera_recorder.stop_recording()

    # Release cameras
    camera_recorder.release_cameras()
    print()


def example_5_data_collector():
    """Example 5: Complete data collection workflow."""
    print("Example 5: Complete Data Collection with DataCollector")
    print("-" * 60)

    # Mock robot for demonstration
    class MockRobot:
        def __init__(self):
            self.is_connected = False

        def connect(self):
            self.is_connected = True
            print("Robot connected")

        def disconnect(self):
            self.is_connected = False

        def execute_action(self, action):
            pass  # Execute action on robot

        def get_state(self):
            return {"joint_positions": np.random.randn(7)}

    # Create mock robot
    robot = MockRobot()

    # Create teleoperation interface
    teleop = KeyboardTeleop(action_dim=7)

    # Create data collector
    collector = DataCollector(
        robot=robot,
        teleop=teleop,
        cameras=["wrist", "front"],
        fps=30,
        output_dir="./collected_data",
    )

    # Get collector status
    status = collector.get_status()
    print(f"Collector status: {status}")

    # In a real scenario, you would call:
    # episodes_collected = collector.collect(
    #     num_episodes=10,
    #     task_name="pick_and_place",
    #     instructions=["Pick up the red block", "Place it in the bin"],
    # )

    print("Note: collector.collect() requires user interaction")
    print()


def example_6_context_managers():
    """Example 6: Using context managers."""
    print("Example 6: Context Managers")
    print("-" * 60)

    # Teleoperation with context manager
    with KeyboardTeleop(action_dim=7) as teleop:
        print(f"Connected inside context: {teleop.is_connected}")
        action = teleop.get_action()
        print(f"Action: {action}")

    print("Automatically disconnected after context")

    # Camera recorder with context manager
    with CameraRecorder(camera_names=["cam1"]) as camera_rec:
        print(f"Cameras setup: {len(camera_rec._cameras)}")

    print("Cameras automatically released")
    print()


def example_7_data_converters():
    """Example 7: Using data format converters."""
    print("Example 7: Data Format Converters")
    print("-" * 60)

    from librobot.collection import LeRobotConverter

    # Create converter
    converter = LeRobotConverter()

    # Create sample episode data
    episode_data = {
        "metadata": {"episode_idx": 0, "task_name": "demo"},
        "action": np.random.randn(10, 7),
        "proprioception": np.random.randn(10, 14),
    }

    # Write episode
    output_path = "./example_data"
    converter.write_episode(output_path, episode_data)
    print(f"Episode written to: {output_path}")

    # Read episode back
    read_data = converter.read_episode(output_path, episode_idx=0)
    print(f"Episode read successfully: {list(read_data.keys())}")

    # Validate dataset
    is_valid = converter.validate_dataset(output_path)
    print(f"Dataset valid: {is_valid}")

    # Get metadata
    metadata = converter.get_metadata(output_path)
    print(f"Metadata: {metadata}")
    print()


def main():
    """Run all examples."""
    print("=" * 60)
    print("Data Collection Module Examples")
    print("=" * 60)
    print()

    example_1_basic_teleoperation()
    example_2_using_registry()
    example_3_episode_recording()
    example_4_camera_recording()
    example_5_data_collector()
    example_6_context_managers()
    example_7_data_converters()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
