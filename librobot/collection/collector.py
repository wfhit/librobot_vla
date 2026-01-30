"""Main data collector orchestrator."""

import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from librobot.collection.recording import CameraRecorder, EpisodeRecorder
from librobot.collection.teleoperation import create_teleop


class DataCollector:
    """
    Main data collector orchestrator.

    Coordinates robot control, teleoperation, camera recording, and
    episode saving for robot demonstration data collection.
    """

    def __init__(
        self,
        robot,
        teleop,
        recorder: Optional[EpisodeRecorder] = None,
        cameras: Optional[list[str]] = None,
        fps: int = 30,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize data collector.

        Args:
            robot: Robot instance
            teleop: Teleoperation interface instance or name
            recorder: Optional episode recorder (created if not provided)
            cameras: List of camera names
            fps: Recording frame rate
            output_dir: Output directory for data
        """
        self.robot = robot
        self.fps = fps
        self.frame_time = 1.0 / fps

        # Setup teleoperation
        if isinstance(teleop, str):
            self.teleop = create_teleop(teleop)
        else:
            self.teleop = teleop

        # Setup recorder
        if recorder is None:
            if output_dir is None:
                output_dir = Path("./collected_data")
            self.recorder = EpisodeRecorder(output_dir=output_dir, format="lerobot")
        else:
            self.recorder = recorder

        # Setup cameras
        self.cameras = cameras or []
        self.camera_recorder = None
        if self.cameras:
            self.camera_recorder = CameraRecorder(
                camera_names=self.cameras, fps=fps
            )

        self._is_collecting = False

    def collect(
        self,
        num_episodes: int = 100,
        output_dir: Optional[Union[str, Path]] = None,
        task_name: Optional[str] = None,
        instructions: Optional[list[str]] = None,
    ) -> int:
        """
        Collect multiple episodes of demonstration data.

        Args:
            num_episodes: Number of episodes to collect
            output_dir: Output directory (overrides recorder's output_dir)
            task_name: Task name for metadata
            instructions: List of language instructions

        Returns:
            Number of episodes successfully collected
        """
        # Update output directory if provided
        if output_dir is not None:
            self.recorder.output_dir = Path(output_dir)
            self.recorder.output_dir.mkdir(parents=True, exist_ok=True)

        # Connect robot and teleoperation
        if not self._setup_connections():
            print("Failed to setup connections")
            return 0

        print(f"\nData Collection Setup:")
        print(f"  Robot: {type(self.robot).__name__}")
        print(f"  Teleoperation: {type(self.teleop).__name__}")
        print(f"  Output: {self.recorder.output_dir}")
        print(f"  FPS: {self.fps}")
        print(f"  Cameras: {self.cameras}")
        print(f"\nReady to collect {num_episodes} episodes!")
        print("Press Enter to start each episode, 'q' to quit")

        episodes_collected = 0

        try:
            for episode_idx in range(num_episodes):
                print(f"\n{'=' * 60}")
                print(f"Episode {episode_idx + 1}/{num_episodes}")
                print(f"{'=' * 60}")

                # Get instruction for this episode
                instruction = None
                if instructions and len(instructions) > 0:
                    instruction = instructions[episode_idx % len(instructions)]

                # Wait for user to start episode
                user_input = input("\nPress Enter to start recording (or 'q' to quit): ")
                if user_input.lower() == "q":
                    break

                # Collect episode
                success = self._collect_episode(
                    task_name=task_name, instruction=instruction
                )

                if success:
                    episodes_collected += 1
                    print(f"✓ Episode saved! Total: {episodes_collected}")
                else:
                    print("✗ Episode collection failed")

        except KeyboardInterrupt:
            print("\n\nCollection interrupted by user")
        finally:
            self._cleanup_connections()

        print(f"\nCollection complete! {episodes_collected} episodes saved")
        return episodes_collected

    def _setup_connections(self) -> bool:
        """
        Setup robot and teleoperation connections.

        Returns:
            bool: True if all connections successful
        """
        # Connect robot
        if hasattr(self.robot, "is_connected") and not self.robot.is_connected:
            try:
                if hasattr(self.robot, "connect"):
                    self.robot.connect()
                    print("✓ Robot connected")
            except Exception as e:
                print(f"✗ Failed to connect robot: {e}")
                return False

        # Connect teleoperation
        if not self.teleop.is_connected:
            try:
                self.teleop.connect()
                print("✓ Teleoperation connected")
            except Exception as e:
                print(f"✗ Failed to connect teleoperation: {e}")
                return False

        # Setup cameras
        if self.camera_recorder:
            try:
                self.camera_recorder.setup_cameras()
                print(f"✓ {len(self.cameras)} camera(s) initialized")
            except Exception as e:
                print(f"✗ Failed to setup cameras: {e}")

        return True

    def _cleanup_connections(self) -> None:
        """Cleanup all connections."""
        try:
            if self.teleop:
                self.teleop.disconnect()
        except Exception:
            pass

        try:
            if self.camera_recorder:
                self.camera_recorder.release_cameras()
        except Exception:
            pass

    def _collect_episode(
        self, task_name: Optional[str] = None, instruction: Optional[str] = None
    ) -> bool:
        """
        Collect a single episode.

        Args:
            task_name: Task name for metadata
            instruction: Language instruction

        Returns:
            bool: True if episode collected successfully
        """
        try:
            # Start recording
            self.recorder.start_episode(task_name=task_name, instruction=instruction)

            if self.camera_recorder:
                self.camera_recorder.start_recording()

            print("\n🔴 Recording... Press Ctrl+C to stop")
            self._is_collecting = True

            # Collection loop
            step_count = 0
            start_time = time.time()

            try:
                while self._is_collecting:
                    step_start = time.time()

                    # Get action from teleoperation
                    action = self.teleop.get_action()

                    # Execute action on robot
                    if hasattr(self.robot, "execute_action"):
                        self.robot.execute_action(action)

                    # Get robot state
                    proprioception = None
                    if hasattr(self.robot, "get_state"):
                        state = self.robot.get_state()
                        if "joint_positions" in state:
                            proprioception = state["joint_positions"]

                    # Capture images
                    images = None
                    if self.camera_recorder:
                        images = self.camera_recorder.capture_frame()

                    # Record timestep
                    self.recorder.add_timestep(
                        images=images, action=action, proprioception=proprioception
                    )

                    step_count += 1

                    # Maintain frame rate
                    elapsed = time.time() - step_start
                    if elapsed < self.frame_time:
                        time.sleep(self.frame_time - elapsed)

            except KeyboardInterrupt:
                print("\n⏹ Stopping recording...")

            # Stop recording
            if self.camera_recorder:
                self.camera_recorder.stop_recording()

            episode_data = self.recorder.stop_episode()
            self._is_collecting = False

            # Save episode
            saved_path = self.recorder.save_episode(episode_data)

            duration = time.time() - start_time
            print(f"\nEpisode stats:")
            print(f"  Duration: {duration:.2f}s")
            print(f"  Steps: {step_count}")
            print(f"  Avg FPS: {step_count / duration:.1f}")
            print(f"  Saved to: {saved_path}")

            return saved_path is not None

        except Exception as e:
            print(f"Error during episode collection: {e}")
            self._is_collecting = False
            return False

    def get_status(self) -> dict[str, Any]:
        """
        Get collector status.

        Returns:
            Dictionary containing status information
        """
        return {
            "robot": type(self.robot).__name__,
            "teleop": type(self.teleop).__name__,
            "cameras": self.cameras,
            "fps": self.fps,
            "is_collecting": self._is_collecting,
            "recorder_status": self.recorder.get_status(),
        }


__all__ = ["DataCollector"]
