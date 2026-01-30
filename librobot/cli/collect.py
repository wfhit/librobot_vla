"""Data collection CLI command."""

import argparse
import sys
from pathlib import Path
from typing import Optional


def collect_cli(args: Optional[list] = None) -> int:
    """
    Collect robot demonstration data.

    Usage:
        librobot-collect --robot franka --output ./data
        librobot-collect --robot so100 --teleoperation spacemouse
    """
    parser = argparse.ArgumentParser(
        description="Collect robot demonstration data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Robot
    parser.add_argument(
        "--robot",
        "-r",
        type=str,
        required=True,
        help="Robot type (franka, so100, ur5, etc.)",
    )
    parser.add_argument(
        "--robot-ip",
        type=str,
        help="Robot IP address",
    )
    parser.add_argument(
        "--robot-port",
        type=str,
        help="Robot port or USB device",
    )

    # Teleoperation
    parser.add_argument(
        "--teleoperation",
        "-t",
        type=str,
        default="keyboard",
        choices=["keyboard", "spacemouse", "vr", "mocap", "leader_follower"],
        help="Teleoperation method",
    )
    parser.add_argument(
        "--leader-robot",
        type=str,
        help="Leader robot for leader-follower teleoperation",
    )

    # Recording
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./collected_data",
        help="Output directory",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="lerobot",
        choices=["lerobot", "hdf5", "zarr"],
        help="Data format",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Recording frame rate",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=100,
        help="Maximum episodes to collect",
    )

    # Cameras
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["wrist", "front"],
        help="Camera names to record",
    )
    parser.add_argument(
        "--camera-resolution",
        type=str,
        default="640x480",
        help="Camera resolution",
    )

    # Task
    parser.add_argument(
        "--task",
        type=str,
        help="Task name for metadata",
    )
    parser.add_argument(
        "--instructions",
        type=str,
        help="Path to file with language instructions",
    )

    # Misc
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show visualization during collection",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )

    parsed_args = parser.parse_args(args)
    return run_collection(parsed_args)


def run_collection(args) -> int:
    """Run data collection."""
    try:
        from librobot.collection import DataCollector

        print("Starting data collection...")
        print(f"  Robot: {args.robot}")
        print(f"  Teleoperation: {args.teleoperation}")
        print(f"  Output: {args.output}")
        print(f"  Format: {args.format}")
        print(f"  FPS: {args.fps}")
        print(f"  Max episodes: {args.max_episodes}")

        # Setup robot
        robot = setup_robot(args)

        # Setup teleoperation
        teleop = setup_teleoperation(args)

        # Setup recorder
        recorder = setup_recorder(args)

        # Load instructions if provided
        instructions = None
        if args.instructions:
            instructions_path = Path(args.instructions)
            if instructions_path.exists():
                with open(instructions_path, "r") as f:
                    instructions = [line.strip() for line in f if line.strip()]

        # Create DataCollector
        collector = DataCollector(
            robot=robot,
            teleop=teleop,
            recorder=recorder,
            cameras=args.cameras,
            fps=args.fps,
            output_dir=args.output,
        )

        # Collect episodes
        episodes_collected = collector.collect(
            num_episodes=args.max_episodes,
            task_name=args.task,
            instructions=instructions,
        )

        print(f"\nCollection complete! {episodes_collected} episodes saved to {args.output}")
        return 0

    except KeyboardInterrupt:
        print("\nCollection interrupted")
        return 0
    except Exception as e:
        print(f"Collection failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


def setup_robot(args):
    """Setup robot connection."""
    from librobot.robots import create_robot

    try:
        robot = create_robot(args.robot)

        connect_kwargs = {}
        if args.robot_ip:
            connect_kwargs["ip"] = args.robot_ip
        if args.robot_port:
            connect_kwargs["port"] = args.robot_port

        robot.connect(**connect_kwargs)
        return robot

    except Exception as e:
        print(f"Could not connect to robot: {e}")
        print("Using simulated robot")
        return None


def setup_teleoperation(args):
    """Setup teleoperation device."""
    from librobot.collection import create_teleop

    print(f"Setting up {args.teleoperation} teleoperation...")

    try:
        # Create teleoperation interface using the collection module
        teleop = create_teleop(args.teleoperation)

        # For leader-follower, pass the leader robot
        if args.teleoperation == "leader_follower" and args.leader_robot:
            from librobot.robots import create_robot

            leader = create_robot(args.leader_robot)
            teleop.connect(leader_robot=leader)
        else:
            teleop.connect()

        return teleop

    except Exception as e:
        print(f"Could not setup teleoperation: {e}")
        print("Using simulated teleoperation")

        # Return dummy teleop as fallback
        import numpy as np

        class DummyTeleop:
            def get_action(self):
                return np.zeros(7)

            def is_connected(self):
                return True

            def disconnect(self):
                pass

        return DummyTeleop()


def setup_recorder(args):
    """Setup data recorder."""
    from librobot.collection import EpisodeRecorder

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Create episode recorder using the collection module
    recorder = EpisodeRecorder(output_dir=args.output, format=args.format)

    return recorder


def record_episode(robot, teleop, args):
    """Record a single episode - placeholder for backward compatibility."""
    import numpy as np

    # This function is now mostly handled by DataCollector
    # Keep a simple version for backward compatibility
    episode_length = np.random.randint(50, 200)

    return {
        "images": np.random.randint(0, 255, (episode_length, 480, 640, 3)),
        "actions": np.random.randn(episode_length, 7),
        "proprioception": np.random.randn(episode_length, 14),
    }


def main():
    sys.exit(collect_cli())


if __name__ == "__main__":
    main()
