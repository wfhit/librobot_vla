"""Data collection CLI command."""

import argparse
from pathlib import Path
from typing import Optional
import sys


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
        "--robot", "-r",
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
        "--teleoperation", "-t",
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
        "--output", "-o",
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
        print(f"Starting data collection...")
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
        
        # Collection loop
        print("\nReady to collect data!")
        print("Press 's' to start/stop recording")
        print("Press 'q' to quit")
        
        episode_count = 0
        
        while episode_count < args.max_episodes:
            # Simplified collection loop
            print(f"\nEpisode {episode_count + 1}/{args.max_episodes}")
            print("Recording... (simulated)")
            
            # Record episode
            episode_data = record_episode(robot, teleop, args)
            
            if episode_data:
                recorder.save_episode(episode_data)
                episode_count += 1
                print(f"Episode saved! Total: {episode_count}")
            
            # Check for user input
            user_input = input("Continue? (y/n/q): ").lower()
            if user_input == 'q':
                break
            elif user_input == 'n':
                continue
        
        print(f"\nCollection complete! {episode_count} episodes saved to {args.output}")
        return 0
        
    except KeyboardInterrupt:
        print("\nCollection interrupted")
        return 0
    except Exception as e:
        print(f"Collection failed: {e}")
        return 1


def setup_robot(args):
    """Setup robot connection."""
    from librobot.robots import get_robot, create_robot
    
    try:
        robot = create_robot(args.robot)
        
        connect_kwargs = {}
        if args.robot_ip:
            connect_kwargs['ip'] = args.robot_ip
        if args.robot_port:
            connect_kwargs['port'] = args.robot_port
        
        robot.connect(**connect_kwargs)
        return robot
        
    except Exception as e:
        print(f"Could not connect to robot: {e}")
        print("Using simulated robot")
        return None


def setup_teleoperation(args):
    """Setup teleoperation device."""
    print(f"Setting up {args.teleoperation} teleoperation...")
    
    # Return dummy teleop
    class DummyTeleop:
        def get_action(self):
            import numpy as np
            return np.zeros(7)
    
    return DummyTeleop()


def setup_recorder(args):
    """Setup data recorder."""
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    class SimpleRecorder:
        def __init__(self, output_dir):
            self.output_dir = Path(output_dir)
            self.episode_count = 0
        
        def save_episode(self, data):
            import json
            episode_path = self.output_dir / f"episode_{self.episode_count:06d}.json"
            with open(episode_path, 'w') as f:
                json.dump({"episode": self.episode_count, "length": len(data)}, f)
            self.episode_count += 1
    
    return SimpleRecorder(args.output)


def record_episode(robot, teleop, args):
    """Record a single episode."""
    import numpy as np
    
    # Simulated episode data
    episode_length = np.random.randint(50, 200)
    
    return {
        'images': np.random.randint(0, 255, (episode_length, 480, 640, 3)),
        'actions': np.random.randn(episode_length, 7),
        'proprioception': np.random.randn(episode_length, 14),
    }


def main():
    sys.exit(collect_cli())


if __name__ == "__main__":
    main()
