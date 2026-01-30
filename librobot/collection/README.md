# Data Collection Module

The `librobot.collection` module provides comprehensive utilities for collecting robot demonstration data. It includes teleoperation interfaces, recording utilities, and data format converters.

## Features

### Teleoperation Interfaces

Support for multiple teleoperation devices:
- **Keyboard** - Simple keyboard-based control
- **SpaceMouse** - 6-DOF 3Dconnexion SpaceMouse
- **VR Controllers** - Oculus Quest, HTC Vive, etc.
- **Motion Capture** - OptiTrack, Vicon, etc.
- **Leader-Follower** - Bilateral teleoperation with leader robot

All interfaces follow a common API:
```python
from librobot.collection import KeyboardTeleop

teleop = KeyboardTeleop(action_dim=7)
teleop.connect()
action = teleop.get_action()  # Returns numpy array
teleop.disconnect()
```

### Recording Utilities

#### DataBuffer
Thread-safe buffer for efficient data storage:
```python
from librobot.collection import DataBuffer

buffer = DataBuffer(max_size=1000)
buffer.append("actions", action)
buffer.append("images", image)
data = buffer.get_all_streams()
```

#### CameraRecorder
Multi-camera recording with configurable settings:
```python
from librobot.collection import CameraRecorder

recorder = CameraRecorder(
    camera_names=["wrist", "front"],
    resolution=(640, 480),
    fps=30
)
recorder.start_recording()
frames = recorder.capture_frame()
```

#### EpisodeRecorder
Complete episode recording with metadata:
```python
from librobot.collection import EpisodeRecorder

recorder = EpisodeRecorder(
    output_dir="./data",
    format="lerobot"
)
recorder.start_episode(task_name="pick_and_place")
recorder.add_timestep(action=action, images=images)
recorder.save_episode()
```

### Data Format Converters

Support for multiple dataset formats:
- **LeRobot** - JSON metadata + numpy arrays
- **HDF5** - Hierarchical Data Format
- **Zarr** - Cloud-native chunked arrays
- **RLDS** - TensorFlow Datasets format

```python
from librobot.collection import LeRobotConverter

converter = LeRobotConverter()
converter.write_episode("./data", episode_data)
episode = converter.read_episode("./data", episode_idx=0)
```

### DataCollector

Main orchestrator for the collection process:
```python
from librobot.collection import DataCollector
from librobot.robots import create_robot

# Setup
robot = create_robot("franka")
collector = DataCollector(
    robot=robot,
    teleop="spacemouse",
    cameras=["wrist", "front"],
    fps=30,
    output_dir="./data"
)

# Collect episodes
collector.collect(
    num_episodes=100,
    task_name="pick_and_place",
    instructions=["Pick up the red block", "Place it in the bin"]
)
```

## CLI Usage

The collection module is integrated with the CLI:

```bash
# Basic usage
librobot-collect --robot franka --output ./data

# With teleoperation
librobot-collect --robot franka --teleoperation spacemouse

# With cameras
librobot-collect --robot franka --cameras wrist front --fps 30

# With task and instructions
librobot-collect --robot franka --task pick_and_place --instructions instructions.txt

# With format
librobot-collect --robot franka --format hdf5
```

## Registry Pattern

All teleoperation interfaces are registered and can be created by name:

```python
from librobot.collection import create_teleop, list_teleoperation

# List available interfaces
available = list_teleoperation()
print(available)  # ['keyboard', 'spacemouse', 'vr', 'mocap', 'leader_follower']

# Create by name
teleop = create_teleop("keyboard", action_dim=7)
```

## Optional Dependencies

The module handles optional dependencies gracefully:

- **pyspacemouse** - For SpaceMouse support
- **openvr** - For VR controller support
- **h5py** - For HDF5 format
- **zarr** - For Zarr format
- **tensorflow-datasets** - For RLDS format

If a dependency is not installed, the module will print a warning but continue to work with other features.

## Thread Safety

Recording components (DataBuffer, CameraRecorder, EpisodeRecorder) are thread-safe and can be used in multi-threaded environments.

## Examples

See `examples/collection_demo.py` for comprehensive usage examples.

## Testing

Run tests with:
```bash
pytest tests/unit/test_collection.py -v
```

## Module Structure

```
librobot/collection/
├── __init__.py              # Public API exports
├── base.py                  # Abstract base classes
├── collector.py             # Main DataCollector class
├── teleoperation/
│   ├── __init__.py
│   ├── base.py              # Registry and AbstractTeleop
│   ├── keyboard.py          # KeyboardTeleop
│   ├── spacemouse.py        # SpaceMouseTeleop
│   ├── vr.py                # VRTeleop
│   ├── mocap.py             # MocapTeleop
│   └── leader_follower.py   # LeaderFollowerTeleop
├── recording/
│   ├── __init__.py
│   ├── data_buffer.py       # DataBuffer
│   ├── camera_recorder.py   # CameraRecorder
│   └── episode_recorder.py  # EpisodeRecorder
└── converters/
    ├── __init__.py
    ├── base.py              # Registry and AbstractConverter
    ├── lerobot.py           # LeRobotConverter
    ├── hdf5.py              # HDF5Converter
    ├── zarr.py              # ZarrConverter
    └── rlds.py              # RLDSConverter
```

## Design Principles

- **Modular**: Each component can be used independently
- **Extensible**: Easy to add new teleoperation interfaces or converters
- **Consistent**: Follows registry pattern used in other modules
- **Robust**: Graceful handling of missing dependencies
- **Safe**: Thread-safe operations where needed
- **Documented**: Comprehensive docstrings and type hints
