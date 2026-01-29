# Guide: Adding New Robots

This guide shows you how to integrate your robot with LibroBot VLA. Whether you have a physical robot arm, mobile robot, or custom hardware, this guide will help you get started.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Robot Interface](#robot-interface)
- [Step-by-Step Guide](#step-by-step-guide)
- [Common Patterns](#common-patterns)
- [Testing Your Robot](#testing-your-robot)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

### What is a Robot Interface?

A robot interface is a Python class that handles:
1. **Action execution**: Sending commands to your robot
2. **State reading**: Getting current joint positions, velocities, etc.
3. **Observation collection**: Reading camera images, sensors
4. **Safety**: Emergency stops, limit checking
5. **Reset**: Returning to home position

### Why Create a Robot Interface?

- **Standardization**: Work with any VLA model
- **Reusability**: Use the same interface for different experiments
- **Safety**: Built-in safety checks and limits
- **Testing**: Easy to mock and test
- **Deployment**: Seamless transition from training to real robot

## Quick Start

### Minimal Robot Interface

Here's the simplest possible robot interface:

```python
from librobot.robots import AbstractRobot, register_robot
import numpy as np

@register_robot(name="my-robot")
class MyRobot(AbstractRobot):
    """Simple robot interface."""
    
    def __init__(self, action_dim=7, state_dim=14):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self._is_connected = False
    
    def connect(self):
        """Connect to robot hardware."""
        # TODO: Add your connection logic
        print("Connecting to robot...")
        self._is_connected = True
    
    def disconnect(self):
        """Disconnect from robot."""
        print("Disconnecting from robot...")
        self._is_connected = False
    
    def get_observation(self):
        """Get current observation."""
        return {
            "images": {
                "wrist_camera": np.zeros((224, 224, 3), dtype=np.uint8),
                "base_camera": np.zeros((224, 224, 3), dtype=np.uint8),
            },
            "proprioception": np.zeros(self.state_dim),
        }
    
    def execute_action(self, action):
        """Execute an action."""
        if not self._is_connected:
            raise RuntimeError("Robot not connected")
        
        # TODO: Send action to robot
        print(f"Executing action: {action}")
        return True
    
    def reset(self):
        """Reset robot to home position."""
        print("Resetting robot...")
        return self.get_observation()
    
    def is_connected(self):
        """Check if robot is connected."""
        return self._is_connected
```

### Using Your Robot

```python
from librobot.robots import create_robot

# Create robot instance
robot = create_robot("my-robot")

# Connect
robot.connect()

# Get observation
obs = robot.get_observation()
print(f"Images: {obs['images'].keys()}")
print(f"Proprioception shape: {obs['proprioception'].shape}")

# Execute action
action = np.array([0.1, 0.2, 0.0, 0.1, 0.0, 0.0, 1.0])
robot.execute_action(action)

# Reset
robot.reset()

# Disconnect
robot.disconnect()
```

## Robot Interface

### Abstract Interface

All robots must implement `AbstractRobot`:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

class AbstractRobot(ABC):
    """Abstract base class for robot interfaces."""
    
    @abstractmethod
    def connect(self) -> None:
        """Connect to robot hardware/simulator."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from robot."""
        pass
    
    @abstractmethod
    def get_observation(self) -> Dict[str, Any]:
        """
        Get current robot observation.
        
        Returns:
            Dictionary containing:
            - "images": Dict of camera images
            - "proprioception": Robot state vector
            - "metadata": Optional metadata
        """
        pass
    
    @abstractmethod
    def execute_action(self, action: np.ndarray) -> bool:
        """
        Execute an action on the robot.
        
        Args:
            action: Action vector
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """
        Reset robot to initial state.
        
        Returns:
            Initial observation
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        pass
    
    # Optional methods with default implementations
    
    def emergency_stop(self) -> None:
        """Emergency stop (default: do nothing)."""
        pass
    
    def get_action_space(self) -> Dict[str, Any]:
        """Get action space specification."""
        return {
            "type": "continuous",
            "dim": 7,
            "bounds": {"min": [-1.0] * 7, "max": [1.0] * 7}
        }
    
    def get_observation_space(self) -> Dict[str, Any]:
        """Get observation space specification."""
        return {
            "images": {},
            "proprioception": {"dim": 14}
        }
```

### Required Methods

You **must** implement:
- `connect()`: Initialize connection
- `disconnect()`: Clean up connection
- `get_observation()`: Return current state
- `execute_action(action)`: Send commands
- `reset()`: Return to home state
- `is_connected()`: Check connection status

### Optional Methods

You **may** implement:
- `emergency_stop()`: E-stop functionality
- `get_action_space()`: Action space definition
- `get_observation_space()`: Observation space definition
- `set_control_mode(mode)`: Switch control modes
- `get_diagnostics()`: Health checks

## Step-by-Step Guide

### Step 1: Define Your Robot

Create a file `librobot/robots/my_robot.py`:

```python
"""
My Robot Interface
==================

Interface for controlling MyRobot hardware.

Connection:
    - Serial port: /dev/ttyUSB0
    - Baud rate: 115200
    
Action Space:
    - 6 DOF arm + 1 gripper
    - Joint position control
    - Action range: [-1, 1] (normalized)

Observation Space:
    - 2 cameras (wrist, base)
    - Joint positions (7D)
    - Joint velocities (7D)
"""

from typing import Dict, Any, Optional
import numpy as np
import time

from librobot.robots.base import AbstractRobot
from librobot.robots.registry import register_robot

@register_robot(
    name="my-robot",
    aliases=["myrobot", "mr"],
    description="Interface for MyRobot hardware"
)
class MyRobot(AbstractRobot):
    """MyRobot hardware interface."""
    
    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baud_rate: int = 115200,
        control_frequency: float = 10.0,
        action_scaling: float = 1.0,
        **kwargs
    ):
        """
        Initialize MyRobot interface.
        
        Args:
            port: Serial port for communication
            baud_rate: Communication baud rate
            control_frequency: Control loop frequency (Hz)
            action_scaling: Scaling factor for actions
        """
        super().__init__()
        
        self.port = port
        self.baud_rate = baud_rate
        self.control_frequency = control_frequency
        self.action_scaling = action_scaling
        
        self._connection = None
        self._last_action_time = 0.0
        
        # Action space: 6 DOF + gripper
        self.action_dim = 7
        self.action_bounds = {
            "min": np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]),
            "max": np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        }
        
        # State space: joint positions + velocities
        self.state_dim = 14
        
        # Cameras
        self.camera_names = ["wrist_camera", "base_camera"]
        
    def connect(self) -> None:
        """Connect to robot."""
        # TODO: Replace with your connection logic
        import serial
        
        try:
            self._connection = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=1.0
            )
            time.sleep(2)  # Wait for connection
            print(f"Connected to MyRobot on {self.port}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to robot: {e}")
    
    def disconnect(self) -> None:
        """Disconnect from robot."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            print("Disconnected from MyRobot")
    
    def get_observation(self) -> Dict[str, Any]:
        """Get current observation from robot."""
        if not self.is_connected():
            raise RuntimeError("Robot not connected")
        
        # Read images from cameras
        images = {}
        for camera_name in self.camera_names:
            images[camera_name] = self._read_camera(camera_name)
        
        # Read proprioception (joint states)
        joint_positions = self._read_joint_positions()
        joint_velocities = self._read_joint_velocities()
        proprioception = np.concatenate([joint_positions, joint_velocities])
        
        return {
            "images": images,
            "proprioception": proprioception,
            "timestamp": time.time(),
        }
    
    def execute_action(self, action: np.ndarray) -> bool:
        """Execute action on robot."""
        if not self.is_connected():
            raise RuntimeError("Robot not connected")
        
        # Validate action
        action = self._validate_action(action)
        
        # Rate limiting
        self._wait_for_control_frequency()
        
        # Send action to robot
        try:
            self._send_action(action)
            self._last_action_time = time.time()
            return True
        except Exception as e:
            print(f"Failed to execute action: {e}")
            return False
    
    def reset(self) -> Dict[str, Any]:
        """Reset robot to home position."""
        print("Resetting robot to home position...")
        
        # Define home position
        home_action = np.zeros(self.action_dim)
        
        # Move to home gradually
        current_pos = self._read_joint_positions()
        steps = 20
        
        for i in range(steps):
            alpha = (i + 1) / steps
            target = current_pos * (1 - alpha) + home_action[:len(current_pos)] * alpha
            self.execute_action(target)
            time.sleep(0.1)
        
        print("Reset complete")
        return self.get_observation()
    
    def is_connected(self) -> bool:
        """Check if robot is connected."""
        return self._connection is not None and self._connection.is_open
    
    def emergency_stop(self) -> None:
        """Emergency stop robot."""
        print("EMERGENCY STOP!")
        if self.is_connected():
            # Send stop command
            self._connection.write(b"ESTOP\n")
            time.sleep(0.1)
    
    def get_action_space(self) -> Dict[str, Any]:
        """Get action space specification."""
        return {
            "type": "continuous",
            "dim": self.action_dim,
            "bounds": self.action_bounds,
            "names": ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]
        }
    
    def get_observation_space(self) -> Dict[str, Any]:
        """Get observation space specification."""
        return {
            "images": {
                name: {"shape": [224, 224, 3], "dtype": "uint8"}
                for name in self.camera_names
            },
            "proprioception": {"dim": self.state_dim}
        }
    
    # Private helper methods
    
    def _read_camera(self, camera_name: str) -> np.ndarray:
        """Read image from camera."""
        # TODO: Implement camera reading
        # Example with OpenCV:
        # import cv2
        # camera_id = 0 if camera_name == "wrist_camera" else 1
        # cap = cv2.VideoCapture(camera_id)
        # ret, frame = cap.read()
        # cap.release()
        # return cv2.resize(frame, (224, 224))
        
        # Placeholder
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    def _read_joint_positions(self) -> np.ndarray:
        """Read joint positions from robot."""
        # TODO: Implement reading from robot
        # Example:
        # self._connection.write(b"GET_POS\n")
        # response = self._connection.readline()
        # positions = parse_positions(response)
        
        # Placeholder
        return np.zeros(7)
    
    def _read_joint_velocities(self) -> np.ndarray:
        """Read joint velocities from robot."""
        # TODO: Implement reading from robot
        return np.zeros(7)
    
    def _send_action(self, action: np.ndarray) -> None:
        """Send action command to robot."""
        # TODO: Implement sending to robot
        # Example:
        # command = f"MOVE {','.join(map(str, action))}\n"
        # self._connection.write(command.encode())
        pass
    
    def _validate_action(self, action: np.ndarray) -> np.ndarray:
        """Validate and clip action to safe bounds."""
        action = np.array(action)
        
        if action.shape[0] != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got {action.shape[0]}")
        
        # Clip to bounds
        action = np.clip(
            action,
            self.action_bounds["min"],
            self.action_bounds["max"]
        )
        
        # Apply scaling
        action = action * self.action_scaling
        
        return action
    
    def _wait_for_control_frequency(self) -> None:
        """Wait to maintain control frequency."""
        time_since_last = time.time() - self._last_action_time
        required_time = 1.0 / self.control_frequency
        
        if time_since_last < required_time:
            time.sleep(required_time - time_since_last)
```

### Step 2: Create Configuration

Create `configs/robot/my_robot.yaml`:

```yaml
# Robot: MyRobot Configuration

name: "my-robot"
type: "hardware"

# Connection settings
connection:
  port: "/dev/ttyUSB0"
  baud_rate: 115200
  timeout: 1.0

# Control settings
control:
  frequency: 10  # Hz
  action_repeat: 1
  action_scaling: 1.0

# Action space
action_space:
  type: "continuous"
  dim: 7
  bounds:
    min: [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]
    max: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  names: ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]

# Observation space
observation_space:
  # Camera configuration
  cameras:
    wrist_camera:
      device_id: 0
      resolution: [224, 224]
      fps: 30
    base_camera:
      device_id: 1
      resolution: [224, 224]
      fps: 30
  
  # Proprioception
  proprioception:
    joint_positions:
      dim: 7
      bounds: [-3.14, 3.14]
    joint_velocities:
      dim: 7
      bounds: [-2.0, 2.0]

# Safety limits
safety:
  max_velocity: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5]
  max_acceleration: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0]
  workspace_limits:
    x: [-0.5, 0.5]
    y: [-0.5, 0.5]
    z: [0.0, 0.8]
  emergency_stop_enabled: true

# Calibration
calibration:
  home_position: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  joint_offsets: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

### Step 3: Register Your Robot

The `@register_robot` decorator automatically registers your robot:

```python
# In librobot/robots/__init__.py (if not already there)
from librobot.robots.my_robot import MyRobot  # This triggers registration
```

Or import in your script:

```python
# Auto-registers when imported
import librobot.robots.my_robot
```

### Step 4: Test Your Robot

Create a test script `test_my_robot.py`:

```python
from librobot.robots import create_robot
import numpy as np

def test_robot():
    """Test robot interface."""
    
    # Create robot
    robot = create_robot("my-robot", port="/dev/ttyUSB0")
    
    # Connect
    print("Connecting...")
    robot.connect()
    assert robot.is_connected()
    
    # Get observation
    print("\nGetting observation...")
    obs = robot.get_observation()
    print(f"Image keys: {obs['images'].keys()}")
    print(f"Proprio shape: {obs['proprioception'].shape}")
    
    # Execute action
    print("\nExecuting action...")
    action = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    success = robot.execute_action(action)
    assert success
    
    # Reset
    print("\nResetting...")
    initial_obs = robot.reset()
    
    # Disconnect
    print("\nDisconnecting...")
    robot.disconnect()
    
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_robot()
```

## Common Patterns

### Pattern 1: Simulation vs Real

Support both simulation and real hardware:

```python
@register_robot(name="my-robot")
class MyRobot(AbstractRobot):
    def __init__(self, sim: bool = False, **kwargs):
        super().__init__()
        self.sim = sim
        
        if sim:
            self._init_simulation()
        else:
            self._init_hardware()
    
    def _init_simulation(self):
        """Initialize in simulation."""
        import pybullet as p
        self.client = p.connect(p.GUI)
        # Load robot URDF, etc.
    
    def _init_hardware(self):
        """Initialize real hardware."""
        # Connect to serial port, etc.
```

### Pattern 2: Multi-Camera Support

Handle multiple cameras efficiently:

```python
class MyRobot(AbstractRobot):
    def __init__(self, camera_config: Dict[str, int]):
        self.cameras = {}
        for name, device_id in camera_config.items():
            self.cameras[name] = Camera(device_id)
    
    def get_observation(self):
        images = {
            name: cam.read()
            for name, cam in self.cameras.items()
        }
        return {"images": images, ...}
```

### Pattern 3: Action Buffering

Buffer actions for smoother control:

```python
from collections import deque

class MyRobot(AbstractRobot):
    def __init__(self, buffer_size: int = 5):
        self.action_buffer = deque(maxlen=buffer_size)
    
    def execute_action(self, action):
        self.action_buffer.append(action)
        
        # Send smoothed action
        smoothed = np.mean(self.action_buffer, axis=0)
        self._send_action(smoothed)
```

### Pattern 4: Asynchronous Observation

Collect observations asynchronously:

```python
import threading
from queue import Queue

class MyRobot(AbstractRobot):
    def __init__(self):
        self.obs_queue = Queue(maxsize=1)
        self.obs_thread = None
    
    def connect(self):
        super().connect()
        self.obs_thread = threading.Thread(target=self._observation_loop)
        self.obs_thread.start()
    
    def _observation_loop(self):
        while self.is_connected():
            obs = self._read_observation()
            if self.obs_queue.full():
                self.obs_queue.get()  # Remove old
            self.obs_queue.put(obs)
    
    def get_observation(self):
        return self.obs_queue.get()
```

### Pattern 5: Safety Wrapper

Add safety checks as a wrapper:

```python
class SafetyWrapper(AbstractRobot):
    """Wraps a robot with safety checks."""
    
    def __init__(self, robot: AbstractRobot, safety_config: Dict):
        self.robot = robot
        self.safety = safety_config
        self.last_position = None
    
    def execute_action(self, action):
        # Check velocity limits
        if self.last_position is not None:
            velocity = action - self.last_position
            if np.any(np.abs(velocity) > self.safety["max_velocity"]):
                print("WARNING: Velocity limit exceeded!")
                action = self._clip_velocity(action, velocity)
        
        # Check workspace limits
        if not self._in_workspace(action):
            print("WARNING: Outside workspace!")
            return False
        
        self.last_position = action
        return self.robot.execute_action(action)

# Use it
base_robot = create_robot("my-robot")
safe_robot = SafetyWrapper(base_robot, safety_config)
```

## Testing Your Robot

### Unit Tests

Create `tests/test_my_robot.py`:

```python
import pytest
import numpy as np
from librobot.robots import create_robot

class TestMyRobot:
    @pytest.fixture
    def robot(self):
        """Create robot instance for testing."""
        robot = create_robot("my-robot", sim=True)  # Use simulation
        robot.connect()
        yield robot
        robot.disconnect()
    
    def test_connection(self, robot):
        """Test robot connection."""
        assert robot.is_connected()
    
    def test_observation_format(self, robot):
        """Test observation format."""
        obs = robot.get_observation()
        
        assert "images" in obs
        assert "proprioception" in obs
        assert obs["proprioception"].shape[0] == robot.state_dim
    
    def test_action_execution(self, robot):
        """Test action execution."""
        action = np.zeros(robot.action_dim)
        success = robot.execute_action(action)
        assert success
    
    def test_action_bounds(self, robot):
        """Test action bound checking."""
        # Action within bounds should work
        action = np.zeros(robot.action_dim)
        success = robot.execute_action(action)
        assert success
        
        # Action outside bounds should be clipped
        action = np.ones(robot.action_dim) * 10.0
        success = robot.execute_action(action)
        assert success  # Should clip and succeed
    
    def test_reset(self, robot):
        """Test reset functionality."""
        initial_obs = robot.reset()
        assert "images" in initial_obs
        assert "proprioception" in initial_obs

if __name__ == "__main__":
    pytest.main([__file__])
```

### Integration Test

Test with actual VLA model:

```python
from librobot.robots import create_robot
from librobot.models import create_vla, create_vlm
import torch

def test_vla_robot_integration():
    """Test VLA model with robot."""
    
    # Create robot
    robot = create_robot("my-robot", sim=True)
    robot.connect()
    
    # Create VLA model
    vlm = create_vlm("qwen2-vl-2b", pretrained=True)
    vla = create_vla("groot", vlm=vlm, action_dim=robot.action_dim)
    
    # Get observation
    obs = robot.get_observation()
    
    # Predict action
    images = torch.from_numpy(obs["images"]["wrist_camera"]).float() / 255.0
    images = images.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    
    text = ["pick up the object"]
    proprio = torch.from_numpy(obs["proprioception"]).float().unsqueeze(0)
    
    with torch.no_grad():
        action = vla.predict_action(images, text, proprio)
    
    # Execute on robot
    success = robot.execute_action(action[0].cpu().numpy())
    assert success
    
    robot.disconnect()
    print("✓ Integration test passed!")

if __name__ == "__main__":
    test_vla_robot_integration()
```

## Examples

### Example 1: UR5 Robot Arm

```python
@register_robot(name="ur5")
class UR5Robot(AbstractRobot):
    """Universal Robots UR5 interface."""
    
    def __init__(self, ip_address: str = "192.168.1.100"):
        import rtde_control
        import rtde_receive
        
        self.control = rtde_control.RTDEControlInterface(ip_address)
        self.receive = rtde_receive.RTDEReceiveInterface(ip_address)
        
        self.action_dim = 6
        self.state_dim = 12
    
    def execute_action(self, action):
        self.control.servoJ(action, velocity=0.5, acceleration=0.5)
        return True
    
    def get_observation(self):
        positions = self.receive.getActualQ()
        velocities = self.receive.getActualQd()
        
        return {
            "images": self._get_camera_images(),
            "proprioception": np.concatenate([positions, velocities])
        }
```

### Example 2: Mobile Robot

```python
@register_robot(name="mobile-robot")
class MobileRobot(AbstractRobot):
    """Differential drive mobile robot."""
    
    def __init__(self):
        self.action_dim = 2  # [linear_vel, angular_vel]
        self.state_dim = 3   # [x, y, theta]
    
    def execute_action(self, action):
        linear_vel, angular_vel = action
        
        # Send to motors
        self.left_motor.set_velocity(linear_vel - angular_vel)
        self.right_motor.set_velocity(linear_vel + angular_vel)
        
        return True
    
    def get_observation(self):
        return {
            "images": {"front_camera": self._read_camera()},
            "proprioception": self._get_odometry()
        }
```

### Example 3: Gripper Only

```python
@register_robot(name="gripper")
class GripperRobot(AbstractRobot):
    """Simple gripper interface."""
    
    def __init__(self, port: str = "/dev/ttyUSB0"):
        import serial
        self.serial = serial.Serial(port, 115200)
        self.action_dim = 1  # Gripper open/close
        self.state_dim = 1   # Gripper position
    
    def execute_action(self, action):
        gripper_pos = np.clip(action[0], 0.0, 1.0)
        command = f"G{int(gripper_pos * 255)}\n"
        self.serial.write(command.encode())
        return True
    
    def get_observation(self):
        self.serial.write(b"P\n")  # Query position
        response = self.serial.readline()
        position = float(response.decode().strip()) / 255.0
        
        return {
            "images": {},  # No cameras
            "proprioception": np.array([position])
        }
```

## Troubleshooting

### Issue 1: Connection Fails

**Problem**: Cannot connect to robot

**Solutions**:
1. Check physical connection (USB, Ethernet)
2. Verify port/IP address: `ls /dev/ttyUSB*`
3. Check permissions: `sudo chmod 666 /dev/ttyUSB0`
4. Test with minimal tool (e.g., `minicom`, `ping`)

### Issue 2: Actions Not Executing

**Problem**: Robot doesn't move

**Solutions**:
1. Check if robot is enabled/powered
2. Verify action is within bounds
3. Check control frequency isn't too high
4. Add debug prints in `_send_action()`
5. Test with known-good action

### Issue 3: Image Quality Issues

**Problem**: Camera images are poor quality

**Solutions**:
1. Check camera focus and exposure
2. Verify resolution and FPS settings
3. Test camera separately with OpenCV
4. Check USB bandwidth (reduce resolution if needed)
5. Ensure proper lighting

### Issue 4: Timing Issues

**Problem**: Control loop is too slow/fast

**Solutions**:
1. Profile your code: `python -m cProfile`
2. Move camera reading to separate thread
3. Reduce image resolution
4. Use hardware acceleration if available
5. Adjust `control_frequency` parameter

### Issue 5: Safety Concerns

**Problem**: Robot makes unsafe movements

**Solutions**:
1. Add velocity limiting
2. Implement emergency stop
3. Check action bounds
4. Add workspace limits
5. Test in simulation first
6. Use a SafetyWrapper

### Getting Help

If you're stuck:

1. Check the examples: `examples/` directory
2. Look at existing robots: `librobot/robots/`
3. Search GitHub issues
4. Ask on Discussions
5. Open an issue with:
   - Robot type and model
   - Error messages
   - Minimal reproduction code
   - Expected vs actual behavior

---

**Next Steps:**

- [Adding Models](adding_models.md): Add custom VLMs or frameworks
- [Configuration Guide](configuration.md): Configure your robot
- [Deployment Guide](deployment.md): Deploy to production

For technical details, see [design/COMPONENT_GUIDE.md](design/COMPONENT_GUIDE.md#adding-a-new-robot-interface).
