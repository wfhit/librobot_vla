# Robot Configuration Directory

This directory contains configuration files for different robot platforms used with LibroBot VLA.

## Purpose

Robot configs define:
- Physical parameters (DOF, joint limits, gripper specs)
- Camera configurations (positions, intrinsics, image sizes)
- Action space specifications
- Control frequencies
- Observation keys and preprocessing

## Configuration Structure

Each robot config should include:

```yaml
# Robot identifier
name: "franka_panda"

# Physical parameters
dof: 7  # Degrees of freedom
has_gripper: true
control_frequency: 20  # Hz

# Joint configuration
joints:
  names: ["joint1", "joint2", ...]
  position_limits: [[-2.8973, 2.8973], ...]
  velocity_limits: [[-2.1750, 2.1750], ...]
  effort_limits: [[-87, 87], ...]

# Camera configuration
cameras:
  - name: "wrist_camera"
    width: 640
    height: 480
    fov: 69.4
    position: [0.04, 0.0, 0.04]  # Relative to end-effector
    
# Action space
action_space:
  type: "continuous"  # continuous, discrete, hybrid
  low: [-1.0, -1.0, ...]
  high: [1.0, 1.0, ...]
  
# Observation space
observation_keys:
  images: ["wrist_image", "base_image"]
  proprio: ["joint_positions", "joint_velocities", "gripper_position"]
```

## Usage

Reference robot configs in experiment configs:
```yaml
robot: "configs/robot/franka_panda.yaml"
```
