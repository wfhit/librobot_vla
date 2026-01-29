# Wheel Loader VLA Training Example

This example demonstrates how to train a Vision-Language-Action (VLA) model for autonomous wheel loader operation using the LibroBot framework.

## Overview

This configuration uses:
- **Framework**: GR00T-style VLA (frozen VLM + diffusion policy)
- **VLM**: Qwen-VL for visual understanding
- **Action Head**: Diffusion policy (100 steps)
- **Robot**: Wheel loader (6 DOF control)

## Quick Start

### 1. Prepare Your Dataset

Organize your wheel loader demonstration data:

```
/workspace/data/wheel_loader/
├── episodes/
│   ├── episode_0001/
│   │   ├── front_camera/
│   │   │   ├── 00000.jpg
│   │   │   ├── 00001.jpg
│   │   │   └── ...
│   │   ├── rear_camera/
│   │   │   └── ...
│   │   ├── bucket_camera/
│   │   │   └── ...
│   │   ├── actions.npy  # Shape: [T, 6]
│   │   ├── states.npy   # Shape: [T, 12]
│   │   └── metadata.json
│   ├── episode_0002/
│   │   └── ...
│   └── ...
└── metadata.json
```

### 2. Train the Model

```bash
# Using Python directly
python scripts/train.py --config examples/wheel_loader/config.yaml

# Using Docker
cd docker/scripts
./run_train.sh examples/wheel_loader/config.yaml

# Using docker-compose
docker-compose up train
```

### 3. Monitor Training

```bash
# TensorBoard
tensorboard --logdir outputs/wheel_loader_groot/tensorboard

# WandB (if configured)
# Check your WandB dashboard at wandb.ai
```

### 4. Evaluate the Model

```bash
python scripts/evaluate.py \
    --checkpoint outputs/wheel_loader_groot/checkpoints/best.pt \
    --config examples/wheel_loader/config.yaml \
    --split test
```

### 5. Run Inference

```bash
# Single inference
python scripts/inference.py \
    --checkpoint outputs/wheel_loader_groot/checkpoints/best.pt \
    --config examples/wheel_loader/config.yaml \
    --image front_camera.jpg \
    --text "Drive forward and scoop material into bucket"

# Start inference server
python scripts/inference.py \
    --checkpoint outputs/wheel_loader_groot/checkpoints/best.pt \
    --config examples/wheel_loader/config.yaml \
    --server rest \
    --port 8000
```

## Configuration Details

### Action Space (6 DOF)

1. **Steering** [-1, 1]: Left/Right steering angle
2. **Throttle** [0, 1]: Forward throttle
3. **Brake** [0, 1]: Brake pressure
4. **Bucket Tilt** [-1, 1]: Bucket tilt angle
5. **Boom Lift** [-1, 1]: Boom lift height
6. **Transmission** {-1, 0, 1}: Reverse/Neutral/Forward

### State Space (12 DOF)

- Position: x, y, heading (3)
- Velocity: linear_x, linear_y, angular_z (3)
- Hydraulic pressure: boom, bucket (2)
- Load weight (1)
- Engine RPM (1)
- Fuel level (1)
- Temperature (1)

### Camera Setup

- **Front Camera**: 224x224, primary navigation view
- **Rear Camera**: 224x224, backup and maneuvering
- **Bucket Camera**: 224x224, material handling view

## Safety Features

This configuration includes safety features:

- **Speed Limiting**: Maximum 5 m/s
- **Geofencing**: Operation bounded to safe area
- **Emergency Stop**: Manual override capability
- **Hydraulic Monitoring**: Prevent over-pressure
- **Stability Checks**: Prevent tipping

## Customization

### Change VLA Framework

Edit `config.yaml`:

```yaml
model:
  framework: "pi0_style"  # or "octo_style", "openvla_style", etc.
```

### Adjust Training Parameters

```yaml
training:
  batch_size: 64  # Increase for faster training
  num_epochs: 200  # More epochs for better convergence
  lr: 0.0005  # Higher learning rate
```

### Add More Cameras

```yaml
robot:
  cameras:
    - name: "left_side_camera"
      resolution: [224, 224]
    - name: "right_side_camera"
      resolution: [224, 224]
```

## Troubleshooting

### Out of Memory

Reduce batch size:
```yaml
training:
  batch_size: 16  # Reduce from 32
  gradient_accumulation_steps: 2  # Compensate with accumulation
```

### Training Not Converging

Try these adjustments:
```yaml
training:
  optimizer:
    lr: 0.00005  # Lower learning rate
  scheduler:
    warmup_steps: 2000  # More warmup
  max_grad_norm: 0.5  # Stricter gradient clipping
```

### Data Loading Slow

Enable data caching:
```yaml
dataset:
  cache_dataset: true
  num_workers: 8
  prefetch_factor: 4
```

## Expected Results

With ~1000 demonstration episodes:
- **Training Time**: ~12 hours on 1x A100
- **Action MSE**: < 0.05 (normalized actions)
- **Success Rate**: > 85% on validation tasks

## Next Steps

1. **Fine-tuning**: Use this checkpoint for domain-specific fine-tuning
2. **Deployment**: Export model for edge deployment
3. **Sim-to-Real**: Transfer learned policy to real hardware
4. **Multi-Task**: Add more task variations to training data

## References

- [GR00T Architecture](https://developer.nvidia.com/blog/...)
- [Diffusion Policies for Robotics](https://arxiv.org/abs/...)
- [LibroBot Documentation](../../docs/)

## Support

For issues or questions:
- GitHub Issues: https://github.com/wfhit/librobot_vla/issues
- Documentation: [docs/](../../docs/)
