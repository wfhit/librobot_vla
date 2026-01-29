# Wheel Loader VLA Example

This example demonstrates training a VLA model for wheel loader control using the GR00T-style framework.

## Configuration

See `config.yaml` for the complete configuration. Key settings:

- **Model**: GR00T-style framework with frozen VLM + diffusion action head
- **Robot**: Wheel loader (6D action space, 22D state space)
- **Training**: 50K steps with batch size 8

## Usage

### Training

```bash
# From repository root
python scripts/train.py --config examples/wheel_loader/config.yaml
```

### With Docker

```bash
# Build images
cd docker && bash scripts/build.sh

# Run training
bash scripts/run_train.sh examples/wheel_loader/config.yaml
```

## Model Architecture

```
Images + Instructions → VLM (frozen) → Features
                                         ↓
State → State Encoder ----------------→ Concat → Diffusion Head → Actions
```

## Data Format

Expected data format (LeRobot v3):
- Images: Multiple camera views
- State: 22D wheel loader state vector
- Actions: 6D control commands (throttle, steering, boom, bucket, brake, gear)
- Instructions: Text descriptions of tasks

## Expected Results

After 50K training steps:
- Action prediction loss should converge
- Model should learn basic wheel loader behaviors
- Can be deployed for inference via REST/gRPC API
