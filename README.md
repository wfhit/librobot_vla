# LibroBot VLA Framework

A comprehensive, extensible Vision-Language-Action (VLA) framework for robotics that supports ANY robot, ALL major VLA architectures, and production deployment.

## Features

- **Universal Robot Support**: Arms, mobile robots, humanoids, vehicles, custom configurations
- **Multiple VLA Architectures**: GR00T, π0, OpenVLA, Octo, ACT styles
- **Flexible Action Heads**: Diffusion, Flow Matching, MLP OFT, Transformer ACT, Autoregressive
- **Production Ready**: Docker, optimization, REST/gRPC APIs
- **Research Friendly**: Registry pattern, config-driven, plugin architecture

## Quick Start

### Installation

```bash
# Basic installation
pip install -e .

# With training dependencies
pip install -e ".[train]"

# With inference dependencies
pip install -e ".[inference]"

# All dependencies
pip install -e ".[all]"
```

### Training Example

```python
from librobot import VLA, Robot
from librobot.utils import load_config

# Load configuration
config = load_config("configs/experiment/wheel_loader_groot.yaml")

# Or build programmatically
from librobot.robots import WheelLoaderRobot
from librobot.models.frameworks import GR00TStyleFramework
from librobot.models.action_heads import DiffusionTransformerHead
from librobot.models.encoders import MLPEncoder

robot = WheelLoaderRobot()
# Build and train your model...
```

### Using the CLI

```bash
# Training
python scripts/train.py --config configs/experiment/wheel_loader_groot.yaml

# Evaluation
python scripts/evaluate.py --config CONFIG --checkpoint CKPT

# Inference
python scripts/inference.py --checkpoint CKPT
```

## Project Structure

```
librobot/
├── docker/              # Docker configurations
├── configs/             # YAML configurations
│   ├── model/          # Model configs (VLM, action heads, encoders)
│   ├── robot/          # Robot definitions
│   ├── training/       # Training configs
│   └── experiment/     # Full experiment configs
├── librobot/
│   ├── models/         # Model implementations
│   │   ├── vlm/       # Vision-Language Models
│   │   ├── action_heads/  # Action prediction heads
│   │   ├── encoders/      # State/history encoders
│   │   └── frameworks/    # VLA frameworks
│   ├── data/          # Data pipeline
│   ├── training/      # Training system
│   ├── inference/     # Inference & deployment
│   ├── robots/        # Robot definitions
│   └── utils/         # Utilities
├── scripts/           # Training/eval scripts
├── examples/          # Example configurations
└── tests/            # Unit and integration tests
```

## Supported Components

### VLA Frameworks

- **GR00T-style**: VLM (frozen) + state encoder (bypass VLM) → action head
- **π0-style**: State tokenized into VLM + flow matching
- **Octo-style**: Unified transformer with readout tokens
- **OpenVLA-style**: VLM + discrete action tokens
- **ACT-style**: Transformer encoder-decoder

### Action Heads

- **Diffusion**: DDPM/DDIM with Transformer/UNet denoiser
- **Flow Matching**: Optimal transport, rectified flow
- **MLP OFT**: Parallel action regression
- **Transformer ACT**: Action chunking transformer
- **Autoregressive FAST**: Discrete action tokens

### Robots

- **Wheel Loader**: 6D action (throttle, steering, boom, bucket, brake, gear), 22D state
- **SO100 Arm**: 6D joints, 12D state
- **Custom**: Define your own via config

## Docker

```bash
# Build images
cd docker && bash scripts/build.sh

# Run training
bash scripts/run_train.sh configs/experiment/wheel_loader_groot.yaml

# Run inference server
bash scripts/run_server.sh
```

## Configuration

The framework uses OmegaConf for hierarchical configuration:

```yaml
# configs/experiment/my_experiment.yaml
model:
  framework:
    name: groot_style
  vlm:
    name: qwen2_vl_7b
    freeze: true
  action_head:
    name: diffusion_transformer
    hidden_dim: 512

robot:
  name: wheel_loader

training:
  batch_size: 8
  max_steps: 50000
```

Override from CLI:

```bash
python scripts/train.py --config CONFIG --overrides model.lr=1e-3 batch_size=16
```

## Registry System

All components use a registry pattern for easy discovery and extension:

```python
from librobot.utils.registry import register_action_head
from librobot.models.action_heads import BaseActionHead

@register_action_head("my_head")
class MyActionHead(BaseActionHead):
    def forward(self, features, actions=None):
        # Your implementation
        pass
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Format code
make format

# Run linters
make lint

# Run tests
make test
```

## Examples

See `examples/` directory for complete examples:

- `examples/wheel_loader/`: Wheel loader with GR00T-style framework

## Architecture

### GR00T-Style Framework

```
┌─────────────────────────────────────────────────────┐
│                    Input Layer                       │
│  ┌─────────────┐  ┌──────────┐  ┌────────────────┐ │
│  │   Images    │  │  State   │  │  Instruction   │ │
│  └──────┬──────┘  └─────┬────┘  └────────┬───────┘ │
└─────────┼───────────────┼────────────────┼─────────┘
          │               │                │
          │               │                │
     ┌────▼────┐     ┌────▼────┐         │
     │   VLM   │◄────┤  State  │         │
     │(Frozen) │     │ Encoder │         │
     └────┬────┘     └────┬────┘         │
          │               │               │
          └───────┬───────┘               │
                  │                       │
            ┌─────▼──────┐               │
            │   Fusion   │◄──────────────┘
            └─────┬──────┘
                  │
            ┌─────▼──────────┐
            │  Action Head   │
            │  (Diffusion/   │
            │   MLP/etc.)    │
            └─────┬──────────┘
                  │
            ┌─────▼──────┐
            │  Actions   │
            └────────────┘
```

## Citation

If you use LibroBot in your research, please cite:

```bibtex
@software{librobot2024,
  title={LibroBot: A Comprehensive VLA Framework for Robotics},
  author={LibroBot Contributors},
  year={2024},
  url={https://github.com/wfhit/librobot_vla}
}
```

## License

Apache License 2.0 - see LICENSE file for details.

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Acknowledgments

This framework is inspired by and builds upon:
- GR00T (NVIDIA)
- π0 (Physical Intelligence)
- OpenVLA (Stanford)
- Octo (UC Berkeley)
- ACT (Tony Zhao et al.)