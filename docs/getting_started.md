# Getting Started with LibroBot VLA

Welcome to LibroBot VLA! This guide will help you get up and running with the framework in minutes.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Your First Training Run](#your-first-training-run)
- [Basic Concepts](#basic-concepts)
- [Next Steps](#next-steps)
- [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

Before installing LibroBot VLA, ensure you have:

- **Python 3.12** (required)
- **PyTorch 2.9+** with CUDA support (for GPU training)
- **CUDA 13.0+** (if using NVIDIA GPUs)
- **16GB+ RAM** (32GB+ recommended)
- **NVIDIA GPU with 8GB+ VRAM** (16GB+ recommended)

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/librobot_vla.git
cd librobot_vla

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install basic dependencies
pip install -e .
```

### Installation with Optional Features

LibroBot VLA offers several optional feature sets:

```bash
# For training (includes distributed training, optimization tools)
pip install -e ".[train]"

# For inference and deployment (includes FastAPI, gRPC, ONNX)
pip install -e ".[inference]"

# For data processing (includes dataset loaders, augmentation)
pip install -e ".[data]"

# For robot interfaces (includes simulation environments)
pip install -e ".[robots]"

# For ROS2 integration
pip install -e ".[ros]"

# For development (includes testing, linting, formatting)
pip install -e ".[dev]"

# Install everything
pip install -e ".[all]"
```

### Verify Installation

```bash
# Check if LibroBot is installed correctly
python -c "import librobot; print(f'LibroBot version: {librobot.__version__}')"

# Check available models
python -c "from librobot.models import list_vlms, list_vlas; print('VLMs:', list_vlms()[:5]); print('VLAs:', list_vlas())"

# Test GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### Docker Installation (Recommended for Production)

For a reproducible environment, use Docker:

```bash
# Build the base image
docker build -t librobot-base -f docker/Dockerfile.base .

# Build the training image
docker build -t librobot-train -f docker/Dockerfile.train .

# Run with GPU support
docker run --gpus all -it -v $(pwd):/workspace librobot-train bash
```

For more details, see the [Deployment Guide](deployment.md).

## Quick Start

### 1. Create Your First VLA Model

```python
from librobot.models.vlm import create_vlm
from librobot.models.frameworks import create_vla
import torch

# Create a Vision-Language Model (VLM) backbone
vlm = create_vlm(
    "qwen2-vl-2b",        # Model name
    pretrained=True,       # Use pretrained weights
)

# Create a Vision-Language-Action (VLA) framework
vla = create_vla(
    "groot",               # Framework: GROOT (diffusion-based)
    vlm=vlm,               # VLM backbone
    action_dim=7,          # Robot action dimension (e.g., 6 DOF + gripper)
    state_dim=14,          # Proprioception dimension (e.g., joint pos + vel)
)

print(f"Model created with {vla.get_num_parameters():,} parameters")
print(f"Trainable parameters: {vla.get_num_parameters(trainable_only=True):,}")
```

### 2. Run Inference

```python
# Prepare inputs
images = torch.randn(2, 3, 224, 224)  # Batch of 2 images
text = ["pick up the red cup", "place the cup on the shelf"]
proprioception = torch.randn(2, 14)   # Current robot state

# Predict actions
with torch.no_grad():
    predicted_actions = vla.predict_action(
        images=images,
        text=text,
        proprioception=proprioception
    )

print(f"Predicted actions shape: {predicted_actions.shape}")
# Output: torch.Size([2, 7])
```

### 3. Training Loop (Minimal Example)

```python
from torch.utils.data import DataLoader

# Assuming you have a dataset (see Data section below)
from librobot.data.datasets import create_dataset

# Create dataset and dataloader
dataset = create_dataset(
    "rlds",                          # Dataset format
    path="/path/to/your/data",       # Data directory
    split="train"
)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Setup optimizer
optimizer = torch.optim.AdamW(vla.parameters(), lr=1e-4)

# Training loop
vla.train()
for epoch in range(10):
    for batch_idx, batch in enumerate(dataloader):
        # Forward pass
        outputs = vla(
            images=batch["images"],
            text=batch["instructions"],
            proprioception=batch["proprioception"],
            actions=batch["actions"]
        )
        
        # Backward pass
        loss = outputs["loss"]
        loss.backward()
        
        # Update weights
        optimizer.step()
        optimizer.zero_grad()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
```

## Your First Training Run

Let's train a complete model using the built-in training script and configuration system.

### Step 1: Prepare Your Data

LibroBot VLA supports multiple dataset formats. For this example, we'll use a simple format:

```python
# Create a minimal dataset for testing
import numpy as np
import os
from pathlib import Path

data_dir = Path("./demo_data/train")
data_dir.mkdir(parents=True, exist_ok=True)

# Create 100 demo trajectories
for i in range(100):
    episode_dir = data_dir / f"episode_{i:04d}"
    episode_dir.mkdir(exist_ok=True)
    
    # Save dummy data (replace with your real robot data)
    np.save(episode_dir / "images.npy", np.random.rand(50, 224, 224, 3))
    np.save(episode_dir / "actions.npy", np.random.rand(50, 7))
    np.save(episode_dir / "proprioception.npy", np.random.rand(50, 14))
    with open(episode_dir / "instruction.txt", "w") as f:
        f.write("pick up the object")

print("Demo data created!")
```

### Step 2: Create a Configuration File

Create a file `configs/my_first_experiment.yaml`:

```yaml
# Model configuration
model:
  framework: groot                  # Use GROOT framework
  vlm:
    name: qwen2-vl-2b
    pretrained: true
    freeze: true                    # Freeze VLM weights
  action_dim: 7
  state_dim: 14

# Data configuration
data:
  dataset_name: rlds
  data_dir: ./demo_data
  batch_size: 16
  num_workers: 4

# Training configuration
training:
  num_epochs: 10
  learning_rate: 1e-4
  weight_decay: 1e-4
  gradient_clip_norm: 1.0
  
  # Logging
  log_interval: 10
  eval_interval: 100
  save_interval: 500

# Logging backends
logging:
  wandb:
    enabled: true
    project: my-first-vla
    entity: null                    # Your wandb username

# Hardware
device: cuda
mixed_precision: true
```

### Step 3: Run Training

```bash
# Basic training
python scripts/train.py --config configs/my_first_experiment.yaml

# With command-line overrides
python scripts/train.py \
    --config configs/my_first_experiment.yaml \
    --override training.num_epochs=20 model.vlm.name=qwen2-vl-7b

# Distributed training on 4 GPUs
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/my_first_experiment.yaml
```

### Step 4: Monitor Training

Training progress can be monitored through:

1. **Console Output**: Real-time loss and metrics
2. **Weights & Biases**: If enabled, open the link printed in console
3. **TensorBoard**: Run `tensorboard --logdir checkpoints/`
4. **Checkpoints**: Saved in `checkpoints/` directory

### Step 5: Evaluate Your Model

```bash
# Evaluate on test set
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/my_first_experiment.yaml \
    --split test

# Run inference on a single example
python scripts/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --image path/to/image.jpg \
    --text "pick up the red cup" \
    --output predictions.json
```

## Basic Concepts

### 1. VLM (Vision-Language Model)

The VLM is the **backbone** that processes visual and language inputs into joint embeddings.

**Available VLMs:**
- **Qwen2-VL** (2B, 7B): High-quality multimodal understanding
- **Qwen3-VL** (4B, 7B): Latest version, best performance
- **Florence-2** (230M, 770M): Lightweight and efficient
- **PaliGemma** (3B): Balanced performance
- **InternVL2** (2B, 8B): Multilingual support
- **LLaVA** (7B, 13B): Strong baseline

```python
from librobot.models.vlm import create_vlm

# Create a VLM
vlm = create_vlm("qwen2-vl-2b", pretrained=True)

# Use it directly
embeddings = vlm.encode_image(images)
text_embeddings = vlm.encode_text(["pick up the cup"])
```

### 2. VLA Framework (Vision-Language-Action)

The VLA framework **combines** the VLM with action prediction mechanisms.

**Available Frameworks:**
- **GROOT**: Diffusion-based, stable training, multi-camera support
- **π0 (Pi-Zero)**: Flow matching, complex action spaces
- **Octo**: Unified transformer, multi-task learning
- **OpenVLA**: VLM fine-tuning, language-conditioned
- **RT-2**: Tokenized actions, discrete control
- **ACT**: Action chunking, bi-manual manipulation
- **Helix**: Hierarchical, long-horizon tasks

```python
from librobot.models.frameworks import create_vla

# Create a VLA framework
vla = create_vla(
    "groot",              # Framework choice
    vlm=vlm,              # VLM backbone
    action_dim=7,         # Action space dimension
    state_dim=14,         # Proprioception dimension
)
```

### 3. Action Heads

Action heads define **how** actions are predicted from multimodal features.

**Types:**
- **MLP**: Direct regression, fast inference
- **Transformer**: Sequential modeling, temporal dependencies
- **Diffusion**: DDPM/DDIM, multimodal distributions
- **Flow Matching**: Continuous normalizing flows
- **Hybrid**: Combinations of the above

Action heads are typically configured within the framework, but can be customized:

```python
from librobot.models.action_heads import create_action_head

action_head = create_action_head(
    "diffusion",
    action_dim=7,
    hidden_dim=512,
    num_diffusion_steps=100
)
```

### 4. Registry System

LibroBot uses a **registry pattern** for dynamic component discovery:

```python
# All components are registered automatically
from librobot.models.vlm import list_vlms, get_vlm_info

# List all available VLMs
print(list_vlms())

# Get info about a specific VLM
info = get_vlm_info("qwen2-vl-2b")
print(info)

# Create from registry
vlm = create_vlm("qwen2-vl-2b")  # Works automatically!
```

### 5. Configuration System

LibroBot uses **YAML configuration files** with Hydra for reproducible experiments.

**Key features:**
- **Hierarchical configs**: Organize by model/data/training
- **Composition**: Combine multiple config files
- **Overrides**: Command-line parameter overrides
- **Interpolation**: Reference other config values

See [Configuration Guide](configuration.md) for details.

### 6. Data Pipeline

LibroBot provides flexible data loading:

```python
from librobot.data.datasets import create_dataset

dataset = create_dataset(
    "rlds",                      # Format: rlds, lerobot, hdf5, zarr
    path="/path/to/data",
    split="train",
    transform=None,              # Optional transforms
    cache_in_memory=False        # Cache for faster loading
)
```

**Supported formats:**
- **RLDS**: TensorFlow Datasets format
- **LeRobot**: HuggingFace datasets format
- **HDF5**: Hierarchical Data Format
- **Zarr**: Cloud-optimized arrays
- **Custom**: Implement your own

## Next Steps

Now that you have the basics, explore more advanced topics:

1. **[Configuration Guide](configuration.md)**: Master the configuration system
2. **[Architecture Overview](architecture.md)**: Understand the framework design
3. **[Adding Robots](adding_robots.md)**: Integrate your robot
4. **[Adding Models](adding_models.md)**: Extend with custom models
5. **[Deployment Guide](deployment.md)**: Deploy to production

### Example Projects

Check out complete examples in the `examples/` directory:

```bash
# VLM demos
python examples/vlm_demo.py

# Framework comparisons
python examples/frameworks/compare_frameworks.py

# Custom components
python examples/frameworks/custom_framework.py

# Robot-specific examples
python examples/wheel_loader/train_wheel_loader.py
```

### Join the Community

- **GitHub Discussions**: Ask questions, share projects
- **Issues**: Report bugs, request features
- **Contributing**: See [COMPONENT_GUIDE](../docs/design/COMPONENT_GUIDE.md)

## Troubleshooting

### Installation Issues

**Problem**: `ImportError: No module named 'librobot'`

**Solution**: Ensure you installed in editable mode:
```bash
pip install -e .
```

**Problem**: CUDA out of memory during training

**Solutions**:
1. Reduce batch size: `--override training.batch_size=8`
2. Enable gradient checkpointing: `--override model.gradient_checkpointing=true`
3. Use smaller VLM: `--override model.vlm.name=qwen2-vl-2b`
4. Freeze VLM weights: `--override model.vlm.freeze=true`

### Training Issues

**Problem**: Loss is NaN or Inf

**Solutions**:
1. Check your data normalization
2. Reduce learning rate: `--override training.learning_rate=1e-5`
3. Enable gradient clipping: `--override training.gradient_clip_norm=1.0`
4. Use mixed precision: `--override training.mixed_precision=true`

**Problem**: Training is very slow

**Solutions**:
1. Increase batch size: `--override training.batch_size=32`
2. Enable gradient accumulation: `--override training.gradient_accumulation_steps=4`
3. Use more workers: `--override data.num_workers=8`
4. Enable distributed training (multi-GPU)
5. Use Flash Attention 2: `pip install flash-attn`

**Problem**: Model doesn't learn anything

**Checklist**:
1. Verify data loading: Check batch shapes and values
2. Check label format: Ensure actions are normalized
3. Verify VLM is frozen: `model.vlm.freeze=true` (faster training)
4. Check learning rate: Try 1e-4 to 1e-3
5. Verify loss computation: Print intermediate losses

### Model Loading Issues

**Problem**: Cannot load pretrained VLM weights

**Solution**: Download weights manually:
```bash
# For HuggingFace models
huggingface-cli login
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct
```

**Problem**: Checkpoint incompatibility

**Solution**: Use strict=False when loading:
```python
vla.load_state_dict(checkpoint["model"], strict=False)
```

### Runtime Issues

**Problem**: Prediction is very slow

**Solutions**:
1. Export to ONNX: See [Deployment Guide](deployment.md)
2. Use TensorRT optimization
3. Enable torch.compile(): `model = torch.compile(model)`
4. Reduce diffusion steps: `num_inference_steps=5`

**Problem**: Memory leak during inference

**Solution**: Use context managers:
```python
with torch.no_grad(), torch.cuda.amp.autocast():
    actions = vla.predict_action(images, text, proprio)
```

### Getting Help

If you're still stuck:

1. **Check Documentation**: Browse `docs/` for detailed guides
2. **Search Issues**: Someone may have faced the same problem
3. **Ask on Discussions**: Community members can help
4. **Open an Issue**: For bugs or feature requests

**When asking for help, include:**
- LibroBot version: `python -c "import librobot; print(librobot.__version__)"`
- PyTorch version: `python -c "import torch; print(torch.__version__)"`
- CUDA version: `python -c "import torch; print(torch.version.cuda)"`
- Full error message and traceback
- Minimal code to reproduce the issue
- Configuration file (if applicable)

---

**Ready to build amazing robot learning systems? Let's get started! 🚀**

For more advanced topics, continue to the [Configuration Guide](configuration.md).
