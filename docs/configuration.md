# Configuration System Guide

LibroBot VLA uses a powerful YAML-based configuration system built on [Hydra](https://hydra.cc/) and [OmegaConf](https://omegaconf.readthedocs.io/). This guide covers everything you need to know about configuring your experiments.

## Table of Contents

- [Overview](#overview)
- [Configuration Structure](#configuration-structure)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [Environment Variables](#environment-variables)
- [Best Practices](#best-practices)
- [Common Patterns](#common-patterns)
- [Troubleshooting](#troubleshooting)

## Overview

### Why YAML Configuration?

- **Reproducibility**: Every experiment is fully specified in a file
- **Version Control**: Track configuration changes with git
- **Experimentation**: Easily try different hyperparameters
- **Composition**: Build complex configs from reusable components
- **Documentation**: Self-documenting experiment setup

### Configuration Hierarchy

LibroBot uses a hierarchical configuration system:

```
configs/
├── defaults.yaml              # Global defaults
├── model/                     # Model configurations
│   ├── vlm/                   # VLM backbone configs
│   ├── framework/             # VLA framework configs
│   └── action_head/           # Action head configs
├── data/                      # Dataset configurations
├── training/                  # Training configurations
├── robot/                     # Robot-specific configs
└── experiment/                # Complete experiment configs
```

### How It Works

1. **Start with defaults**: `configs/defaults.yaml` provides sensible defaults
2. **Compose components**: Combine model, data, and training configs
3. **Override values**: Use CLI arguments or environment variables
4. **Merge hierarchically**: Child configs override parent values

## Configuration Structure

### defaults.yaml - Global Settings

The root configuration file with framework-wide defaults:

```yaml
# configs/defaults.yaml

# Random seed for reproducibility
seed: 42

# Device configuration
device: "cuda"
mixed_precision: true

# Logging configuration
logging:
  wandb:
    enabled: true
    project: "librobot-vla"
    entity: null
  log_interval: 10
  save_interval: 1000
  eval_interval: 500

# Checkpoint configuration
checkpoint:
  save_dir: "checkpoints"
  keep_last_n: 3
  save_best: true
  resume_from: null

# Data loading
dataloader:
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2

# Optimization
optimization:
  gradient_clip_norm: 1.0
  gradient_accumulation_steps: 1
  warmup_steps: 1000

# Evaluation
evaluation:
  num_episodes: 50
  save_videos: true
  video_fps: 10

# Model defaults
model:
  use_pretrained: true
  freeze_backbone: false

# Action space
action:
  normalization: "bounds"
  clip_actions: true

# Vision preprocessing
vision:
  image_size: [224, 224]
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
```

### Model Configuration

#### VLM Configuration

```yaml
# configs/model/vlm/qwen2_vl_2b.yaml

name: "qwen2_vl_2b"
pretrained: true
pretrained_path: "Qwen/Qwen2-VL-2B-Instruct"
freeze: true

# Model architecture
hidden_dim: 1536
num_layers: 28
num_attention_heads: 12

# Feature extraction
extract_layer: -2
pool_method: "mean"  # Options: "mean", "last", "attention"

# Memory optimization
gradient_checkpointing: true
use_flash_attention: true

# Quantization (optional)
quantization:
  enabled: false
  bits: 8  # 4 or 8
  method: "bnb"  # "bnb" or "gptq"
```

#### Framework Configuration

```yaml
# configs/model/framework/groot_style.yaml

type: "groot_style"

# VLM configuration
vlm:
  name: "qwen2_vl_2b"
  pretrained: true
  freeze: true

# Proprioceptive encoder
encoder:
  type: "mlp"
  input_dim: 14
  hidden_dims: [256, 128]
  output_dim: 128
  activation: "relu"
  dropout: 0.1
  normalize_input: true

# Action head
action_head:
  type: "diffusion"
  action_dim: 7
  action_horizon: 16
  chunk_size: 8
  num_diffusion_steps: 100
  num_inference_steps: 10
  noise_schedule: "cosine"
  
  backbone:
    type: "temporal_unet"
    hidden_dims: [512, 256, 128]
    kernel_size: 5
    num_groups: 8

# Feature fusion
fusion:
  method: "cross_attention"
  num_attention_heads: 8
  vision_proj_dim: 512
  lang_proj_dim: 512
  proprio_proj_dim: 128
  fused_dim: 512

# Observation preprocessing
observation:
  image_keys: ["wrist_image", "base_image"]
  image_size: [224, 224]
  normalize: true
  augmentation:
    enabled: true
    random_crop: true
    color_jitter: 0.1
  proprio_keys: ["joint_positions", "joint_velocities"]
  normalize_proprio: true

# Language preprocessing
language:
  max_length: 77
  tokenizer: "qwen2_vl"
  padding: "max_length"

# Training strategy
training:
  freeze_vlm: true
  freeze_encoder: false
  freeze_action_head: false
  gradient_checkpointing: true
  loss_weights:
    action_loss: 1.0
    auxiliary_losses: 0.1
```

### Data Configuration

```yaml
# configs/data/rlds_dataset.yaml

# Dataset format
format: "rlds"
dataset_name: "fractal20220817_data"

# Paths
data_dir: "/path/to/data"
cache_dir: "${env:HOME}/.cache/librobot"

# Splits
train_split: "train[:90%]"
val_split: "train[90%:]"
test_split: "test"

# Data loading
batch_size: 32
shuffle: true
num_workers: 4
pin_memory: true
drop_last: true

# Preprocessing
preprocessing:
  resize_images: true
  target_size: [224, 224]
  normalize_images: true
  normalize_actions: true
  normalize_proprio: true
  
  # Action normalization
  action_bounds:
    min: [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]
    max: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

# Data augmentation
augmentation:
  enabled: true
  
  # Image augmentations
  random_crop: true
  crop_scale: [0.8, 1.0]
  random_flip: false
  color_jitter:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1
  random_rotation: 5  # degrees
  
  # Action augmentations
  action_noise: 0.01

# Filtering
filtering:
  min_episode_length: 10
  max_episode_length: 1000
  success_only: false

# Memory management
cache_in_memory: false
persistent_workers: true
prefetch_factor: 2
```

### Training Configuration

```yaml
# configs/training/default.yaml

# Duration
num_epochs: 50
max_steps: null

# Batch configuration
batch_size: 16
gradient_accumulation_steps: 2
eval_batch_size: 32

# Optimizer
optimizer:
  type: "adamw"
  lr: 1e-4
  weight_decay: 1e-4
  betas: [0.9, 0.999]
  eps: 1e-8
  fused: true

# Learning rate schedule
lr_schedule:
  type: "cosine_with_warmup"
  warmup_steps: 1000
  warmup_ratio: 0.1
  min_lr: 1e-6
  num_cycles: 0.5

# Regularization
regularization:
  gradient_clip_norm: 1.0
  dropout: 0.1
  attention_dropout: 0.1
  label_smoothing: 0.0

# Mixed precision
mixed_precision:
  enabled: true
  dtype: "fp16"  # or "bf16"
  opt_level: "O1"
  loss_scale: "dynamic"

# Stability
stability:
  detect_anomaly: false
  skip_nan_gradients: true
  max_loss_value: 1e4

# Checkpointing
checkpoint:
  save_interval: 2000
  save_interval_epochs: 1
  keep_last_n: 3
  save_best: true
  best_metric: "val/success_rate"
  best_mode: "max"
  save_optimizer: true

# Evaluation
evaluation:
  interval: 1000
  interval_epochs: 1
  num_eval_episodes: 50
  save_eval_videos: true
  video_fps: 10
  metrics:
    - "success_rate"
    - "average_return"
    - "episode_length"
    - "action_mse"

# Logging
logging:
  log_interval: 10
  log_gradients: false
  log_weights: false
  log_learning_rate: true
  log_loss_components: true
  histogram_interval: 500
  histogram_enabled: false

# Distributed training
distributed:
  enabled: false
  backend: "nccl"
  find_unused_parameters: false
  gradient_as_bucket_view: true
  use_fsdp: false

# Reproducibility
seed: 42
deterministic: false
benchmark: true

# Early stopping
early_stopping:
  enabled: false
  patience: 10
  min_delta: 0.001
  metric: "val/success_rate"
  mode: "max"
```

### Robot Configuration

```yaml
# configs/robot/wheel_loader.yaml

name: "wheel_loader"
type: "custom"

# Action space
action_space:
  type: "continuous"
  dim: 7
  bounds:
    min: [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]
    max: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  names: ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]

# Observation space
observation_space:
  # Images
  images:
    wrist_camera:
      shape: [224, 224, 3]
      dtype: "uint8"
    base_camera:
      shape: [224, 224, 3]
      dtype: "uint8"
  
  # Proprioception
  proprio:
    joint_positions:
      dim: 7
      bounds: [-3.14, 3.14]
    joint_velocities:
      dim: 7
      bounds: [-1.0, 1.0]

# Control configuration
control:
  frequency: 10  # Hz
  action_repeat: 1
  action_scaling: 1.0

# Safety limits
safety:
  velocity_limits: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5]
  acceleration_limits: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0]
  emergency_stop_enabled: true
```

## Basic Usage

### 1. Using Default Configuration

```bash
# Use all defaults
python scripts/train.py

# Equivalent to:
python scripts/train.py --config configs/defaults.yaml
```

### 2. Using a Complete Experiment Config

```bash
# Use a pre-configured experiment
python scripts/train.py --config configs/experiment/groot_wheel_loader.yaml
```

Example experiment config:

```yaml
# configs/experiment/groot_wheel_loader.yaml

# Inherit from defaults
defaults:
  - /defaults
  - /model/framework/groot_style
  - /data/rlds_dataset
  - /training/default
  - /robot/wheel_loader

# Override specific values
model:
  vlm:
    name: "qwen2_vl_2b"
    freeze: true
  action_dim: ${robot.action_space.dim}

data:
  data_dir: "/data/wheel_loader/demonstrations"
  batch_size: 32

training:
  num_epochs: 100
  learning_rate: 1e-4

logging:
  wandb:
    enabled: true
    project: "wheel-loader-vla"
    tags: ["groot", "qwen2-vl", "wheel-loader"]
```

### 3. Command-Line Overrides

Override any configuration value from the command line:

```bash
# Single override
python scripts/train.py --override training.learning_rate=1e-3

# Multiple overrides
python scripts/train.py \
    --override training.learning_rate=1e-3 \
    --override training.batch_size=64 \
    --override model.vlm.name=qwen2-vl-7b

# Nested overrides
python scripts/train.py \
    --override model.action_head.num_diffusion_steps=50 \
    --override logging.wandb.project=my-experiment
```

### 4. Using in Python

```python
from librobot.utils.config import Config

# Load configuration
config = Config.load("configs/experiment/my_experiment.yaml")

# Access values
print(config.model.vlm.name)
print(config.training.learning_rate)

# Override programmatically
config.training.batch_size = 32
config.model.vlm.freeze = True

# Save configuration
config.save("configs/experiment/modified_config.yaml")
```

## Advanced Features

### 1. Config Composition

Combine multiple config files:

```yaml
# configs/experiment/my_experiment.yaml

defaults:
  - /defaults                        # Global defaults
  - /model/framework/groot_style     # Model architecture
  - /data/rlds_dataset              # Dataset configuration
  - /training/default               # Training setup
  - _self_                          # This file (last to take precedence)

# Override inherited values
model:
  vlm:
    name: "qwen3-vl-4b"  # Override from groot_style.yaml

data:
  batch_size: 64  # Override from rlds_dataset.yaml
```

### 2. Variable Interpolation

Reference other config values:

```yaml
model:
  action_dim: 7
  state_dim: 14
  
# Reference other values
action_head:
  input_dim: ${model.action_dim}
  
encoder:
  output_dim: ${model.state_dim}

# Arithmetic operations
training:
  batch_size: 16
  gradient_accumulation_steps: 2
  effective_batch_size: ${training.batch_size} * ${training.gradient_accumulation_steps}
```

### 3. Environment Variables

Access environment variables in configs:

```yaml
# Using environment variables
data:
  data_dir: "${env:DATA_ROOT}/robot_demos"
  cache_dir: "${env:HOME}/.cache/librobot"

logging:
  wandb:
    entity: "${env:WANDB_ENTITY}"
    api_key: "${env:WANDB_API_KEY}"

# With defaults
paths:
  output_dir: "${env:OUTPUT_DIR, checkpoints}"  # Use OUTPUT_DIR or default to "checkpoints"
```

Set environment variables:

```bash
export DATA_ROOT=/mnt/data
export WANDB_ENTITY=my-username
export OUTPUT_DIR=/experiments/run_001

python scripts/train.py --config configs/experiment/my_experiment.yaml
```

### 4. Conditional Configuration

Use conditions based on other values:

```yaml
model:
  use_large_model: false
  
  vlm:
    # Conditional value
    name: ${if:${model.use_large_model},"qwen2-vl-7b","qwen2-vl-2b"}
    
training:
  # Adjust batch size based on model
  batch_size: ${if:${model.use_large_model},8,32}
```

### 5. List and Dict Merging

Merge lists and dictionaries:

```yaml
# Parent config
augmentation:
  transforms:
    - random_crop
    - color_jitter

# Child config (appends to list)
augmentation:
  transforms:
    - random_rotation
    - gaussian_noise

# Result: [random_crop, color_jitter, random_rotation, gaussian_noise]
```

### 6. Type Annotations

Ensure type safety:

```yaml
# With type annotations
training:
  learning_rate: 1e-4  # float
  batch_size: 32       # int
  use_amp: true        # bool
  tags:                # list
    - experiment_1
    - groot
  
# Validation happens automatically
# This would raise an error:
# training:
#   learning_rate: "not-a-number"  # TypeError!
```

## Environment Variables

### Standard Environment Variables

LibroBot recognizes these environment variables:

```bash
# Data paths
export LIBROBOT_DATA_DIR=/path/to/data
export LIBROBOT_CACHE_DIR=/path/to/cache

# Logging
export WANDB_API_KEY=your_key
export WANDB_ENTITY=your_username
export WANDB_PROJECT=my-project

# Hardware
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

# Debug mode
export LIBROBOT_DEBUG=1
export LIBROBOT_LOG_LEVEL=DEBUG

# Output
export LIBROBOT_OUTPUT_DIR=/experiments/run_001
```

### Using .env Files

Create a `.env` file:

```bash
# .env
DATA_ROOT=/mnt/data/robot_demos
WANDB_ENTITY=my-username
WANDB_PROJECT=robot-learning
OUTPUT_DIR=./outputs
CUDA_VISIBLE_DEVICES=0,1
```

Load automatically:

```python
from dotenv import load_dotenv
load_dotenv()

# Now environment variables are available
```

Or use with scripts:

```bash
# Load .env file
export $(cat .env | xargs)

# Run training
python scripts/train.py --config configs/experiment/my_experiment.yaml
```

## Best Practices

### 1. Organize Configs by Purpose

```
configs/
├── defaults.yaml              # Framework defaults
├── model/                     # Model architectures
│   ├── vlm/                   # VLM variants
│   ├── framework/             # VLA frameworks
│   └── action_head/           # Action heads
├── data/                      # Dataset configs
│   ├── rlds_dataset.yaml
│   ├── lerobot_dataset.yaml
│   └── custom_dataset.yaml
├── training/                  # Training strategies
│   ├── default.yaml
│   ├── distributed.yaml
│   └── low_resource.yaml
├── robot/                     # Robot-specific
│   ├── panda.yaml
│   ├── wheel_loader.yaml
│   └── ur5.yaml
└── experiment/                # Complete experiments
    ├── groot_panda.yaml
    ├── pi0_wheel_loader.yaml
    └── custom_experiment.yaml
```

### 2. Version Your Configs

```bash
# Track configs with git
git add configs/experiment/my_experiment.yaml
git commit -m "Add experiment config for groot + wheel loader"

# Tag experiments
git tag -a exp-001 -m "Baseline GROOT experiment"
```

### 3. Document Your Configs

```yaml
# Add comments to explain choices
model:
  vlm:
    # Using 2B model for faster iteration during development
    # Switch to 7B for final training runs
    name: "qwen2_vl_2b"
    
    # Freeze VLM to reduce memory and focus on action head training
    freeze: true

training:
  # Batch size tuned for 4x RTX 3090 GPUs
  batch_size: 16
  gradient_accumulation_steps: 2  # Effective batch = 32
```

### 4. Use Sensible Defaults

```yaml
# Start with working defaults
defaults:
  - /defaults

# Override only what's necessary
model:
  vlm:
    name: "qwen2_vl_2b"  # Change model

# Keep everything else as default
```

### 5. Create Experiment Templates

```yaml
# configs/experiment/_template.yaml
# Copy this file to create new experiments

defaults:
  - /defaults
  - /model/framework/groot_style
  - /data/rlds_dataset
  - /training/default

# Fill in these required values
data:
  data_dir: ???  # REQUIRED: Set your data path

logging:
  wandb:
    project: ???  # REQUIRED: Set your project name
```

### 6. Validate Configs

```python
# Validate before training
from librobot.utils.config import Config

config = Config.load("configs/experiment/my_experiment.yaml")

# Check for required fields
assert config.data.data_dir is not None, "data_dir must be set"
assert config.model.action_dim > 0, "action_dim must be positive"

# Validate paths exist
import os
assert os.path.exists(config.data.data_dir), f"Data dir not found: {config.data.data_dir}"
```

## Common Patterns

### Pattern 1: Multi-GPU Training

```yaml
# configs/training/distributed.yaml

distributed:
  enabled: true
  backend: "nccl"
  find_unused_parameters: false

# Adjust batch size for multi-GPU
training:
  batch_size: 8  # Per device
  # Total batch = 8 * num_gpus
```

```bash
# Run with torchrun
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/experiment/my_experiment.yaml \
    --override distributed.enabled=true
```

### Pattern 2: Low-Resource Training

```yaml
# configs/training/low_resource.yaml

model:
  vlm:
    freeze: true  # Don't train VLM
    quantization:
      enabled: true
      bits: 8

training:
  batch_size: 4
  gradient_accumulation_steps: 8
  mixed_precision:
    enabled: true
    dtype: "fp16"

dataloader:
  num_workers: 2
```

### Pattern 3: Hyperparameter Sweeps

```yaml
# configs/experiment/sweep.yaml

# Use with wandb sweeps
defaults:
  - /defaults
  - /model/framework/groot_style

# Sweep over these parameters
sweep:
  method: "bayes"
  metric:
    name: "val/success_rate"
    goal: "maximize"
  parameters:
    training.learning_rate:
      distribution: "log_uniform_values"
      min: 1e-5
      max: 1e-3
    training.weight_decay:
      values: [0, 1e-5, 1e-4, 1e-3]
    model.action_head.num_diffusion_steps:
      values: [50, 100, 200]
```

### Pattern 4: Stage-wise Training

```yaml
# Stage 1: Train action head only
# configs/training/stage1_action_head.yaml
model:
  vlm:
    freeze: true
  encoder:
    freeze: true
  action_head:
    freeze: false

training:
  num_epochs: 20
  learning_rate: 1e-3

---

# Stage 2: Fine-tune encoder
# configs/training/stage2_encoder.yaml
model:
  vlm:
    freeze: true
  encoder:
    freeze: false
  action_head:
    freeze: false

training:
  num_epochs: 30
  learning_rate: 1e-4
  resume_from: "checkpoints/stage1/best_model.pt"

---

# Stage 3: Fine-tune everything
# configs/training/stage3_full.yaml
model:
  vlm:
    freeze: false
  encoder:
    freeze: false
  action_head:
    freeze: false

training:
  num_epochs: 50
  learning_rate: 1e-5
  resume_from: "checkpoints/stage2/best_model.pt"
```

## Troubleshooting

### Issue 1: Config Not Found

**Error**: `FileNotFoundError: configs/my_config.yaml`

**Solutions**:
- Check file path is correct
- Use absolute path: `--config /full/path/to/config.yaml`
- Ensure you're in the repository root

### Issue 2: Override Not Working

**Error**: Override value is ignored

**Solutions**:
- Check override syntax: `--override training.lr=1e-4` (no spaces around `=`)
- Use dot notation for nested values: `model.vlm.name=qwen2-vl-7b`
- Quote strings with spaces: `logging.wandb.project="my project"`

### Issue 3: Environment Variable Not Resolved

**Error**: `${env:MY_VAR}` appears literally in config

**Solutions**:
- Ensure variable is exported: `export MY_VAR=value`
- Check variable name matches exactly (case-sensitive)
- Use default fallback: `${env:MY_VAR,default_value}`

### Issue 4: Config Validation Fails

**Error**: `ValidationError: Invalid value for field X`

**Solutions**:
- Check data types match (int, float, str, bool)
- Ensure required fields are set
- Validate against schema if available
- Check for typos in field names

### Issue 5: Circular Reference

**Error**: `ConfigAttributeError: Recursive interpolation`

**Solutions**:
- Check for circular references in interpolation
- Use `_self_` in defaults to control merge order
- Avoid referencing values that reference back

### Issue 6: Merge Conflict

**Error**: Values not merging as expected

**Solutions**:
- Use `_self_` in defaults list to control order
- Place `_self_` last to override inherited values
- Check list merge behavior (append vs replace)

## Advanced: Custom Config Loaders

For advanced users, create custom config loaders:

```python
from librobot.utils.config import Config, ConfigLoader

class CustomConfigLoader(ConfigLoader):
    """Custom configuration loader with validation."""
    
    def load(self, path: str) -> Config:
        config = super().load(path)
        
        # Custom validation
        self._validate_paths(config)
        self._validate_dimensions(config)
        
        return config
    
    def _validate_paths(self, config: Config):
        """Ensure all paths exist."""
        import os
        if config.data.data_dir:
            assert os.path.exists(config.data.data_dir), \
                f"Data directory not found: {config.data.data_dir}"
    
    def _validate_dimensions(self, config: Config):
        """Ensure dimension consistency."""
        assert config.model.action_dim == len(config.robot.action_space.bounds.min), \
            "Action dim mismatch with robot config"

# Use custom loader
loader = CustomConfigLoader()
config = loader.load("configs/experiment/my_experiment.yaml")
```

---

**Next Steps:**

- [Architecture Overview](architecture.md): Understand the system design
- [Getting Started](getting_started.md): Basic usage guide  
- [Adding Models](adding_models.md): Extend with custom components

For technical details on configuration internals, see [Design Principles](design/DESIGN_PRINCIPLES.md).
