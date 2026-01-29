# Dataset Configuration Directory

This directory contains configuration files for different datasets used to train VLA models.

## Purpose

Dataset configs define:
- Data source locations (local paths, URLs, HuggingFace datasets)
- Data format and structure
- Preprocessing and augmentation pipelines
- Train/val/test splits
- Batching and sampling strategies

## Configuration Structure

Each dataset config should include:

```yaml
# Dataset identifier
name: "bridge_v2"

# Data source
source:
  type: "rlds"  # Options: "rlds", "hdf5", "zarr", "huggingface"
  path: "/data/bridge_v2"
  version: "1.0.0"

# Data splits
splits:
  train: 0.9
  val: 0.1
  test: null  # Optional test split

# Observation configuration
observations:
  images: ["wrist_image", "base_image"]
  proprio: ["joint_positions", "joint_velocities", "gripper_position"]
  
# Action configuration
actions:
  type: "continuous"
  dim: 7
  normalization: "bounds"

# Preprocessing
preprocessing:
  image_size: [224, 224]
  normalize_images: true
  normalize_actions: true
  
# Data augmentation (training only)
augmentation:
  random_crop: true
  color_jitter: 0.1
  random_flip: false

# Batching
batch_size: 32
shuffle: true
drop_last: true
```

## Usage

Reference dataset configs in experiment configs:
```yaml
dataset: "configs/dataset/bridge_v2.yaml"
```
