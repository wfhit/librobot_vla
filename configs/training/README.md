# Training Configuration Directory

This directory contains configuration files for training hyperparameters and strategies.

## Purpose

Training configs define:
- Optimizer configuration (type, learning rate, weight decay)
- Learning rate schedule
- Training duration (epochs/steps)
- Batch size and gradient accumulation
- Regularization techniques
- Distributed training settings

## Configuration Structure

Each training config should include:

```yaml
# Training duration
num_epochs: 100
max_steps: 100000  # Alternative to num_epochs

# Batch size
batch_size: 32
gradient_accumulation_steps: 1
effective_batch_size: 32  # batch_size * gradient_accumulation_steps * num_gpus

# Optimizer
optimizer:
  type: "adamw"
  lr: 1e-4
  weight_decay: 1e-4
  betas: [0.9, 0.999]
  
# Learning rate schedule
lr_schedule:
  type: "cosine"  # Options: "constant", "linear", "cosine", "polynomial"
  warmup_steps: 1000
  min_lr: 1e-6

# Regularization
regularization:
  gradient_clip_norm: 1.0
  dropout: 0.1
  label_smoothing: 0.0

# Distributed training
distributed:
  enabled: false
  backend: "nccl"
  find_unused_parameters: false
```

## Usage

Reference training configs in experiment configs:
```yaml
training: "configs/training/default.yaml"
```
