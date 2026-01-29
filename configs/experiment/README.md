# Experiment Configuration Directory

This directory contains complete experiment configurations that combine all components: model, robot, dataset, and training settings.

## Purpose

Experiment configs are the main entry point for running training or evaluation. They compose all the individual component configs into a complete specification.

## Configuration Structure

Each experiment config should include:

```yaml
# Experiment metadata
name: "wheel_loader_groot_vla"
description: "Train GROOT-style VLA for wheel loader material handling"
tags: ["wheel_loader", "construction", "diffusion_policy"]

# Component configurations (paths to individual configs)
model: "configs/model/framework/groot_style.yaml"
robot: "configs/robot/wheel_loader.yaml"
dataset: "configs/dataset/wheel_loader_demos.yaml"
training: "configs/training/default.yaml"

# Overrides for specific components (optional)
overrides:
  model:
    vlm:
      freeze: true
  training:
    batch_size: 8  # Override default batch size
    lr: 5e-5

# Evaluation settings
evaluation:
  checkpoint: "best"  # Which checkpoint to use for evaluation
  environments: ["sim", "real"]  # Which environments to evaluate on
  num_episodes: 100
  
# Experiment-specific parameters
seed: 42
experiment_dir: "experiments/wheel_loader_groot"
```

## Usage

Run experiments using:
```bash
python scripts/train.py --config configs/experiment/my_experiment.yaml
python scripts/eval.py --config configs/experiment/my_experiment.yaml
```

## Best Practices

1. **Use descriptive names**: Name experiments clearly (e.g., `wheel_loader_groot_diffusion_v1`)
2. **Version experiments**: Keep track of different versions and iterations
3. **Document changes**: Use git to track config changes
4. **Override sparingly**: Prefer editing component configs over too many overrides
5. **Test on small scale**: Start with smaller models/datasets to verify configs work
