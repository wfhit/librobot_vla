# Action Head Configuration Directory

This directory contains configuration files for action prediction heads used in the VLA framework.

## Supported Action Heads

The action head configurations should specify models registered in the action head factory:

- **Diffusion**: Diffusion-based action prediction (e.g., DDPM, DDIM)
- **GMM**: Gaussian Mixture Model head
- **Deterministic**: Simple MLP-based deterministic prediction
- **Autoregressive**: Sequential action prediction

## Configuration Structure

Each action head config should include:

```yaml
# Action head type (must match registry name)
type: "diffusion"

# Action space configuration
action_dim: 7  # Dimension of action space
action_horizon: 10  # Number of future actions to predict
chunk_size: 10  # Number of actions executed per step

# Architecture parameters
hidden_dims: [512, 256]
activation: "relu"

# Head-specific parameters
# (e.g., for diffusion: num_diffusion_steps, noise_schedule)
```

## Usage

Reference action head configs in experiment configs:
```yaml
model:
  action_head: "configs/model/action_head/diffusion.yaml"
```
