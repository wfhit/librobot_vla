# VLA Framework Configuration Directory

This directory contains configuration files for different VLA framework architectures that combine VLMs, encoders, and action heads.

## Supported Frameworks

The framework configurations should specify architectures registered in the VLA factory:

- **GROOT-style**: Combines frozen VLM with diffusion policy
- **OpenVLA-style**: Fine-tuned VLM with autoregressive actions
- **RT-2-style**: Vision-language-action transformer
- **OCTO-style**: Generalist transformer policy

## Configuration Structure

Each framework config should include:

```yaml
# Framework type (must match registry name)
type: "groot_style"

# Component specifications (paths to other configs or inline configs)
vlm: "configs/model/vlm/qwen2_vl_2b.yaml"
encoder: "configs/model/encoder/mlp.yaml"
action_head: "configs/model/action_head/diffusion.yaml"

# Feature fusion configuration
fusion:
  method: "cross_attention"  # How to fuse vision, language, and proprio
  num_layers: 2
  hidden_dim: 512

# Training strategy
freeze_vlm: true
freeze_encoder: false
```

## Usage

Reference framework configs in experiment configs:
```yaml
model:
  framework: "configs/model/framework/groot_style.yaml"
```
