# VLM Configuration Directory

This directory contains configuration files for Vision-Language Models (VLMs) used as backbones in the VLA framework.

## Supported VLMs

The VLM configurations should specify models registered in the VLM factory:

- **Qwen-VL**: Qwen2-VL models (e.g., `qwen2_vl_2b`, `qwen2_vl_7b`)
- **Florence**: Florence-2 models (e.g., `florence2_base`, `florence2_large`)
- **PaliGemma**: PaliGemma models
- **Other VLMs**: Any model registered via `@register_vlm`

## Configuration Structure

Each VLM config should include:

```yaml
# Model identifier (must match registry name)
name: "qwen2_vl_2b"

# Model initialization parameters
pretrained: true
pretrained_path: "Qwen/Qwen2-VL-2B-Instruct"

# Feature extraction settings
hidden_dim: 1536  # Output dimension
use_projection: true  # Whether to add a projection layer

# Freezing strategy
freeze_vision: false
freeze_text: false
freeze_layers: []  # List of layer indices to freeze

# Low-rank adaptation (optional)
lora:
  enabled: false
  r: 16
  alpha: 32
  target_modules: ["q_proj", "v_proj"]
```

## Usage

Reference VLM configs in experiment configs:
```yaml
model:
  vlm: "configs/model/vlm/qwen2_vl_2b.yaml"
```
