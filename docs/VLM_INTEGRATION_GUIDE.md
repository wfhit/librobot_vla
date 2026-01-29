# VLM Integration Guide for VLA Framework

This guide explains how to integrate Vision-Language Models (VLMs) with the LibroBot VLA framework.

## Quick Start

### 1. Basic VLM Usage

```python
from librobot.models.vlm import create_vlm
import torch

# Create a VLM
model = create_vlm(
    'qwen2-vl-2b',
    pretrained='Qwen/Qwen2-VL-2B-Instruct',  # Optional
)

# Prepare inputs
images = torch.randn(2, 3, 224, 224)
input_ids = torch.randint(0, 1000, (2, 50))

# Forward pass
outputs = model(images=images, input_ids=input_ids)
```

## Model Selection Guide

| Task | Recommended Model | Reason |
|------|-------------------|---------|
| **Robotic Manipulation** | Qwen2-VL-2B | Dynamic resolution, efficient |
| **Multi-task Learning** | Florence-2 | Built-in multi-task support |
| **Instruction Following** | LLaVA-1.5-7B | Strong instruction tuning |
| **High-res Images** | InternVL2-2B | Up to 4K resolution support |
| **General Purpose** | PaliGemma-3B | Good balance |

## Resources

- [VLM README](../librobot/models/vlm/README.md) - Detailed documentation
- [VLM Demo](../examples/vlm_demo.py) - Example usage
- [Tests](../tests/test_vlm_implementations.py) - Test suite
