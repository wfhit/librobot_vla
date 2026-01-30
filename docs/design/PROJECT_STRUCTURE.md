# LibroBot VLA Project Structure

## Table of Contents
- [Overview](#overview)
- [Complete Directory Tree](#complete-directory-tree)
- [Module Descriptions](#module-descriptions)
- [File Organization](#file-organization)
- [Naming Conventions](#naming-conventions)
- [Import Structure](#import-structure)

## Overview

This document provides a comprehensive guide to the LibroBot VLA framework's file and directory structure. The project follows Python best practices with clear separation of concerns and intuitive organization.

## Complete Directory Tree

```
librobot_vla/
├── librobot/                          # Main package
│   ├── __init__.py                    # Package initialization
│   ├── version.py                     # Version information
│   │
│   ├── models/                        # Neural network models
│   │   ├── __init__.py
│   │   │
│   │   ├── vlm/                       # Vision-Language Models
│   │   │   ├── __init__.py            # VLM exports
│   │   │   ├── base.py                # AbstractVLM interface (110 lines)
│   │   │   ├── registry.py            # VLM registry (69 lines)
│   │   │   ├── qwen_vl.py            # Qwen2/3-VL (795 lines)
│   │   │   ├── florence.py           # Florence-2 (730 lines)
│   │   │   ├── paligemma.py          # PaliGemma (653 lines)
│   │   │   ├── internvl.py           # InternVL2 (701 lines)
│   │   │   ├── llava.py              # LLaVA (741 lines)
│   │   │   ├── adapters/             # Fine-tuning adapters
│   │   │   │   ├── __init__.py
│   │   │   │   ├── lora.py           # LoRA adapter
│   │   │   │   └── qlora.py          # QLoRA adapter
│   │   │   └── utils/                # VLM utilities
│   │   │       ├── __init__.py
│   │   │       ├── kv_cache.py       # KV cache for generation
│   │   │       └── attention_sink.py  # Attention sink
│   │   │
│   │   ├── frameworks/                # VLA frameworks
│   │   │   ├── __init__.py            # Framework exports
│   │   │   ├── base.py                # AbstractVLA interface (180 lines)
│   │   │   ├── registry.py            # VLA registry (69 lines)
│   │   │   ├── groot_style.py        # NVIDIA GR00T (290 lines)
│   │   │   ├── pi0_style.py          # Physical Intelligence π0 (295 lines)
│   │   │   ├── octo_style.py         # Berkeley Octo (355 lines)
│   │   │   ├── openvla_style.py      # Berkeley OpenVLA (315 lines)
│   │   │   ├── rt2_style.py          # Google RT-2 (365 lines)
│   │   │   ├── act_style.py          # ALOHA ACT (420 lines)
│   │   │   ├── helix_style.py        # Figure AI Helix (440 lines)
│   │   │   └── custom.py             # Custom template (430 lines)
│   │   │
│   │   ├── action_heads/              # Action prediction heads
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # AbstractActionHead
│   │   │   ├── registry.py            # Action head registry
│   │   │   ├── mlp_oft.py            # Output-from-tokens MLP
│   │   │   ├── transformer_act.py    # Transformer-based ACT
│   │   │   ├── autoregressive_fast.py # Fast autoregressive
│   │   │   ├── hybrid.py              # Hybrid approaches
│   │   │   ├── diffusion/            # Diffusion models
│   │   │   │   ├── __init__.py
│   │   │   │   ├── ddpm.py           # DDPM
│   │   │   │   ├── ddim.py           # DDIM
│   │   │   │   └── edm.py            # EDM
│   │   │   └── flow_matching/        # Flow matching models
│   │   │       ├── __init__.py
│   │   │       ├── rectified_flow.py # Rectified flow
│   │   │       └── ot_cfm.py         # OT-CFM
│   │   │
│   │   ├── encoders/                  # Various encoders
│   │   │   ├── __init__.py
│   │   │   ├── state/                # State encoders
│   │   │   │   ├── __init__.py
│   │   │   │   ├── mlp.py            # MLP encoder
│   │   │   │   └── transformer.py    # Transformer encoder
│   │   │   ├── history/              # History encoders
│   │   │   │   ├── __init__.py
│   │   │   │   ├── lstm.py           # LSTM encoder
│   │   │   │   └── transformer.py    # Transformer encoder
│   │   │   └── fusion/               # Fusion modules
│   │   │       ├── __init__.py
│   │   │       ├── attention_fusion.py
│   │   │       └── film_fusion.py
│   │   │
│   │   └── components/                # Reusable components
│   │       ├── __init__.py
│   │       ├── activations.py         # SwiGLU, GeGLU, etc.
│   │       ├── attention/            # Attention mechanisms
│   │       │   ├── __init__.py
│   │       │   ├── standard.py       # Standard attention
│   │       │   ├── flash_attention.py # Flash Attention 2
│   │       │   ├── sliding_window.py # Sliding window
│   │       │   └── block_wise.py     # Block-wise attention
│   │       ├── normalization/        # Normalization layers
│   │       │   ├── __init__.py
│   │       │   ├── layernorm.py
│   │       │   ├── rmsnorm.py
│   │       │   └── groupnorm.py
│   │       └── positional/           # Position embeddings
│   │           ├── __init__.py
│   │           ├── sinusoidal.py     # Sinusoidal
│   │           ├── rotary.py         # RoPE
│   │           └── alibi.py          # ALiBi
│   │
│   ├── data/                          # Data handling
│   │   ├── __init__.py
│   │   ├── datasets/                 # Dataset implementations
│   │   │   ├── __init__.py
│   │   │   ├── base.py               # AbstractDataset
│   │   │   ├── registry.py           # Dataset registry
│   │   │   ├── rlds_dataset.py      # RLDS format
│   │   │   ├── hdf5_dataset.py      # HDF5 format
│   │   │   └── dummy_dataset.py     # Dummy/testing
│   │   ├── transforms/               # Data transformations
│   │   │   ├── __init__.py
│   │   │   ├── image_transforms.py
│   │   │   └── action_transforms.py
│   │   └── tokenizers/               # Tokenizers
│   │       ├── __init__.py
│   │       ├── action_tokenizer.py
│   │       └── text_tokenizer.py
│   │
│   ├── training/                      # Training infrastructure
│   │   ├── __init__.py
│   │   ├── trainer.py                 # Main trainer
│   │   ├── losses/                   # Loss functions
│   │   │   ├── __init__.py
│   │   │   ├── action_loss.py
│   │   │   ├── vq_loss.py
│   │   │   └── diffusion_loss.py
│   │   └── callbacks/                # Training callbacks
│   │       ├── __init__.py
│   │       ├── checkpoint.py
│   │       ├── logging.py
│   │       └── early_stopping.py
│   │
│   ├── inference/                     # Model inference
│   │   ├── __init__.py
│   │   ├── predictor.py              # Base predictor
│   │   ├── batched_predictor.py     # Batched inference
│   │   └── server/                   # Inference servers
│   │       ├── __init__.py
│   │       ├── fastapi_server.py    # FastAPI server
│   │       └── grpc_server.py       # gRPC server
│   │
│   ├── robots/                        # Robot interfaces
│   │   ├── __init__.py
│   │   ├── base.py                   # AbstractRobot
│   │   ├── registry.py               # Robot registry
│   │   ├── so100_arm.py             # SO-100 arm
│   │   ├── humanoid.py              # Humanoid robot
│   │   └── wheel_loader.py          # Wheel loader
│   │
│   ├── evaluation/                    # Model evaluation
│   │   ├── __init__.py
│   │   ├── metrics.py                # Evaluation metrics
│   │   ├── benchmarks.py            # Standard benchmarks
│   │   └── sim_evaluation.py        # Simulation eval
│   │
│   ├── collection/                    # Data collection
│   │   ├── __init__.py
│   │   └── collector.py             # Data collector
│   │
│   └── utils/                         # Utilities
│       ├── __init__.py
│       ├── registry.py               # Global registry
│       ├── config.py                # Config management
│       ├── checkpoint.py            # Checkpointing
│       └── device.py                # Device management
│
├── configs/                           # Configuration files
│   ├── models/                       # Model configs
│   │   ├── groot.yaml
│   │   ├── pi0.yaml
│   │   ├── octo.yaml
│   │   ├── openvla.yaml
│   │   ├── rt2.yaml
│   │   ├── act.yaml
│   │   └── helix.yaml
│   ├── data/                         # Data configs
│   │   └── default.yaml
│   └── training/                     # Training configs
│       └── default.yaml
│
├── scripts/                           # Executable scripts
│   ├── __init__.py
│   ├── train.py                      # Training script
│   ├── inference.py                  # Inference script
│   ├── evaluate.py                   # Evaluation script
│   └── export.py                     # Model export script
│
├── examples/                          # Example code
│   ├── vlm_demo.py                   # VLM usage examples
│   └── frameworks/
│       └── complete_demo.py          # Framework examples
│
├── tests/                             # Test suite
│   ├── __init__.py
│   ├── test_vlm_implementations.py  # VLM tests
│   ├── test_frameworks.py           # Framework tests
│   ├── test_action_heads.py         # Action head tests
│   ├── test_registry.py             # Registry tests
│   └── test_integration.py          # Integration tests
│
├── docs/                              # Documentation
│   ├── design/                       # Design docs
│   │   ├── ARCHITECTURE.md
│   │   ├── PROJECT_STRUCTURE.md
│   │   ├── DESIGN_PRINCIPLES.md
│   │   ├── COMPONENT_GUIDE.md
│   │   ├── API_CONTRACTS.md
│   │   ├── ROADMAP.md
│   │   └── QUICK_REFERENCE.md
│   ├── PROJECT_COMPLETION_SUMMARY.md
│   ├── VLM_INTEGRATION_GUIDE.md
│   └── tutorials/
│       └── getting_started.md
│
├── docker/                            # Docker configurations
│   ├── Dockerfile.base               # Base image with CUDA + PyTorch
│   ├── Dockerfile.train              # Training environment
│   ├── Dockerfile.deploy             # Lightweight inference server
│   └── docker-compose.yml
│
├── requirements/                      # Requirements
│   ├── base.txt                      # Base requirements
│   ├── dev.txt                       # Development requirements
│   └── optional.txt                  # Optional dependencies
│
├── .github/                           # GitHub configuration
│   └── workflows/
│       ├── ci.yml
│       └── tests.yml
│
├── setup.py                           # Package setup
├── pyproject.toml                    # Project metadata
├── pytest.ini                        # Pytest configuration
├── Makefile                          # Build automation
├── README.md                         # Main README
├── LICENSE                           # License file
├── IMPLEMENTATION_SUMMARY.md         # Framework implementation summary
├── VLM_IMPLEMENTATION_SUMMARY.md     # VLM implementation summary
└── ARCHITECTURE_OVERVIEW.md          # Architecture overview (to be created)
```

## Module Descriptions

### Core Modules

#### `librobot/models/`
Contains all neural network model implementations.

**Key Submodules:**
- `vlm/`: Vision-Language Model backends (5 families, 11 variants)
- `frameworks/`: VLA framework implementations (8 frameworks)
- `action_heads/`: Action prediction mechanisms (7 types)
- `encoders/`: State, history, and fusion encoders
- `components/`: Reusable building blocks

**Purpose:** Provides all neural network architectures needed for VLA systems.

#### `librobot/data/`
Handles data loading, preprocessing, and augmentation.

**Key Submodules:**
- `datasets/`: Dataset loaders (RLDS, HDF5, custom)
- `transforms/`: Image and action transformations
- `tokenizers/`: Text and action tokenization

**Purpose:** Unified data pipeline for training and inference.

#### `librobot/training/`
Training infrastructure and utilities.

**Key Submodules:**
- `losses/`: Loss function implementations
- `callbacks/`: Training callbacks (checkpointing, logging, early stopping)

**Purpose:** Complete training loop with monitoring and checkpointing.

#### `librobot/inference/`
Model serving and deployment.

**Key Submodules:**
- `server/`: FastAPI and gRPC servers

**Purpose:** Production-ready model serving infrastructure.

#### `librobot/robots/`
Hardware interfaces for different robots.

**Key Implementations:**
- SO-100 robot arm
- Humanoid robots
- Wheel loader

**Purpose:** Abstract hardware differences behind common interface.

#### `librobot/evaluation/`
Model evaluation and benchmarking.

**Purpose:** Standardized evaluation protocols for comparing models.

#### `librobot/utils/`
Shared utilities across the framework.

**Key Utilities:**
- Global registry system
- Configuration management
- Checkpointing utilities
- Device management

**Purpose:** Common functionality to avoid code duplication.

### Supporting Directories

#### `configs/`
YAML configuration files for models, data, and training.

**Organization:**
- `models/`: Per-framework configurations
- `data/`: Dataset configurations
- `training/`: Training hyperparameters

**Purpose:** Reproducible experiments through configuration files.

#### `scripts/`
Executable Python scripts for common tasks.

**Scripts:**
- `train.py`: Model training
- `inference.py`: Running inference
- `evaluate.py`: Model evaluation
- `export.py`: Model export (ONNX, TorchScript)

**Purpose:** Command-line interface for framework operations.

#### `examples/`
Example code demonstrating framework usage.

**Examples:**
- VLM usage demonstrations
- Framework initialization and training
- Multi-framework comparisons

**Purpose:** Educational examples and starting points.

#### `tests/`
Comprehensive test suite.

**Test Organization:**
- Unit tests for individual components
- Integration tests for full pipelines
- Registry tests
- End-to-end tests

**Purpose:** Ensure code quality and catch regressions.

#### `docs/`
Complete documentation.

**Documentation Structure:**
- `design/`: Architecture and design documentation
- Tutorials and guides
- API reference

**Purpose:** Comprehensive developer and user documentation.

## File Organization

### Module Structure Pattern

Every module follows this structure:

```
module_name/
├── __init__.py          # Public API exports
├── base.py              # Abstract base classes
├── registry.py          # Module registry
├── implementation_1.py  # Implementations
├── implementation_2.py
└── submodule/           # Submodules if needed
    ├── __init__.py
    └── ...
```

### File Naming Conventions

| File Type | Convention | Example |
|-----------|-----------|---------|
| Abstract base class | `base.py` | `librobot/models/vlm/base.py` |
| Registry | `registry.py` | `librobot/models/vlm/registry.py` |
| Implementation | `descriptive_name.py` | `qwen_vl.py`, `groot_style.py` |
| Test file | `test_*.py` | `test_vlm_implementations.py` |
| Config file | `descriptive.yaml` | `groot.yaml` |
| Script | `verb.py` | `train.py`, `evaluate.py` |

### Code Organization Within Files

```python
"""Module docstring."""

# Standard library imports
import os
from typing import Dict, List

# Third-party imports
import torch
import torch.nn as nn

# Local imports
from librobot.models.base import AbstractVLM
from librobot.utils.registry import register_vlm

# Constants
HIDDEN_DIM = 768

# Class definitions
@register_vlm(name="my-vlm")
class MyVLM(AbstractVLM):
    """Class docstring."""
    pass

# Helper functions
def helper_function():
    """Helper docstring."""
    pass
```

## Naming Conventions

### Python Naming

| Element | Convention | Example |
|---------|-----------|---------|
| Package | lowercase | `librobot` |
| Module | snake_case | `qwen_vl.py` |
| Class | PascalCase | `AbstractVLM`, `QwenVLModel` |
| Function | snake_case | `create_vlm()`, `get_embedding_dim()` |
| Variable | snake_case | `hidden_dim`, `num_layers` |
| Constant | UPPER_CASE | `HIDDEN_DIM`, `MAX_SEQ_LEN` |
| Private | _prefix | `_internal_method()` |

### Abbreviations

| Abbreviation | Full Form |
|-------------|-----------|
| VLM | Vision-Language Model |
| VLA | Vision-Language-Action |
| RLDS | Reverb Learning Data Store |
| MLP | Multi-Layer Perceptron |
| RoPE | Rotary Position Embedding |
| LoRA | Low-Rank Adaptation |
| QLoRA | Quantized LoRA |
| DDPM | Denoising Diffusion Probabilistic Model |
| CFM | Conditional Flow Matching |
| ACT | Action Chunking Transformer |
| RT | Robotics Transformer |

### Framework Naming

VLA frameworks use `_style` suffix to indicate they're inspired by but not exact replicas:
- `groot_style.py` - GR00T-inspired
- `pi0_style.py` - π0-inspired
- `rt2_style.py` - RT-2-inspired

VLM implementations use descriptive names matching the original:
- `qwen_vl.py` - Qwen-VL family
- `florence.py` - Florence-2
- `llava.py` - LLaVA

## Import Structure

### Package Imports

```python
# From top-level
from librobot import __version__

# From models
from librobot.models.vlm import create_vlm, list_vlms
from librobot.models.frameworks import create_vla, list_vlas

# From specific implementations
from librobot.models.vlm.qwen_vl import QwenVLModel
from librobot.models.frameworks.groot_style import GR00TVLA

# From utilities
from librobot.utils.config import load_config
from librobot.utils.checkpoint import save_checkpoint
```

### Recommended Import Pattern

```python
# Option 1: Use registry (recommended)
from librobot.models.vlm import create_vlm
from librobot.models.frameworks import create_vla

vlm = create_vlm("qwen2-vl-2b", pretrained=True)
vla = create_vla("groot", vlm=vlm, action_dim=7)

# Option 2: Direct import
from librobot.models.vlm.qwen_vl import QwenVLModel
from librobot.models.frameworks.groot_style import GR00TVLA

vlm = QwenVLModel(model_size="2b")
vla = GR00TVLA(vlm=vlm, action_dim=7)
```

### Circular Import Prevention

The framework uses several strategies to prevent circular imports:

1. **Registry pattern**: Components register themselves at import time
2. **Late imports**: Import inside functions when needed
3. **Abstract base classes**: Define interfaces separately from implementations
4. **Type hints**: Use string literals for forward references

## File Size Guidelines

| Component Type | Target Size | Max Size |
|---------------|-------------|----------|
| Abstract base | 100-200 lines | 300 lines |
| Registry | 50-100 lines | 150 lines |
| Simple implementation | 200-400 lines | 600 lines |
| Complex implementation | 400-800 lines | 1000 lines |
| Test file | 200-500 lines | No limit |

Large implementations are acceptable for complex architectures (e.g., frameworks with multiple components).

## Code Style

The project follows PEP 8 with these specific guidelines:

- **Line length**: 100 characters (not 80)
- **Quotes**: Double quotes for strings
- **Docstrings**: Google style
- **Type hints**: Required for public APIs
- **Imports**: Sorted with isort

## Directory Permissions

- Source code: 644 (rw-r--r--)
- Scripts: 755 (rwxr-xr-x)
- Configs: 644 (rw-r--r--)
- Documentation: 644 (rw-r--r--)

## Version Control

### Excluded Files (.gitignore)

```
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
dist/
*.egg-info/
.pytest_cache/
.coverage
.env
checkpoints/
logs/
*.pth
*.ckpt
```

### Important Files (Always Include)

- `__init__.py` in all packages
- `base.py` defining interfaces
- `registry.py` for component registration
- Configuration files
- Documentation

## Navigation Tips

### Finding Components

1. **VLM implementations**: `librobot/models/vlm/*.py`
2. **VLA frameworks**: `librobot/models/frameworks/*_style.py`
3. **Action heads**: `librobot/models/action_heads/*.py`
4. **Datasets**: `librobot/data/datasets/*.py`
5. **Robots**: `librobot/robots/*.py`

### Understanding Component Relationships

1. Start with `base.py` to understand the interface
2. Check `registry.py` to see registration mechanism
3. Look at implementation files for concrete examples
4. Review tests to understand usage patterns

### Extending the Framework

1. **Add VLM**: Create file in `librobot/models/vlm/`
2. **Add VLA**: Create file in `librobot/models/frameworks/`
3. **Add Action Head**: Create file in `librobot/models/action_heads/`
4. **Add Dataset**: Create file in `librobot/data/datasets/`
5. **Add Robot**: Create file in `librobot/robots/`

See [COMPONENT_GUIDE.md](./COMPONENT_GUIDE.md) for detailed instructions.

## Build Artifacts

### Generated During Build

- `build/`: Build artifacts
- `dist/`: Distribution packages
- `*.egg-info/`: Package metadata

### Generated During Testing

- `.pytest_cache/`: Pytest cache
- `.coverage`: Coverage data
- `htmlcov/`: Coverage reports

### Generated During Training

- `checkpoints/`: Model checkpoints
- `logs/`: Training logs
- `tensorboard/`: TensorBoard logs

## Conclusion

The LibroBot VLA framework follows a clear, hierarchical structure that makes it easy to navigate, understand, and extend. The consistent organization and naming conventions ensure that developers can quickly locate and modify components.

For more information:
- [Architecture](./ARCHITECTURE.md) - System architecture
- [Design Principles](./DESIGN_PRINCIPLES.md) - Design decisions
- [Component Guide](./COMPONENT_GUIDE.md) - Adding new components
