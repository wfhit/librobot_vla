# LibroBot VLA

> **Comprehensive Framework for Vision-Language-Action Models for Robot Learning**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

## What is LibroBot VLA?

LibroBot VLA is a modular, extensible framework for building Vision-Language-Action (VLA) models that enable robots to understand natural language instructions, perceive their environment through vision, and execute appropriate actions. The framework provides:

- **5 VLM Backend Families** (11 variants) - State-of-the-art vision-language models
- **8 VLA Framework Architectures** - Complete robot learning systems
- **Comprehensive Action Heads** - Diverse action prediction mechanisms
- **Registry-Based Plugin System** - Easy extensibility without code modification
- **Production-Ready Infrastructure** - Training, inference, and deployment tools

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/wfhit/librobot_vla.git
cd librobot_vla

# Install dependencies
pip install -e .

# Install optional dependencies
pip install -e ".[dev]"  # Development tools
pip install -e ".[train]"  # Training tools
pip install -e ".[inference]"  # Inference tools
```

### Basic Usage

```python
import torch
from librobot.models.vlm import create_vlm
from librobot.models.frameworks import create_vla

# Create VLM backend
vlm = create_vlm("qwen2-vl-2b", pretrained=True)

# Create VLA framework
vla = create_vla(
    "groot",              # Framework name
    vlm=vlm,              # VLM backbone
    action_dim=7,         # Robot action dimension
    state_dim=14,         # Proprioception dimension
)

# Training
images = torch.randn(2, 3, 224, 224)
text = ["pick up the cup", "move forward"]
proprioception = torch.randn(2, 14)
actions = torch.randn(2, 7)

outputs = vla(images, text, proprioception, actions)
loss = outputs["loss"]
loss.backward()

# Inference
predicted_actions = vla.predict_action(images, text, proprioception)
```

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         LibroBot VLA                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐           │
│  │    VLM     │───▶│    VLA     │───▶│   Action   │           │
│  │  Backend   │    │ Framework  │    │    Head    │           │
│  └────────────┘    └────────────┘    └────────────┘           │
│       ▲                  ▲                  ▲                  │
│       │                  │                  │                  │
│  ┌────┴──────────────────┴──────────────────┴────┐            │
│  │           Component Registry System            │            │
│  └────────────────────────────────────────────────┘            │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │   Data   │  │ Training │  │Inference │  │  Robot   │       │
│  │ Pipeline │  │   Loop   │  │  Server  │  │Interface │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Component Overview

#### VLM Backends

Vision-Language Models for multimodal understanding:

| Model | Parameters | Hidden Dim | Specialty |
|-------|------------|------------|-----------|
| **Qwen2-VL** | 2B, 7B | 1536, 3072 | 3D RoPE, high-res support |
| **Qwen3-VL** | 4B, 7B | 2048, 3072 | Latest version, best quality |
| **Florence-2** | 230M, 770M | 768, 1024 | Multi-task, lightweight |
| **PaliGemma** | 3B | 2048 | SigLIP vision, efficient |
| **InternVL2** | 2B, 8B | 2048, 4096 | High-res, multilingual |
| **LLaVA** | 7B, 13B | 4096, 5120 | CLIP vision, baseline |

#### VLA Frameworks

Complete robot learning architectures:

| Framework | Type | Best For |
|-----------|------|----------|
| **GR00T** | Diffusion | Multi-camera, stable training |
| **π0** | Flow Matching | Complex state spaces |
| **Octo** | Unified Transformer | Multi-task learning |
| **OpenVLA** | VLM Fine-tuning | Language-guided tasks |
| **RT-2** | Token-based | Discrete actions |
| **ACT** | Action Chunking | Bi-manual manipulation |
| **Helix** | Hierarchical | Long-horizon planning |
| **Custom** | User-defined | Experimentation |

#### Action Heads

Various action prediction mechanisms:
- **MLP Output-from-Tokens** - Fast, direct prediction
- **Transformer-based** - Sequential modeling
- **Diffusion Models** - DDPM, DDIM, EDM
- **Flow Matching** - Rectified Flow, OT-CFM
- **Hybrid Approaches** - Combined methods

## Key Features

### 🎯 Registry Pattern

Dynamic component discovery and instantiation:

```python
# Automatic registration
@register_vlm(name="my-vlm", aliases=["mvlm"])
class MyVLM(AbstractVLM):
    pass

# Use anywhere
vlm = create_vlm("my-vlm")
```

### ⚙️ Config-Driven Design

Reproducible experiments through configuration:

```yaml
model:
  framework: groot
  vlm:
    name: qwen2-vl-2b
    freeze: true
  action_dim: 7
```

### 🔌 Plugin Architecture

Extend without modifying core code:

```python
# Third-party plugin
import my_custom_plugin  # Auto-registers

# Use immediately
component = create_component("my-plugin-name")
```

### 🛡️ Type Safety

Complete type annotations:

```python
def forward(
    self,
    images: torch.Tensor,
    text: Optional[Union[str, List[str]]] = None,
    **kwargs: Any
) -> Dict[str, torch.Tensor]:
    pass
```

### 🧪 Comprehensive Testing

Quality assurance:
- 200+ test cases
- Interface compliance tests
- Integration tests
- End-to-end tests

## Directory Structure

```
librobot_vla/
├── librobot/              # Main package
│   ├── models/            # VLMs, frameworks, action heads
│   ├── data/              # Datasets, transforms, tokenizers
│   ├── training/          # Training infrastructure
│   ├── inference/         # Model serving
│   ├── robots/            # Hardware interfaces
│   ├── evaluation/        # Evaluation tools
│   └── utils/             # Shared utilities
│
├── configs/               # YAML configurations
│   ├── models/            # Model configs
│   ├── data/              # Data configs
│   └── training/          # Training configs
│
├── scripts/               # Executable scripts
│   ├── train.py           # Training
│   ├── inference.py       # Inference
│   ├── evaluate.py        # Evaluation
│   └── export.py          # Model export
│
├── examples/              # Example code
├── tests/                 # Test suite
└── docs/                  # Documentation
    └── design/            # Design documentation
```

## Documentation

### Design Documentation

Comprehensive architecture and design documentation in `docs/design/`:

1. **[ARCHITECTURE.md](docs/design/ARCHITECTURE.md)** - Complete system architecture with Mermaid diagrams
2. **[PROJECT_STRUCTURE.md](docs/design/PROJECT_STRUCTURE.md)** - Complete file tree with descriptions
3. **[DESIGN_PRINCIPLES.md](docs/design/DESIGN_PRINCIPLES.md)** - Core design principles
4. **[COMPONENT_GUIDE.md](docs/design/COMPONENT_GUIDE.md)** - Step-by-step guides for adding components
5. **[API_CONTRACTS.md](docs/design/API_CONTRACTS.md)** - Abstract base class interfaces
6. **[ROADMAP.md](docs/design/ROADMAP.md)** - Implementation phases
7. **[QUICK_REFERENCE.md](docs/design/QUICK_REFERENCE.md)** - Visual directory tree and quick lookup tables

### Implementation Summaries

- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - VLA Framework implementation details
- **[VLM_IMPLEMENTATION_SUMMARY.md](VLM_IMPLEMENTATION_SUMMARY.md)** - VLM backend implementation details

## Usage Examples

### Training a GR00T Model

```python
import torch
from librobot.models.vlm import create_vlm
from librobot.models.frameworks import create_vla
from librobot.data.datasets import create_dataset
from torch.utils.data import DataLoader

# Setup
vlm = create_vlm("qwen2-vl-2b", pretrained=True)
vla = create_vla("groot", vlm=vlm, action_dim=7, state_dim=14)

dataset = create_dataset("rlds", path="/data/robot_demos", split="train")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

optimizer = torch.optim.AdamW(vla.parameters(), lr=1e-4)

# Training loop
for epoch in range(100):
    for batch in dataloader:
        outputs = vla(
            batch["images"],
            batch["text"],
            batch["proprioception"],
            batch["actions"]
        )
        
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Multi-Framework Comparison

```python
from librobot.models.vlm import create_vlm
from librobot.models.frameworks import create_vla

# Shared VLM
vlm = create_vlm("qwen2-vl-2b", pretrained=True)

# Test different frameworks
frameworks = ["groot", "pi0", "octo", "rt2", "act"]
results = {}

for framework_name in frameworks:
    vla = create_vla(framework_name, vlm=vlm, action_dim=7)
    
    # Evaluate
    outputs = vla(test_images, test_text, test_proprio, test_actions)
    results[framework_name] = {
        "loss": outputs["loss"].item(),
        "params": vla.get_num_parameters(trainable_only=True)
    }

print(results)
```

### Custom VLM Plugin

```python
from librobot.models.vlm import AbstractVLM, register_vlm

@register_vlm(name="my-vlm", aliases=["mvlm"])
class MyVLM(AbstractVLM):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Initialize your architecture
    
    def forward(self, images, text, **kwargs):
        # Your implementation
        pass
    
    def encode_image(self, images):
        # Your implementation
        pass
    
    def encode_text(self, text):
        # Your implementation
        pass
    
    def get_embedding_dim(self):
        return self.hidden_dim
    
    @property
    def config(self):
        return {"hidden_dim": self.hidden_dim}

# Use immediately
vlm = create_vlm("my-vlm", hidden_dim=1024)
```

## Hardware Requirements

### Minimum Requirements
- **GPU:** NVIDIA GPU with 8GB+ VRAM (GTX 1080, RTX 2070, etc.)
- **CPU:** 4+ cores
- **RAM:** 16GB+
- **Storage:** 50GB+ for models and data

### Recommended Requirements
- **GPU:** NVIDIA GPU with 16GB+ VRAM (RTX 3090, A4000, etc.)
- **CPU:** 8+ cores
- **RAM:** 32GB+
- **Storage:** 500GB+ SSD

### Production Requirements
- **GPU:** NVIDIA GPU with 24GB+ VRAM (RTX 4090, A5000, A6000)
- **CPU:** 16+ cores
- **RAM:** 64GB+
- **Storage:** 1TB+ NVMe SSD

## Current Status

### ✅ Completed (v0.1.0)
- **5 VLM families, 11 variants** (~3,620 LOC)
- **8 VLA frameworks** (~2,900 LOC)
- **Complete action head library**
- **Registry-based architecture**
- **Comprehensive documentation**
- **200+ test cases**

### 🔄 In Progress
- Distributed training (DDP, FSDP)
- Inference server (FastAPI, gRPC)
- Additional robot interfaces
- Model optimization (ONNX, TensorRT)

### 📋 Planned (v0.2.0+)
- Advanced evaluation benchmarks
- Sim-to-real transfer tools
- Multi-robot coordination
- Online learning and adaptation
- Zero-shot generalization

See [ROADMAP.md](docs/design/ROADMAP.md) for detailed development plans.

## Contributing

We welcome contributions! Areas of interest:

1. **High Priority:**
   - Distributed training implementations
   - Inference server implementations
   - Additional robot interfaces
   - Evaluation benchmarks

2. **Medium Priority:**
   - Dataset format support
   - Data augmentation techniques
   - Model optimization tools
   - Documentation improvements

3. **Community:**
   - Bug reports and fixes
   - Feature requests
   - Tutorials and examples
   - Third-party plugins

See [COMPONENT_GUIDE.md](docs/design/COMPONENT_GUIDE.md) for detailed instructions on contributing new components.

## Citation

If you use LibroBot VLA in your research, please cite:

```bibtex
@software{librobot_vla2024,
  title={LibroBot VLA: A Comprehensive Framework for Vision-Language-Action Models},
  author={LibroBot Contributors},
  year={2024},
  url={https://github.com/wfhit/librobot_vla}
}
```

## Related Work

This framework builds upon and is inspired by:

- **Qwen-VL** - Alibaba Cloud
- **Florence-2** - Microsoft
- **PaliGemma** - Google
- **InternVL** - OpenGVLab
- **LLaVA** - University of Wisconsin-Madison, Microsoft
- **GR00T** - NVIDIA
- **π0** - Physical Intelligence
- **Octo** - UC Berkeley
- **OpenVLA** - UC Berkeley
- **RT-2** - Google DeepMind
- **ACT** - Stanford (ALOHA)
- **Helix** - Figure AI

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation:** [docs/](docs/)
- **Issues:** [GitHub Issues](https://github.com/wfhit/librobot_vla/issues)
- **Discussions:** [GitHub Discussions](https://github.com/wfhit/librobot_vla/discussions)

---

**Built with ❤️ for the robotics community**