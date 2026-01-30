# Architecture Overview

This guide provides a user-friendly overview of LibroBot VLA's architecture. For detailed technical specifications, see [design/ARCHITECTURE.md](design/ARCHITECTURE.md).

## Table of Contents

- [What is LibroBot VLA?](#what-is-librobot-vla)
- [Core Concepts](#core-concepts)
- [System Architecture](#system-architecture)
- [Component Relationships](#component-relationships)
- [Data Flow](#data-flow)
- [Key Design Patterns](#key-design-patterns)
- [Extension Points](#extension-points)

## What is LibroBot VLA?

LibroBot VLA is a **comprehensive framework** for building Vision-Language-Action (VLA) models that enable robots to:

1. **See** their environment (Vision)
2. **Understand** natural language instructions (Language)
3. **Execute** appropriate actions (Action)

Think of it as a complete toolkit for teaching robots to follow instructions like "pick up the red cup" by:
- Processing camera images
- Understanding the text instruction
- Predicting the right motor commands

## Core Concepts

### The VLA Pipeline

```
Camera Images  ─┐
                ├──▶ VLM ──▶ Features ──▶ VLA ──▶ Actions ──▶ Robot
Text Command  ─┘                ▲
                                 │
Robot State ────────────────────┘
```

### Three Main Components

#### 1. VLM (Vision-Language Model)
**What it does**: Converts images and text into a shared representation

**Think of it as**: The "brain" that understands what it sees and reads

**Examples**: 
- Qwen2-VL: High-quality multimodal understanding
- Florence-2: Lightweight and fast
- PaliGemma: Balanced performance

```python
# VLM converts raw inputs to embeddings
vlm = create_vlm("qwen2-vl-2b")
embeddings = vlm(images=camera_images, text="pick up the cup")
```

#### 2. VLA Framework (Vision-Language-Action)
**What it does**: Combines VLM features with robot state to predict actions

**Think of it as**: The "policy" that decides what to do

**Examples**:
- GROOT: Stable diffusion-based actions
- π0: Flow matching for smooth control
- Octo: Multi-task transformer
- RT-2: Tokenized action prediction

```python
# VLA predicts actions from multimodal inputs
vla = create_vla("groot", vlm=vlm, action_dim=7)
actions = vla.predict_action(images, text, robot_state)
```

#### 3. Action Head
**What it does**: Transforms features into robot motor commands

**Think of it as**: The "translator" from abstract understanding to concrete movement

**Types**:
- MLP: Direct regression (fast)
- Transformer: Sequential modeling (temporal)
- Diffusion: Multimodal distributions (robust)
- Flow Matching: Continuous flows (smooth)

## System Architecture

### High-Level View

```
┌─────────────────────────────────────────────────────────┐
│                     LibroBot VLA                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │   VLM    │─▶│   VLA    │─▶│  Action  │            │
│  │ Backend  │  │Framework │  │   Head   │            │
│  └──────────┘  └──────────┘  └──────────┘            │
│       ▲             ▲             ▲                    │
│       │             │             │                    │
│  ┌────┴─────────────┴─────────────┴─────┐            │
│  │      Component Registry System        │            │
│  └──────────────────────────────────────┘            │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                  Supporting Systems                     │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │   Data   │  │ Training │  │Inference │            │
│  │ Pipeline │  │  System  │  │  Server  │            │
│  └──────────┘  └──────────┘  └──────────┘            │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Directory Structure

Understanding the codebase organization:

```
librobot_vla/
│
├── librobot/                  # Main package
│   ├── models/                # Model implementations
│   │   ├── vlm/              # Vision-Language Models
│   │   │   ├── base.py       # Abstract interfaces
│   │   │   ├── qwen.py       # Qwen implementations
│   │   │   ├── florence.py   # Florence implementations
│   │   │   └── registry.py   # Model registry
│   │   │
│   │   ├── frameworks/       # VLA Frameworks
│   │   │   ├── base.py       # Abstract interfaces
│   │   │   ├── groot.py      # GROOT implementation
│   │   │   ├── pi0.py        # π0 implementation
│   │   │   └── registry.py   # Framework registry
│   │   │
│   │   └── action_heads/     # Action prediction heads
│   │       ├── base.py       # Abstract interfaces
│   │       ├── mlp.py        # MLP heads
│   │       ├── diffusion.py  # Diffusion heads
│   │       └── registry.py   # Action head registry
│   │
│   ├── data/                  # Data loading
│   │   ├── datasets/         # Dataset loaders
│   │   ├── transforms/       # Data augmentation
│   │   └── tokenizers/       # Text processing
│   │
│   ├── training/             # Training infrastructure
│   │   ├── trainer.py        # Main training loop
│   │   ├── distributed.py    # Multi-GPU training
│   │   └── callbacks.py      # Training callbacks
│   │
│   ├── inference/            # Model serving
│   │   ├── server.py         # Inference server
│   │   └── client.py         # Client interface
│   │
│   ├── robots/               # Robot interfaces
│   │   ├── base.py           # Abstract robot base class
│   │   ├── registry.py       # Robot registry
│   │   │
│   │   ├── arms/             # Robot arm interfaces
│   │   │   ├── arm.py        # Arm base class
│   │   │   └── arm_robot.py  # Arm implementations (Franka, UR5, SO100, etc.)
│   │   │
│   │   ├── mobile/           # Mobile robot interfaces
│   │   │   ├── mobile.py         # MobileRobot base class
│   │   │   └── mobile_robot.py   # Mobile robot implementations
│   │   │
│   │   ├── mobile_manipulators/  # Mobile manipulator interfaces
│   │   │   ├── mobile_manipulator.py       # MobileManipulator base class
│   │   │   └── mobile_manipulator_robot.py # Implementations (Fetch, TIAGo)
│   │   │
│   │   ├── humanoids/        # Humanoid robot interfaces
│   │   │   ├── humanoid.py       # Humanoid base class
│   │   │   └── humanoid_robot.py # Humanoid implementations (Figure, GR1, H1)
│   │   │
│   │   ├── wheel_loaders/    # Wheel loader interfaces
│   │   │   ├── wheel_loader.py       # WheelLoaderRobot base class
│   │   │   └── wheel_loader_robot.py # WheelLoader comprehensive implementation
│   │   │
│   │   ├── excavators/       # Excavator interfaces
│   │   │   ├── excavator.py       # ExcavatorRobot base class
│   │   │   └── excavator_robot.py # Excavator comprehensive implementation
│   │   │
│   │   ├── articulated_trucks/  # Articulated truck interfaces
│   │   │   ├── articulated_truck.py       # ArticulatedTruckRobot base class
│   │   │   └── articulated_truck_robot.py # ArticulatedTruck comprehensive implementation
│   │   │
│   │   └── sensors/          # Sensor interfaces
│   │
│   └── utils/                # Shared utilities
│       ├── config.py         # Configuration
│       ├── logging.py        # Logging
│       └── registry.py       # Registry pattern
│
├── configs/                   # YAML configurations
├── scripts/                   # Executable scripts
├── examples/                  # Example code
└── docs/                      # Documentation
```

## Component Relationships

### How Components Work Together

```
┌─────────────────── VLA Framework ────────────────────┐
│                                                       │
│  ┌─────────────────────────────────────────────┐   │
│  │           VLM Backbone (Frozen)             │   │
│  │  ┌─────────┐        ┌──────────┐          │   │
│  │  │ Vision  │        │ Language │          │   │
│  │  │ Encoder │        │ Encoder  │          │   │
│  │  └────┬────┘        └─────┬────┘          │   │
│  │       │                   │                │   │
│  │       └───────┬───────────┘                │   │
│  │               ▼                            │   │
│  │    [Vision-Language Features]             │   │
│  └─────────────────┬───────────────────────────┘   │
│                    │                               │
│                    ▼                               │
│         ┌──────────────────────┐                  │
│         │  Feature Projection  │                  │
│         └──────────┬───────────┘                  │
│                    │                               │
│  ┌─────────────────┴──────────┐                   │
│  │  Proprioception Encoder    │                   │
│  └─────────────────┬──────────┘                   │
│                    │                               │
│                    ▼                               │
│         ┌──────────────────────┐                  │
│         │   Feature Fusion     │                  │
│         └──────────┬───────────┘                  │
│                    │                               │
│                    ▼                               │
│         ┌──────────────────────┐                  │
│         │     Action Head      │                  │
│         │  (Diffusion/Flow/    │                  │
│         │   Transformer/MLP)   │                  │
│         └──────────┬───────────┘                  │
│                    │                               │
└────────────────────┼───────────────────────────────┘
                     ▼
               [Robot Actions]
```

### Interface Contracts

Each component implements a well-defined interface:

```python
# VLM Interface
class AbstractVLM:
    def forward(images, text) -> embeddings
    def encode_image(images) -> image_embeddings
    def encode_text(text) -> text_embeddings
    def get_embedding_dim() -> int

# VLA Framework Interface
class AbstractVLA:
    def forward(images, text, proprio, actions) -> loss
    def predict_action(images, text, proprio) -> actions
    def get_num_parameters() -> int

# Action Head Interface
class AbstractActionHead:
    def forward(features, actions) -> loss
    def predict(features) -> actions
    def get_action_dim() -> int
```

## Data Flow

### Training Data Flow

```
Raw Data (HDF5/RLDS/etc)
        ↓
   [Dataset Loader]
        ↓
   Data Batches
   ├─ images: [B, C, H, W]
   ├─ text: [B, seq_len]
   ├─ proprio: [B, state_dim]
   └─ actions: [B, action_dim]
        ↓
    [VLM Encoder]
        ↓
   Embeddings [B, hidden_dim]
        ↓
   [VLA Framework]
   ├─ Feature Projection
   ├─ Proprio Encoding
   ├─ Feature Fusion
   └─ Action Head
        ↓
   Predicted Actions [B, action_dim]
        ↓
   [Loss Computation]
        ↓
   Backward Pass
        ↓
   Weight Update
```

### Inference Data Flow

```
Real-time Inputs
├─ Camera Image
├─ Text Instruction
└─ Robot State
        ↓
   [Preprocessing]
        ↓
   Normalized Inputs
        ↓
    [VLM Encoder]
        ↓
   Feature Embeddings
        ↓
   [VLA Framework]
        ↓
   Raw Action Prediction
        ↓
   [Post-processing]
   ├─ Denormalization
   ├─ Safety Checks
   └─ Action Smoothing
        ↓
   Robot Commands
        ↓
   Execute on Robot
```

### Memory Flow

Understanding where data lives:

```
┌─────────────────────────────────────────────┐
│              GPU Memory                      │
│                                              │
│  ┌────────────────────────────────────┐    │
│  │  VLM Weights (Frozen)              │    │
│  │  Size: ~2-8 GB                     │    │
│  └────────────────────────────────────┘    │
│                                              │
│  ┌────────────────────────────────────┐    │
│  │  Trainable Weights                 │    │
│  │  - Encoder: ~50-200 MB             │    │
│  │  - Action Head: ~100-500 MB        │    │
│  └────────────────────────────────────┘    │
│                                              │
│  ┌────────────────────────────────────┐    │
│  │  Activations & Gradients           │    │
│  │  Size: batch_size dependent        │    │
│  │  ~2-8 GB for batch_size=32         │    │
│  └────────────────────────────────────┘    │
│                                              │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│              CPU Memory                      │
│                                              │
│  ┌────────────────────────────────────┐    │
│  │  Dataset Cache (optional)          │    │
│  └────────────────────────────────────┘    │
│                                              │
│  ┌────────────────────────────────────┐    │
│  │  Data Loading Buffers              │    │
│  └────────────────────────────────────┘    │
│                                              │
└─────────────────────────────────────────────┘
```

## Key Design Patterns

### 1. Registry Pattern

**Purpose**: Dynamic component discovery and instantiation

**How it works**:

```python
# Components register themselves
@register_vlm(name="my-vlm", aliases=["mvlm"])
class MyVLM(AbstractVLM):
    pass

# Create from anywhere in code
vlm = create_vlm("my-vlm")

# List all registered components
all_vlms = list_vlms()
```

**Benefits**:
- No central import list to maintain
- Third-party plugins work automatically
- Type-safe component creation
- Discoverable at runtime

### 2. Abstract Base Classes

**Purpose**: Define clear interfaces and contracts

**How it works**:

```python
# Define interface
class AbstractVLM(ABC):
    @abstractmethod
    def forward(self, images, text):
        pass

# Implementations must provide all methods
class MyVLM(AbstractVLM):
    def forward(self, images, text):
        # Implementation here
        return embeddings
```

**Benefits**:
- Compile-time interface checking
- Clear documentation of requirements
- Consistent APIs across implementations
- Easy to test and mock

### 3. Composition Over Inheritance

**Purpose**: Build complex models from simple components

**How it works**:

```python
# Instead of deep inheritance hierarchies
class VLA:
    def __init__(self, vlm, encoder, action_head):
        self.vlm = vlm              # Composed component
        self.encoder = encoder       # Composed component
        self.action_head = action_head  # Composed component
    
    def forward(self, images, text, proprio):
        # Coordinate components
        features = self.vlm(images, text)
        proprio_features = self.encoder(proprio)
        actions = self.action_head(features, proprio_features)
        return actions
```

**Benefits**:
- Flexible component swapping
- Easier testing (test each component separately)
- Better code reuse
- Clearer dependencies

### 4. Config-Driven Design

**Purpose**: Separate configuration from code

**How it works**:

```python
# Configuration in YAML
model:
  vlm: qwen2-vl-2b
  action_dim: 7

# Code reads config
config = load_config("config.yaml")
model = create_vla(config.model.vlm, action_dim=config.model.action_dim)
```

**Benefits**:
- Reproducible experiments
- Easy hyperparameter tuning
- Version-controlled configurations
- No code changes needed

### 5. Plugin Architecture

**Purpose**: Extend without modifying core code

**How it works**:

```python
# Third-party plugin (separate package)
from librobot.models.vlm import AbstractVLM, register_vlm

@register_vlm(name="custom-vlm")
class CustomVLM(AbstractVLM):
    # Implementation
    pass

# In user code
import custom_vlm_plugin  # Auto-registers
from librobot.models import create_vlm

vlm = create_vlm("custom-vlm")  # Just works!
```

**Benefits**:
- No core code modification needed
- Community can contribute easily
- Clean separation of concerns
- Easy to distribute

## Extension Points

### Where Can You Customize?

LibroBot VLA is designed to be extended at multiple levels:

#### 1. Model Level

**Add New VLMs**:
```python
from librobot.models.vlm import AbstractVLM, register_vlm

@register_vlm(name="my-vlm")
class MyVLM(AbstractVLM):
    # Implement required methods
    pass
```

**Add New VLA Frameworks**:
```python
from librobot.models.frameworks import AbstractVLA, register_vla

@register_vla(name="my-vla")
class MyVLA(AbstractVLA):
    # Implement required methods
    pass
```

**Add New Action Heads**:
```python
from librobot.models.action_heads import AbstractActionHead, register_action_head

@register_action_head(name="my-head")
class MyActionHead(AbstractActionHead):
    # Implement required methods
    pass
```

See: [Adding Models Guide](adding_models.md)

#### 2. Data Level

**Add New Dataset Loaders**:
```python
from librobot.data.datasets import AbstractDataset, register_dataset

@register_dataset(name="my-dataset")
class MyDataset(AbstractDataset):
    # Implement required methods
    pass
```

**Add New Transforms**:
```python
from librobot.data.transforms import register_transform

@register_transform(name="my-transform")
class MyTransform:
    def __call__(self, sample):
        # Transform logic
        return transformed_sample
```

#### 3. Robot Level

**Add New Robot Interfaces**:
```python
from librobot.robots import AbstractRobot, register_robot

@register_robot(name="my-robot")
class MyRobot(AbstractRobot):
    # Implement required methods
    pass
```

See: [Adding Robots Guide](adding_robots.md)

#### 4. Training Level

**Add Custom Trainers**:
```python
from librobot.training import BaseTrainer

class CustomTrainer(BaseTrainer):
    def training_step(self, batch):
        # Custom training logic
        pass
```

**Add Training Callbacks**:
```python
from librobot.training.callbacks import Callback

class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs):
        # Custom logic at epoch end
        pass
```

### Extension Best Practices

1. **Follow Interfaces**: Implement all required methods from abstract base classes
2. **Register Components**: Use the registry decorators for discoverability
3. **Add Tests**: Write tests for your custom components
4. **Document**: Add docstrings and examples
5. **Type Hints**: Use type annotations for better IDE support
6. **Error Handling**: Provide clear error messages

### Example: Complete Custom VLA

Here's how all the pieces fit together:

```python
# my_custom_vla.py

from librobot.models.vlm import create_vlm
from librobot.models.frameworks import AbstractVLA, register_vla
from librobot.models.action_heads import create_action_head

@register_vla(
    name="my-custom-vla",
    description="My custom VLA implementation"
)
class MyCustomVLA(AbstractVLA):
    def __init__(self, vlm_name: str, action_dim: int, **kwargs):
        super().__init__()
        
        # Use existing VLM
        self.vlm = create_vlm(vlm_name, pretrained=True)
        
        # Custom encoder
        self.encoder = nn.Sequential(
            nn.Linear(14, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
        # Use existing action head
        self.action_head = create_action_head(
            "diffusion",
            action_dim=action_dim
        )
    
    def forward(self, images, text, proprio, actions=None):
        # Your custom logic
        features = self.vlm(images, text)
        proprio_features = self.encoder(proprio)
        combined = torch.cat([features, proprio_features], dim=-1)
        
        if actions is not None:
            # Training mode
            loss = self.action_head(combined, actions)
            return {"loss": loss}
        else:
            # Inference mode
            predicted_actions = self.action_head.predict(combined)
            return {"actions": predicted_actions}
    
    def predict_action(self, images, text, proprio):
        outputs = self.forward(images, text, proprio, actions=None)
        return outputs["actions"]
    
    def get_num_parameters(self, trainable_only=False):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

# Use it
from librobot.models.frameworks import create_vla

vla = create_vla("my-custom-vla", vlm_name="qwen2-vl-2b", action_dim=7)
```

## Performance Considerations

### Memory Usage

**Typical Memory Breakdown** (for batch_size=32):

| Component | Memory | Notes |
|-----------|--------|-------|
| VLM Weights (2B) | ~4 GB | Frozen, no gradients |
| VLM Weights (7B) | ~14 GB | Frozen, no gradients |
| Trainable Weights | ~200 MB | Encoder + Action Head |
| Activations | ~2-4 GB | Depends on batch size |
| Gradients | ~200 MB | Only trainable params |
| Data Batch | ~500 MB | Images + text + actions |
| **Total** | **~7-20 GB** | Depends on model size |

### Speed Optimization

**Techniques LibroBot Uses**:

1. **Frozen VLM**: Don't compute gradients for backbone
2. **Flash Attention**: 2-4x faster attention computation
3. **Mixed Precision**: FP16/BF16 for 2x speedup
4. **Gradient Checkpointing**: Trade compute for memory
5. **Compiled Models**: `torch.compile()` for faster execution

### Scaling

**Single GPU → Multi-GPU → Multi-Node**:

```
Single GPU (RTX 3090)
├─ Batch size: 16
├─ Training speed: ~1000 steps/hour
└─ Best for: Development, small models

4x GPUs (DDP)
├─ Effective batch size: 64
├─ Training speed: ~3500 steps/hour
└─ Best for: Most training runs

8x GPUs (FSDP)
├─ Effective batch size: 128
├─ Training speed: ~6000 steps/hour
└─ Best for: Large-scale training
```

## Next Steps

- **[Getting Started](getting_started.md)**: Basic usage
- **[Configuration Guide](configuration.md)**: Master configs
- **[Adding Robots](adding_robots.md)**: Integrate your robot
- **[Adding Models](adding_models.md)**: Extend with custom models
- **[Deployment Guide](deployment.md)**: Production deployment

For deep technical details, see:
- **[design/ARCHITECTURE.md](design/ARCHITECTURE.md)**: Complete technical architecture
- **[design/API_CONTRACTS.md](design/API_CONTRACTS.md)**: Detailed API specifications
- **[design/DESIGN_PRINCIPLES.md](design/DESIGN_PRINCIPLES.md)**: Core design principles

---

**Questions?** Open an issue or discussion on GitHub!
