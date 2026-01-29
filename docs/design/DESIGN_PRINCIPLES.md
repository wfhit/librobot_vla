# LibroBot VLA Design Principles

## Table of Contents
- [Overview](#overview)
- [Core Design Principles](#core-design-principles)
- [Registry Pattern](#registry-pattern)
- [Config-Driven Design](#config-driven-design)
- [Plugin Architecture](#plugin-architecture)
- [Type Safety](#type-safety)
- [Testing Strategy](#testing-strategy)
- [Extensibility](#extensibility)
- [Performance Considerations](#performance-considerations)

## Overview

LibroBot VLA is built on a foundation of well-established software engineering principles that prioritize modularity, maintainability, and extensibility. This document outlines the key design decisions and patterns used throughout the framework.

## Core Design Principles

### 1. Separation of Concerns

Each component has a single, well-defined responsibility:

```python
# ❌ Bad: Everything in one class
class VLA:
    def load_data(self): ...
    def train(self): ...
    def encode_image(self): ...
    def predict_action(self): ...
    def save_model(self): ...

# ✅ Good: Separated concerns
class VLM:
    def encode_image(self): ...
    def encode_text(self): ...

class VLA:
    def __init__(self, vlm, action_head): ...
    def predict_action(self): ...

class Trainer:
    def train(self, model, dataset): ...

class DataLoader:
    def load_data(self): ...
```

### 2. Interface Segregation

Clients shouldn't depend on interfaces they don't use:

```python
# ✅ Focused interfaces
class AbstractVLM(ABC):
    @abstractmethod
    def encode_image(self, images): ...
    
    @abstractmethod
    def encode_text(self, text): ...

class AbstractVLA(ABC):
    @abstractmethod
    def predict_action(self, images, text, proprioception): ...
    
    @abstractmethod
    def compute_loss(self, predictions, targets): ...
```

### 3. Dependency Inversion

Depend on abstractions, not concretions:

```python
# ✅ Depend on abstract base class
class GR00TVLA(AbstractVLA):
    def __init__(self, vlm: AbstractVLM, action_head: AbstractActionHead):
        self.vlm = vlm  # Any VLM implementation works
        self.action_head = action_head  # Any action head works
```

### 4. Open/Closed Principle

Open for extension, closed for modification:

```python
# ✅ Extend without modifying existing code
@register_vlm(name="my-custom-vlm")
class MyCustomVLM(AbstractVLM):
    """New VLM without changing core code"""
    pass

# Use immediately via registry
vlm = create_vlm("my-custom-vlm")
```

### 5. Don't Repeat Yourself (DRY)

Reuse code through composition and inheritance:

```python
# ✅ Shared components
from librobot.models.components import (
    FlashAttention,
    RMSNorm,
    RotaryPositionEmbedding
)

# Used across multiple models
class QwenVLModel(AbstractVLM):
    def __init__(self):
        self.attention = FlashAttention(...)
        self.norm = RMSNorm(...)
        self.rope = RotaryPositionEmbedding(...)

class InternVL(AbstractVLM):
    def __init__(self):
        self.attention = FlashAttention(...)  # Reuse
        self.norm = RMSNorm(...)  # Reuse
```

## Registry Pattern

The registry pattern is central to LibroBot's architecture, enabling runtime component discovery and instantiation.

### Design Goals

1. **Zero Core Modifications**: Add components without changing existing code
2. **Dynamic Discovery**: Components register themselves at import time
3. **Alias Support**: Multiple names for the same component
4. **Type Safety**: Maintain type information through registration
5. **Easy Testing**: Mock components for unit tests

### Implementation

#### Global Registry System

```python
class GlobalRegistry:
    """
    Central registry manager.
    
    Maintains separate registries for different component types
    (VLMs, VLA frameworks, action heads, etc.)
    """
    _registries: Dict[str, Registry] = {}
    
    @classmethod
    def get_registry(cls, name: str) -> Registry:
        """Get or create a registry by name."""
        if name not in cls._registries:
            cls._registries[name] = Registry(name)
        return cls._registries[name]
```

#### Component Registry

```python
class Registry:
    """
    Registry for a specific component type.
    
    Supports registration, retrieval, and instantiation of components.
    """
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type] = {}
        self._aliases: Dict[str, str] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(
        self,
        name: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        **metadata
    ):
        """
        Decorator to register a component.
        
        Args:
            name: Registration name (defaults to class name)
            aliases: Alternative names for the component
            **metadata: Additional metadata (tags, description, etc.)
        """
        def decorator(cls: Type) -> Type:
            # Use class name if no name provided
            reg_name = name or cls.__name__
            
            # Register the class
            self._registry[reg_name] = cls
            self._metadata[reg_name] = metadata
            
            # Register aliases
            if aliases:
                for alias in aliases:
                    self._aliases[alias] = reg_name
            
            return cls
        return decorator
    
    def get(self, name: str) -> Type:
        """Get registered component class."""
        # Resolve alias if necessary
        if name in self._aliases:
            name = self._aliases[name]
        
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"Component '{name}' not found in {self.name} registry. "
                f"Available: {available}"
            )
        
        return self._registry[name]
    
    def create(self, name: str, *args, **kwargs):
        """Create instance of registered component."""
        cls = self.get(name)
        return cls(*args, **kwargs)
    
    def list(self) -> List[str]:
        """List all registered component names."""
        return list(self._registry.keys())
    
    def list_with_aliases(self) -> Dict[str, List[str]]:
        """List components with their aliases."""
        result = {}
        for name in self._registry:
            aliases = [a for a, n in self._aliases.items() if n == name]
            result[name] = aliases
        return result
```

### Usage Pattern

#### Step 1: Create Registry

```python
# In librobot/models/vlm/registry.py
from librobot.utils.registry import GlobalRegistry

VLM_REGISTRY = GlobalRegistry.get_registry("vlm")

def register_vlm(name=None, aliases=None, **kwargs):
    return VLM_REGISTRY.register(name=name, aliases=aliases, **kwargs)

def get_vlm(name):
    return VLM_REGISTRY.get(name)

def create_vlm(name, *args, **kwargs):
    return VLM_REGISTRY.create(name, *args, **kwargs)

def list_vlms():
    return VLM_REGISTRY.list()
```

#### Step 2: Register Component

```python
# In librobot/models/vlm/qwen_vl.py
from librobot.models.vlm.registry import register_vlm
from librobot.models.vlm.base import AbstractVLM

@register_vlm(
    name="qwen2-vl-2b",
    aliases=["qwen2-vl", "qwen-vl"],
    description="Qwen2-VL 2B parameter model",
    tags=["multimodal", "vision-language"]
)
class QwenVLModel(AbstractVLM):
    def __init__(self, model_size="2b", **kwargs):
        super().__init__()
        # Implementation
```

#### Step 3: Use Component

```python
# Create via registry
from librobot.models.vlm import create_vlm

vlm = create_vlm("qwen2-vl-2b", pretrained=True)
vlm = create_vlm("qwen-vl", pretrained=True)  # Alias works too

# List available components
from librobot.models.vlm import list_vlms
print(list_vlms())  # ['qwen2-vl-2b', 'florence-2-base', ...]
```

### Registry Benefits

1. **Discoverability**: `list_vlms()` shows all available components
2. **Flexibility**: Switch components by changing a string
3. **Testing**: Easy to mock or replace components
4. **Plugin System**: Third-party plugins register themselves
5. **Configuration**: Config files can reference components by name

### Registry Best Practices

```python
# ✅ Good: Clear, descriptive names
@register_vlm(name="qwen2-vl-2b", aliases=["qwen2-vl"])
class QwenVLModel(AbstractVLM): ...

# ✅ Good: Helpful aliases
@register_vlm(
    name="florence-2-base",
    aliases=["florence-2", "florence", "f2"]
)
class FlorenceModel(AbstractVLM): ...

# ❌ Bad: Cryptic names
@register_vlm(name="q2v2")
class QwenVLModel(AbstractVLM): ...

# ❌ Bad: Too many aliases
@register_vlm(
    name="qwen2-vl-2b",
    aliases=["qwen", "qvl", "q2", "qv2", "qwen2", ...]
)
```

## Config-Driven Design

Configuration files drive model creation, training, and inference, enabling reproducibility and easy experimentation.

### Configuration Hierarchy

```yaml
# config/models/groot.yaml
model:
  framework: groot
  
  vlm:
    name: qwen2-vl-2b
    pretrained: true
    freeze_vision: true
    freeze_language: true
  
  action_head:
    type: diffusion
    num_timesteps: 100
    noise_schedule: linear
  
  state_encoder:
    type: mlp
    hidden_dim: 512
    num_layers: 3
    activation: gelu
  
  fusion:
    type: film
    conditioning_dim: 768

data:
  dataset: rlds
  path: /data/robot_demos
  batch_size: 32
  num_workers: 4
  
  transforms:
    - type: random_crop
      size: 224
    - type: normalize
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]

training:
  optimizer:
    type: adamw
    lr: 1e-4
    weight_decay: 0.01
  
  scheduler:
    type: cosine
    warmup_steps: 1000
  
  epochs: 100
  gradient_clip: 1.0
  mixed_precision: true
```

### Configuration Loading

```python
from librobot.utils.config import load_config
from librobot.models.vlm import create_vlm
from librobot.models.frameworks import create_vla

# Load config
config = load_config("configs/models/groot.yaml")

# Create components from config
vlm = create_vlm(**config["model"]["vlm"])
vla = create_vla(
    config["model"]["framework"],
    vlm=vlm,
    **config["model"]["action_head"]
)
```

### Configuration Benefits

1. **Reproducibility**: Exact configuration saved with results
2. **Experimentation**: Easy to try different hyperparameters
3. **Version Control**: Track configuration changes
4. **Sharing**: Share configurations with collaborators
5. **Automation**: Grid search and hyperparameter tuning

### Configuration Best Practices

```yaml
# ✅ Good: Hierarchical, clear structure
model:
  vlm:
    name: qwen2-vl-2b
    freeze: true

# ✅ Good: Descriptive names
training:
  optimizer:
    learning_rate: 1e-4

# ❌ Bad: Flat structure
vlm_name: qwen2-vl-2b
vlm_freeze: true
optimizer_lr: 1e-4

# ❌ Bad: Cryptic names
training:
  opt:
    lr: 1e-4
```

## Plugin Architecture

The plugin architecture allows third-party extensions without modifying core code.

### Plugin Structure

```python
# my_custom_plugin/my_vlm.py
from librobot.models.vlm import AbstractVLM, register_vlm

@register_vlm(name="my-custom-vlm")
class MyCustomVLM(AbstractVLM):
    """Custom VLM implementation."""
    
    def forward(self, images, text, **kwargs):
        # Custom implementation
        pass
    
    def encode_image(self, images):
        # Custom implementation
        pass
    
    def encode_text(self, text):
        # Custom implementation
        pass
    
    def get_embedding_dim(self):
        return 768
    
    @property
    def config(self):
        return {"name": "my-custom-vlm"}
```

### Plugin Installation

```python
# Option 1: Direct import
import my_custom_plugin.my_vlm  # Registers automatically

from librobot.models.vlm import create_vlm
vlm = create_vlm("my-custom-vlm")

# Option 2: Plugin system (future)
# librobot install-plugin my-custom-plugin
```

### Plugin Best Practices

1. **Follow Interfaces**: Implement all abstract methods
2. **Register Correctly**: Use appropriate registry
3. **Document Well**: Provide clear usage examples
4. **Test Thoroughly**: Include comprehensive tests
5. **Version Carefully**: Specify compatible LibroBot versions

## Type Safety

Type safety is enforced throughout the codebase using Python type hints and runtime checks.

### Type Hints

```python
from typing import Dict, List, Optional, Union, Tuple, Any
import torch
import torch.nn as nn

class VLM(AbstractVLM):
    def forward(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, List[str]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with complete type annotations.
        
        Args:
            images: Input images [B, C, H, W]
            text: Optional text input
            attention_mask: Optional attention mask [B, seq_len]
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with 'embeddings' and other outputs
        """
        pass
```

### Runtime Type Checking

```python
def validate_input(
    images: torch.Tensor,
    expected_shape: Tuple[int, ...]
) -> None:
    """Validate input tensor shape."""
    if images.dim() != len(expected_shape):
        raise ValueError(
            f"Expected {len(expected_shape)}D tensor, "
            f"got {images.dim()}D"
        )
    
    for i, (actual, expected) in enumerate(
        zip(images.shape, expected_shape)
    ):
        if expected != -1 and actual != expected:
            raise ValueError(
                f"Dimension {i}: expected {expected}, got {actual}"
            )
```

### Benefits of Type Safety

1. **Early Error Detection**: Catch bugs at development time
2. **IDE Support**: Autocomplete and inline documentation
3. **Self-Documentation**: Types clarify expected inputs/outputs
4. **Refactoring Safety**: Type errors caught automatically
5. **Better Testing**: Generate test cases from type signatures

### Type Checking Tools

```bash
# Static type checking
mypy librobot/

# Runtime type checking
python -m pytest --mypy librobot/

# Type coverage
mypy --html-report mypy_report librobot/
```

## Testing Strategy

Comprehensive testing ensures code quality and prevents regressions.

### Test Pyramid

```
        /\
       /  \
      / E2E\         <- End-to-End (Few, slow, comprehensive)
     /------\
    /  Int.  \       <- Integration (Some, medium speed)
   /----------\
  /   Unit     \     <- Unit (Many, fast, focused)
 /--------------\
```

### Unit Tests

Test individual components in isolation:

```python
import pytest
import torch
from librobot.models.vlm.qwen_vl import QwenVLModel

class TestQwenVL:
    def test_initialization(self):
        """Test model initialization."""
        model = QwenVLModel(model_size="2b")
        assert model.get_embedding_dim() == 1536
    
    def test_encode_image(self):
        """Test image encoding."""
        model = QwenVLModel(model_size="2b")
        images = torch.randn(2, 3, 224, 224)
        embeddings = model.encode_image(images)
        
        assert embeddings.shape[0] == 2  # Batch size
        assert embeddings.shape[-1] == 1536  # Hidden dim
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = QwenVLModel(model_size="2b")
        images = torch.randn(2, 3, 224, 224)
        
        outputs = model(images)
        assert "embeddings" in outputs
        assert outputs["embeddings"].requires_grad
```

### Integration Tests

Test component interactions:

```python
def test_vla_framework_integration():
    """Test VLA framework with real VLM."""
    from librobot.models.vlm import create_vlm
    from librobot.models.frameworks import create_vla
    
    # Create components
    vlm = create_vlm("qwen2-vl-2b", pretrained=False)
    vla = create_vla("groot", vlm=vlm, action_dim=7)
    
    # Test forward pass
    images = torch.randn(2, 3, 224, 224)
    text = ["pick up the cup", "move forward"]
    proprioception = torch.randn(2, 14)
    actions = torch.randn(2, 7)
    
    outputs = vla(images, text, proprioception, actions)
    
    assert "loss" in outputs
    assert "actions" in outputs
    assert outputs["loss"].requires_grad
```

### End-to-End Tests

Test complete workflows:

```python
def test_training_pipeline():
    """Test full training pipeline."""
    # Setup
    config = load_config("configs/test.yaml")
    model = create_model_from_config(config)
    dataset = create_dataset_from_config(config)
    trainer = Trainer(model, dataset, config)
    
    # Train for a few steps
    trainer.train(num_steps=10)
    
    # Verify checkpoints created
    assert os.path.exists("checkpoints/step_10.pth")
    
    # Test inference
    predictor = Predictor(model)
    actions = predictor.predict(test_images, test_text)
    assert actions.shape == (batch_size, action_dim)
```

### Test Organization

```
tests/
├── unit/
│   ├── test_vlm/
│   │   ├── test_qwen_vl.py
│   │   ├── test_florence.py
│   │   └── ...
│   ├── test_frameworks/
│   │   ├── test_groot.py
│   │   ├── test_pi0.py
│   │   └── ...
│   └── test_components/
├── integration/
│   ├── test_vla_integration.py
│   └── test_training_integration.py
└── e2e/
    ├── test_training_pipeline.py
    └── test_inference_pipeline.py
```

### Testing Best Practices

1. **Fast Tests**: Unit tests should run in milliseconds
2. **Isolated**: Tests don't depend on each other
3. **Reproducible**: Use fixed random seeds
4. **Clear Names**: Test names describe what they test
5. **Arrange-Act-Assert**: Clear test structure
6. **Mock External Dependencies**: Use mocks for external systems

```python
# ✅ Good: Clear, fast, isolated
def test_encode_image_shape():
    """Test encode_image returns correct shape."""
    model = QwenVLModel(model_size="2b")
    images = torch.randn(4, 3, 224, 224)
    embeddings = model.encode_image(images)
    assert embeddings.shape == (4, 196, 1536)

# ❌ Bad: Slow, coupled, unclear
def test_model():
    """Test model."""
    model = load_pretrained_model()  # Slow
    images = load_real_images()  # External dependency
    result = model(images)
    assert result is not None  # Weak assertion
```

## Extensibility

The framework is designed for easy extension without modifying core code.

### Extension Points

1. **VLM Backends**: Add new vision-language models
2. **VLA Frameworks**: Create new VLA architectures
3. **Action Heads**: Implement new action prediction mechanisms
4. **Datasets**: Support new data formats
5. **Robots**: Add hardware interfaces
6. **Losses**: Create custom loss functions
7. **Callbacks**: Add training callbacks

### Extension Example

Adding a new VLM:

```python
# 1. Implement interface
from librobot.models.vlm import AbstractVLM, register_vlm

@register_vlm(name="my-vlm")
class MyVLM(AbstractVLM):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Initialize layers
    
    def forward(self, images, text, **kwargs):
        # Implementation
        pass
    
    def encode_image(self, images):
        # Implementation
        pass
    
    def encode_text(self, text):
        # Implementation
        pass
    
    def get_embedding_dim(self):
        return self.hidden_dim
    
    @property
    def config(self):
        return {"hidden_dim": self.hidden_dim}

# 2. Use immediately
from librobot.models.vlm import create_vlm
vlm = create_vlm("my-vlm", hidden_dim=1024)
```

### Composition Over Inheritance

Prefer composition for flexibility:

```python
# ✅ Good: Composition
class VLA:
    def __init__(
        self,
        vlm: AbstractVLM,
        action_head: AbstractActionHead,
        state_encoder: AbstractEncoder
    ):
        self.vlm = vlm
        self.action_head = action_head
        self.state_encoder = state_encoder

# ❌ Less flexible: Deep inheritance
class VLA(VLM, ActionHead, StateEncoder):
    pass
```

## Performance Considerations

### Memory Optimization

```python
# Gradient checkpointing
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    x = checkpoint(self.layer1, x)  # Save memory
    x = checkpoint(self.layer2, x)
    return x

# Mixed precision training
from torch.cuda.amp import autocast

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)
```

### Compute Optimization

```python
# Flash Attention for efficiency
from librobot.models.components import FlashAttention

self.attention = FlashAttention(
    hidden_dim=768,
    num_heads=12,
    use_flash=True  # 2-4x faster
)

# Operator fusion
self.fused_mlp = FusedMLP(hidden_dim=768)  # Faster than separate ops
```

### Profiling

```python
# Profile code
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
) as prof:
    outputs = model(inputs)

print(prof.key_averages().table())
```

## Conclusion

LibroBot VLA's design principles create a robust, maintainable, and extensible framework. The registry pattern, config-driven design, and strong type safety enable rapid experimentation while maintaining code quality.

For more details:
- [Architecture](./ARCHITECTURE.md) - System architecture
- [Component Guide](./COMPONENT_GUIDE.md) - Adding components
- [API Contracts](./API_CONTRACTS.md) - Interface definitions
