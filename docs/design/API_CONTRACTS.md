# LibroBot VLA API Contracts

## Table of Contents
- [Overview](#overview)
- [Abstract Base Classes](#abstract-base-classes)
- [Input/Output Formats](#inputoutput-formats)
- [Configuration Schemas](#configuration-schemas)
- [Registry Contracts](#registry-contracts)
- [Error Handling](#error-handling)
- [Versioning and Compatibility](#versioning-and-compatibility)

## Overview

This document defines the API contracts for LibroBot VLA framework. All implementations must adhere to these contracts to ensure compatibility and interoperability.

## Abstract Base Classes

### AbstractVLM

Vision-Language Model base class. All VLM implementations must inherit from this class and implement its abstract methods.

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn


class AbstractVLM(ABC, nn.Module):
    """
    Abstract base class for Vision-Language Models.
    
    All VLM implementations must inherit from this class and implement
    the required abstract methods.
    """
    
    @abstractmethod
    def forward(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, List[str]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the VLM.
        
        Args:
            images: Input images tensor [batch_size, channels, height, width]
                - dtype: torch.float32
                - range: [0, 1] (normalized) or [0, 255] (raw)
            text: Optional text input (string or list of strings)
                - Single string for batch processing same text
                - List of strings for per-sample text
            attention_mask: Optional attention mask for text [batch_size, seq_len]
                - dtype: torch.float32 or torch.bool
                - 1.0/True = attend, 0.0/False = ignore
            **kwargs: Additional model-specific arguments
            
        Returns:
            Dictionary containing:
                - 'embeddings': Vision-language embeddings [batch_size, seq_len, hidden_dim]
                    - dtype: torch.float32
                    - Must have gradient tracking enabled if model is trainable
                - Additional model-specific outputs (optional)
        
        Raises:
            ValueError: If input shapes are invalid
            RuntimeError: If forward pass fails
        """
        pass
    
    @abstractmethod
    def encode_image(
        self,
        images: torch.Tensor,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Encode images to embeddings.
        
        Args:
            images: Input images [batch_size, channels, height, width]
                - dtype: torch.float32
                - Typically expects normalized images
            **kwargs: Additional arguments
            
        Returns:
            Image embeddings [batch_size, num_patches, hidden_dim]
                - dtype: torch.float32
                - num_patches depends on image size and patch size
        
        Raises:
            ValueError: If image shape is invalid
        """
        pass
    
    @abstractmethod
    def encode_text(
        self,
        text: Union[str, List[str]],
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Encode text to embeddings.
        
        Args:
            text: Input text (string or list of strings)
                - Single string or list of strings
            **kwargs: Additional arguments
            
        Returns:
            Text embeddings [batch_size, seq_len, hidden_dim]
                - dtype: torch.float32
                - seq_len varies based on text length
        
        Raises:
            ValueError: If text is empty or invalid
        """
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """
        Get the dimensionality of the embeddings.
        
        Returns:
            int: Embedding dimension (e.g., 768, 1024, 1536)
        """
        pass
    
    @property
    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Returns:
            Dictionary containing model configuration:
                - name: Model name (str)
                - hidden_dim: Hidden dimension (int)
                - num_layers: Number of layers (int, optional)
                - Additional model-specific config
        """
        pass
    
    # Optional methods with default implementations
    
    def freeze(self) -> None:
        """
        Freeze all parameters in the VLM.
        
        Sets requires_grad=False for all parameters.
        """
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self) -> None:
        """
        Unfreeze all parameters in the VLM.
        
        Sets requires_grad=True for all parameters.
        """
        for param in self.parameters():
            param.requires_grad = True
    
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """
        Get number of parameters.
        
        Args:
            trainable_only: If True, count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def save_pretrained(self, path: str) -> None:
        """
        Save model weights.
        
        Args:
            path: Path to save directory
        """
        torch.save({
            'state_dict': self.state_dict(),
            'config': self.config,
        }, path)
    
    def load_pretrained(self, path: str) -> None:
        """
        Load model weights.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['state_dict'])
```

### AbstractVLA

Vision-Language-Action framework base class.

```python
class AbstractVLA(ABC, nn.Module):
    """
    Abstract base class for Vision-Language-Action frameworks.
    
    VLA frameworks combine VLMs with action prediction heads and additional
    components to create end-to-end systems for robot learning.
    """
    
    @abstractmethod
    def forward(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, List[str]]] = None,
        proprioception: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the VLA framework.
        
        Args:
            images: Input images [batch_size, channels, height, width]
                - dtype: torch.float32
                - Normalized to [0, 1] or standard ImageNet normalization
            text: Optional text instructions
                - String or list of strings
                - Natural language commands (e.g., "pick up the cup")
            proprioception: Optional proprioceptive state [batch_size, state_dim]
                - dtype: torch.float32
                - Robot joint positions, velocities, forces, etc.
            actions: Optional action targets for training [batch_size, action_dim]
                - dtype: torch.float32
                - Ground truth actions for supervised learning
            **kwargs: Additional arguments
                - task_id: Task identifier for multi-task learning
                - history: Historical observations for temporal context
            
        Returns:
            Dictionary containing:
                - 'actions': Predicted actions [batch_size, action_dim]
                    - dtype: torch.float32
                    - Continuous or discretized actions
                - 'loss': Training loss (if actions provided)
                    - dtype: torch.float32
                    - Scalar tensor with gradient
                - Additional framework-specific outputs
                    - 'action_dist': Action distribution parameters
                    - 'features': Intermediate features
                    - 'attention_weights': Attention weights
        
        Raises:
            ValueError: If input shapes are incompatible
            RuntimeError: If forward pass fails
        """
        pass
    
    @abstractmethod
    def predict_action(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, List[str]]] = None,
        proprioception: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Predict actions for inference.
        
        This method runs the model in evaluation mode and returns
        deterministic or sampled actions.
        
        Args:
            images: Input images [batch_size, channels, height, width]
            text: Optional text instructions
            proprioception: Optional proprioceptive state [batch_size, state_dim]
            **kwargs: Additional arguments
                - temperature: Sampling temperature (float)
                - deterministic: Use deterministic actions (bool)
            
        Returns:
            Predicted actions [batch_size, action_dim]
                - dtype: torch.float32
                - Ready for robot execution
        
        Raises:
            ValueError: If inputs are invalid
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses for training.
        
        Args:
            predictions: Model predictions
                - 'actions': Predicted actions
                - Additional framework-specific predictions
            targets: Ground truth targets
                - 'actions': Ground truth actions
                - Additional targets
            **kwargs: Additional loss computation arguments
                - loss_weights: Dictionary of loss weights
            
        Returns:
            Dictionary containing:
                - 'total_loss': Total weighted loss (required)
                    - dtype: torch.float32
                    - Scalar with gradient
                - Component losses (for logging)
                    - 'action_loss': Action prediction loss
                    - 'kl_loss': KL divergence (if applicable)
                    - Additional losses
        
        Raises:
            KeyError: If required keys are missing
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get framework configuration.
        
        Returns:
            Dictionary containing configuration:
                - name: Framework name (str)
                - vlm_config: VLM configuration (Dict)
                - action_dim: Action dimension (int)
                - state_dim: State dimension (int, optional)
                - Additional framework-specific config
        """
        pass
    
    # Optional methods
    
    def freeze_backbone(self) -> None:
        """Freeze VLM backbone parameters."""
        pass
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze VLM backbone parameters."""
        pass
    
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Get number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
```

### AbstractActionHead

Action prediction head base class.

```python
class AbstractActionHead(ABC, nn.Module):
    """
    Abstract base class for action prediction heads.
    """
    
    @abstractmethod
    def forward(
        self,
        features: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        **kwargs: Any
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of action head.
        
        Args:
            features: Input features [batch_size, feature_dim]
                - dtype: torch.float32
                - Features from VLA framework
            actions: Optional ground truth actions [batch_size, action_dim]
                - dtype: torch.float32
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing:
                - 'actions': Predicted actions [batch_size, action_dim]
                - 'loss': Loss (if actions provided)
                - Additional outputs
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        features: torch.Tensor,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Predict actions (inference mode).
        
        Args:
            features: Input features [batch_size, feature_dim]
            **kwargs: Additional arguments
            
        Returns:
            Predicted actions [batch_size, action_dim]
        """
        pass
```

### AbstractDataset

Dataset base class.

```python
class AbstractDataset(ABC, Dataset):
    """
    Abstract base class for datasets.
    """
    
    @abstractmethod
    def __len__(self) -> int:
        """Return dataset size."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single data sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - 'images': Image tensor [C, H, W] (torch.Tensor)
                - 'text': Text instruction (str)
                - 'proprioception': State [state_dim] (torch.Tensor)
                - 'actions': Action [action_dim] (torch.Tensor)
                - Additional data (optional)
        """
        pass
```

### AbstractRobot

Robot interface base class.

```python
class AbstractRobot(ABC):
    """
    Abstract base class for robot interfaces.
    """
    
    @abstractmethod
    def get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get current robot observation.
        
        Returns:
            Dictionary containing:
                - 'images': Camera images [num_cameras, H, W, C] (np.ndarray)
                - 'proprioception': Joint states [state_dim] (np.ndarray)
        """
        pass
    
    @abstractmethod
    def execute_action(self, action: np.ndarray) -> bool:
        """
        Execute an action on the robot.
        
        Args:
            action: Action to execute [action_dim] (np.ndarray)
            
        Returns:
            Success status (bool)
        """
        pass
    
    @abstractmethod
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset robot to initial state.
        
        Returns:
            Initial observation (same format as get_observation)
        """
        pass
```

## Input/Output Formats

### Image Tensors

```python
# Format: [batch_size, channels, height, width]
images = torch.randn(32, 3, 224, 224)

# Data type: float32
assert images.dtype == torch.float32

# Value range: [0, 1] (normalized) or [0, 255] (raw)
# Most models expect normalized images
images = images / 255.0  # If raw

# Standard ImageNet normalization (if required by model)
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
images = (images - mean) / std
```

### Text Instructions

```python
# Single text for all samples in batch
text = "pick up the red cup"

# Different text for each sample
text = [
    "pick up the red cup",
    "move forward slowly",
    "grasp the blue block",
]

# Length: list of strings with len == batch_size
assert len(text) == batch_size
```

### Proprioception/State

```python
# Format: [batch_size, state_dim]
proprioception = torch.randn(32, 14)

# Data type: float32
assert proprioception.dtype == torch.float32

# Common state dimensions:
# - 7: Single 7-DOF arm (positions)
# - 14: Single 7-DOF arm (positions + velocities)
# - 14: Dual 7-DOF arms (positions)
# - 28: Dual 7-DOF arms (positions + velocities)
```

### Actions

```python
# Format: [batch_size, action_dim]
actions = torch.randn(32, 7)

# Data type: float32
assert actions.dtype == torch.float32

# Common action dimensions:
# - 7: Single 7-DOF arm (joint positions or velocities)
# - 8: 7-DOF arm + gripper
# - 14: Dual 7-DOF arms
# - 16: Dual 7-DOF arms + grippers

# Continuous actions (typical range)
actions = torch.clamp(actions, -1.0, 1.0)

# Discrete actions (token IDs)
actions = torch.randint(0, 256, (32, 7))  # RT-2 style
```

### Model Outputs

```python
# Forward pass output
outputs = model(images, text, proprioception, actions)

# Required keys
assert "actions" in outputs  # Predicted actions
assert outputs["actions"].shape == (batch_size, action_dim)

# Optional keys
if "loss" in outputs:
    assert outputs["loss"].ndim == 0  # Scalar
    assert outputs["loss"].requires_grad  # Has gradient

if "features" in outputs:
    # Intermediate features
    pass

if "attention_weights" in outputs:
    # Attention visualization
    pass
```

## Configuration Schemas

### VLM Configuration

```yaml
# config/vlm/qwen2-vl.yaml
name: qwen2-vl-2b
model_size: 2b
hidden_dim: 1536
vision_encoder:
  patch_size: 14
  image_size: 224
language_model:
  num_layers: 24
  num_heads: 12
  vocab_size: 151936
pretrained: true
freeze_vision: false
freeze_language: false
```

### VLA Configuration

```yaml
# config/vla/groot.yaml
framework: groot
vlm:
  name: qwen2-vl-2b
  pretrained: true
  freeze: true
action_head:
  type: diffusion
  num_timesteps: 100
  noise_schedule: linear
state_encoder:
  type: mlp
  hidden_dim: 512
  num_layers: 3
fusion:
  type: film
action_dim: 7
state_dim: 14
```

### Training Configuration

```yaml
# config/training/default.yaml
optimizer:
  type: adamw
  lr: 1.0e-4
  weight_decay: 0.01
  betas: [0.9, 0.999]

scheduler:
  type: cosine
  warmup_steps: 1000
  min_lr: 1.0e-6

training:
  batch_size: 32
  epochs: 100
  gradient_clip: 1.0
  mixed_precision: true
  gradient_accumulation: 4

logging:
  log_every: 10
  eval_every: 1000
  save_every: 5000
```

## Registry Contracts

### Registration

All components must register themselves using the appropriate decorator:

```python
# VLM registration
@register_vlm(
    name="component-name",  # Required: primary name
    aliases=["alias1", "alias2"],  # Optional: alternative names
    description="Component description",  # Optional: for documentation
    tags=["tag1", "tag2"],  # Optional: for categorization
)
class MyVLM(AbstractVLM):
    pass

# VLA registration
@register_vla(name="framework-name", aliases=["fw"])
class MyVLA(AbstractVLA):
    pass

# Action head registration
@register_action_head(name="head-name", aliases=["hd"])
class MyActionHead(AbstractActionHead):
    pass
```

### Registry Access

```python
# List registered components
from librobot.models.vlm import list_vlms
print(list_vlms())  # ['qwen2-vl-2b', 'florence-2-base', ...]

# Get component class
from librobot.models.vlm import get_vlm
VLMClass = get_vlm("qwen2-vl-2b")

# Create component instance
from librobot.models.vlm import create_vlm
vlm = create_vlm("qwen2-vl-2b", hidden_dim=1536)

# Alias support
vlm = create_vlm("qwen2-vl")  # Using alias
vlm = create_vlm("qwen-vl")  # Another alias
```

### Registry Errors

```python
# Component not found
try:
    vlm = create_vlm("nonexistent-model")
except KeyError as e:
    print(f"Component not found: {e}")
    # Shows available components

# Invalid arguments
try:
    vlm = create_vlm("qwen2-vl-2b", invalid_arg=True)
except TypeError as e:
    print(f"Invalid argument: {e}")
```

## Error Handling

### Input Validation

```python
def forward(self, images, text, **kwargs):
    # Validate images
    if images.dim() != 4:
        raise ValueError(
            f"Expected 4D image tensor [B, C, H, W], got {images.dim()}D"
        )
    
    if images.shape[1] != 3:
        raise ValueError(
            f"Expected 3 channels (RGB), got {images.shape[1]}"
        )
    
    # Validate text
    if text is not None:
        if isinstance(text, list):
            if len(text) != images.shape[0]:
                raise ValueError(
                    f"Text list length ({len(text)}) must match "
                    f"batch size ({images.shape[0]})"
                )
    
    # Process...
```

### Error Messages

Error messages should be clear and actionable:

```python
# ✅ Good: Clear, specific, actionable
raise ValueError(
    f"Image size {images.shape[2:]} not supported. "
    f"Supported sizes: 224, 384, 448. "
    f"Use transform.Resize() to resize images."
)

# ❌ Bad: Vague, not helpful
raise ValueError("Invalid input")
```

### Exception Hierarchy

```python
# Base exception
class LibroBotError(Exception):
    """Base exception for LibroBot."""
    pass

# Specific exceptions
class ModelNotFoundError(LibroBotError):
    """Raised when model is not found in registry."""
    pass

class ConfigError(LibroBotError):
    """Raised when configuration is invalid."""
    pass

class InferenceError(LibroBotError):
    """Raised when inference fails."""
    pass
```

## Versioning and Compatibility

### API Version

```python
# librobot/version.py
__version__ = "0.1.0"
API_VERSION = "0.1"

# Semantic versioning: MAJOR.MINOR.PATCH
# - MAJOR: Breaking API changes
# - MINOR: New features, backward compatible
# - PATCH: Bug fixes, backward compatible
```

### Deprecation Warnings

```python
import warnings

def old_method(self):
    warnings.warn(
        "old_method() is deprecated and will be removed in version 0.2.0. "
        "Use new_method() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return self.new_method()
```

### Backward Compatibility

```python
# Support old argument names
def __init__(self, hidden_dim=None, hidden_size=None, **kwargs):
    # New name: hidden_dim
    # Old name: hidden_size (deprecated)
    if hidden_size is not None and hidden_dim is None:
        warnings.warn(
            "hidden_size is deprecated, use hidden_dim instead",
            DeprecationWarning
        )
        hidden_dim = hidden_size
    
    self.hidden_dim = hidden_dim or 768
```

### Version Compatibility Check

```python
def check_compatibility(component_version: str, required_version: str) -> bool:
    """
    Check if component version is compatible with required version.
    
    Args:
        component_version: Component version string (e.g., "0.1.0")
        required_version: Required version string (e.g., ">=0.1.0,<0.2.0")
        
    Returns:
        True if compatible, False otherwise
    """
    # Implementation using packaging.version
    from packaging.version import Version, parse
    from packaging.specifiers import SpecifierSet
    
    spec = SpecifierSet(required_version)
    return Version(component_version) in spec
```

## Type Annotations

All public APIs must have complete type annotations:

```python
from typing import Dict, List, Optional, Union, Tuple, Any
import torch

def forward(
    self,
    images: torch.Tensor,
    text: Optional[Union[str, List[str]]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs: Any
) -> Dict[str, torch.Tensor]:
    """Method with complete type annotations."""
    pass
```

## Testing Contracts

All components must have tests verifying:

1. Interface compliance
2. Input/output shapes
3. Gradient flow
4. Device compatibility
5. Edge cases

```python
def test_interface_compliance():
    """Test that component implements required interface."""
    from librobot.models.vlm.base import AbstractVLM
    from librobot.models.vlm.my_vlm import MyVLM
    
    assert issubclass(MyVLM, AbstractVLM)
    
    # Check all abstract methods are implemented
    vlm = MyVLM()
    assert hasattr(vlm, 'forward')
    assert hasattr(vlm, 'encode_image')
    assert hasattr(vlm, 'encode_text')
    assert hasattr(vlm, 'get_embedding_dim')
    assert hasattr(vlm, 'config')
```

## Conclusion

These API contracts ensure consistency, compatibility, and quality across the LibroBot VLA framework. All implementations must adhere to these contracts.

For more information:
- [Architecture](./ARCHITECTURE.md) - System architecture
- [Component Guide](./COMPONENT_GUIDE.md) - Adding components
- [Design Principles](./DESIGN_PRINCIPLES.md) - Design patterns
