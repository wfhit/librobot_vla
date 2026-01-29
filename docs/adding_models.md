# Guide: Adding New Models

This guide shows you how to extend LibroBot VLA with custom models. Whether you want to add a new VLM backend, create a custom VLA framework, or implement a novel action head, this guide covers everything you need.

## Table of Contents

- [Overview](#overview)
- [Adding a New VLM](#adding-a-new-vlm)
- [Adding a New VLA Framework](#adding-a-new-vla-framework)
- [Adding a New Action Head](#adding-a-new-action-head)
- [Using External Frameworks](#using-external-frameworks)
- [Best Practices](#best-practices)
- [Testing Your Model](#testing-your-model)
- [Troubleshooting](#troubleshooting)

## Overview

### Model Hierarchy

```
VLA Framework
    ├── VLM Backend (Vision-Language Model)
    │   ├── Vision Encoder
    │   └── Language Encoder
    │
    ├── Proprioception Encoder
    │   └── MLP/Transformer
    │
    ├── Feature Fusion
    │   └── Cross-Attention/FiLM/Concat
    │
    └── Action Head
        └── MLP/Diffusion/Flow/Transformer
```

### What Can You Add?

1. **VLM Backends**: New vision-language models (e.g., GPT-4V, Gemini)
2. **VLA Frameworks**: Complete architectures (e.g., your research model)
3. **Action Heads**: Novel action prediction methods
4. **Components**: Attention mechanisms, encoders, etc.

## Adding a New VLM

### Step 1: Understand the Interface

VLMs must implement `AbstractVLM`:

```python
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union

class AbstractVLM(ABC, nn.Module):
    """Abstract base class for Vision-Language Models."""
    
    @abstractmethod
    def forward(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, List[str]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Process images and text, return embeddings."""
        pass
    
    @abstractmethod
    def encode_image(
        self,
        images: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Encode images to embeddings."""
        pass
    
    @abstractmethod
    def encode_text(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> torch.Tensor:
        """Encode text to embeddings."""
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return embedding dimensionality."""
        pass
    
    @property
    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """Return model configuration."""
        pass
```

### Step 2: Implement Your VLM

Create `librobot/models/vlm/my_vlm.py`:

```python
"""
My Custom VLM
=============

Custom Vision-Language Model implementation.

Features:
    - Custom vision encoder
    - Custom language encoder
    - Efficient attention mechanism

Architecture:
    Vision: ViT-style encoder
    Language: Transformer decoder
    Fusion: Cross-attention
"""

from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

from librobot.models.vlm.base import AbstractVLM
from librobot.models.vlm.registry import register_vlm
from librobot.models.components import (
    FlashAttention,
    RMSNorm,
    SwiGLU,
    RotaryPositionalEncoding
)


@register_vlm(
    name="my-vlm",
    aliases=["mvlm", "custom-vlm"],
    description="My custom Vision-Language Model",
    tags=["custom", "efficient", "multimodal"]
)
class MyVLM(AbstractVLM):
    """
    My custom Vision-Language Model.
    
    This model combines a vision encoder with a language decoder
    for multimodal understanding.
    
    Args:
        hidden_dim: Hidden dimension size
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        image_size: Input image size (H, W)
        patch_size: Patch size for vision encoder
        vocab_size: Vocabulary size for text
        dropout: Dropout rate
        use_flash_attention: Whether to use Flash Attention 2
        
    Example:
        >>> vlm = MyVLM(hidden_dim=768, num_layers=12)
        >>> images = torch.randn(2, 3, 224, 224)
        >>> text = ["A cat", "A dog"]
        >>> outputs = vlm(images, text)
        >>> print(outputs["embeddings"].shape)
        torch.Size([2, seq_len, 768])
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        image_size: int = 224,
        patch_size: int = 16,
        vocab_size: int = 50000,
        dropout: float = 0.1,
        use_flash_attention: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.image_size = image_size
        self.patch_size = patch_size
        self.vocab_size = vocab_size
        
        # Vision encoder
        self.vision_encoder = self._build_vision_encoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers // 2,
            num_heads=num_heads,
            image_size=image_size,
            patch_size=patch_size,
            dropout=dropout,
        )
        
        # Vision projection
        self.vision_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Text embedding
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.text_pos_encoding = RotaryPositionalEncoding(hidden_dim // num_heads)
        
        # Language decoder
        self.language_decoder = self._build_language_decoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_flash_attention=use_flash_attention,
        )
        
        # Language modeling head
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Tie embeddings
        self.lm_head.weight = self.text_embedding.weight
        
        self._init_weights()
    
    def _build_vision_encoder(self, **kwargs):
        """Build vision encoder."""
        # Your custom vision encoder
        return VisionTransformer(**kwargs)
    
    def _build_language_decoder(self, **kwargs):
        """Build language decoder."""
        # Your custom language decoder
        return TransformerDecoder(**kwargs)
    
    def _init_weights(self):
        """Initialize weights with appropriate scheme."""
        def _init_module(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
        
        self.apply(_init_module)
    
    def forward(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, List[str]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: Images [B, C, H, W]
            text: Optional text input
            attention_mask: Optional attention mask [B, seq_len]
            return_loss: Whether to compute loss
            
        Returns:
            Dictionary with keys:
                - embeddings: Vision-language embeddings
                - logits: Language modeling logits (if text provided)
                - loss: Language modeling loss (if return_loss=True)
        """
        batch_size = images.shape[0]
        
        # Encode images
        vision_embeds = self.encode_image(images)  # [B, N, D]
        
        outputs = {"embeddings": vision_embeds}
        
        if text is not None:
            # Encode text
            if isinstance(text, str):
                text = [text]
            
            # Tokenize (you'd use a real tokenizer here)
            text_ids = self._tokenize(text)  # [B, T]
            text_embeds = self.encode_text(text_ids)  # [B, T, D]
            
            # Concatenate vision and text embeddings
            combined_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
            
            # Pass through language decoder
            decoder_output = self.language_decoder(
                combined_embeds,
                attention_mask=attention_mask
            )
            
            # Compute logits
            logits = self.lm_head(decoder_output)
            
            outputs["embeddings"] = combined_embeds
            outputs["logits"] = logits
            
            # Compute loss if requested
            if return_loss:
                # Shift for next-token prediction
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = text_ids[..., 1:].contiguous()
                
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, self.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                outputs["loss"] = loss
        
        return outputs
    
    def encode_image(
        self,
        images: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode images to embeddings.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            Image embeddings [B, num_patches, hidden_dim]
        """
        # Pass through vision encoder
        vision_features = self.vision_encoder(images)  # [B, N, D]
        
        # Project to common space
        vision_embeds = self.vision_proj(vision_features)
        
        return vision_embeds
    
    def encode_text(
        self,
        text: Union[str, List[str], torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        Encode text to embeddings.
        
        Args:
            text: Text input (strings or token IDs)
            
        Returns:
            Text embeddings [B, seq_len, hidden_dim]
        """
        # If text is strings, tokenize
        if isinstance(text, (str, list)):
            text = self._tokenize(text)
        
        # Embed tokens
        text_embeds = self.text_embedding(text)
        
        # Add positional encoding
        text_embeds = self.text_pos_encoding(text_embeds)
        
        return text_embeds
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        return self.hidden_dim
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "model_type": "my-vlm",
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "vocab_size": self.vocab_size,
        }
    
    def _tokenize(self, text: Union[str, List[str]]) -> torch.Tensor:
        """
        Tokenize text (placeholder - use real tokenizer).
        
        Args:
            text: Input text
            
        Returns:
            Token IDs [B, seq_len]
        """
        # TODO: Replace with actual tokenizer
        # For now, return dummy tokens
        if isinstance(text, str):
            text = [text]
        
        max_len = 77
        batch_size = len(text)
        tokens = torch.randint(0, self.vocab_size, (batch_size, max_len))
        return tokens


# Helper classes (implement these based on your needs)

class VisionTransformer(nn.Module):
    """Vision Transformer encoder."""
    
    def __init__(self, hidden_dim, num_layers, num_heads, image_size, patch_size, dropout):
        super().__init__()
        # Your implementation
        pass
    
    def forward(self, images):
        # Your implementation
        return features


class TransformerDecoder(nn.Module):
    """Transformer decoder."""
    
    def __init__(self, hidden_dim, num_layers, num_heads, dropout, use_flash_attention):
        super().__init__()
        # Your implementation
        pass
    
    def forward(self, embeddings, attention_mask=None):
        # Your implementation
        return output
```

### Step 3: Create Configuration

Create `configs/model/vlm/my_vlm.yaml`:

```yaml
# My Custom VLM Configuration

name: "my-vlm"
pretrained: false  # Set to true if you have pretrained weights
pretrained_path: null

# Architecture
hidden_dim: 768
num_layers: 12
num_heads: 12

# Vision
image_size: 224
patch_size: 16

# Language
vocab_size: 50000

# Training
dropout: 0.1
use_flash_attention: true

# Memory optimization
gradient_checkpointing: true
```

### Step 4: Test Your VLM

```python
from librobot.models.vlm import create_vlm
import torch

# Create VLM
vlm = create_vlm("my-vlm", hidden_dim=768)

# Test image encoding
images = torch.randn(2, 3, 224, 224)
image_embeds = vlm.encode_image(images)
print(f"Image embeddings: {image_embeds.shape}")

# Test text encoding
text = ["pick up the cup", "move forward"]
text_embeds = vlm.encode_text(text)
print(f"Text embeddings: {text_embeds.shape}")

# Test full forward pass
outputs = vlm(images, text)
print(f"Combined embeddings: {outputs['embeddings'].shape}")

print("✓ VLM tests passed!")
```

## Adding a New VLA Framework

### Step 1: Understand the Interface

VLA frameworks must implement `AbstractVLA`:

```python
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List

class AbstractVLA(ABC, nn.Module):
    """Abstract base class for VLA frameworks."""
    
    @abstractmethod
    def forward(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, List[str]]] = None,
        proprioception: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass. Returns loss during training."""
        pass
    
    @abstractmethod
    def predict_action(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, List[str]]] = None,
        proprioception: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Predict actions (inference mode)."""
        pass
    
    @abstractmethod
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Get parameter count."""
        pass
```

### Step 2: Implement Your Framework

Create `librobot/models/frameworks/my_framework.py`:

```python
"""
My Custom VLA Framework
=======================

Custom Vision-Language-Action framework.

Features:
    - Custom fusion mechanism
    - Novel action prediction
    - Efficient training

Architecture:
    VLM → Feature Fusion → Action Prediction
"""

from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

from librobot.models.frameworks.base import AbstractVLA
from librobot.models.frameworks.registry import register_vla
from librobot.models.vlm import AbstractVLM
from librobot.models.action_heads import create_action_head


@register_vla(
    name="my-framework",
    aliases=["my-vla", "custom-vla"],
    description="My custom VLA framework",
    tags=["custom", "research"]
)
class MyVLAFramework(AbstractVLA):
    """
    My custom VLA framework.
    
    Args:
        vlm: Vision-Language Model backbone
        action_dim: Action space dimension
        state_dim: Proprioception dimension
        action_head_type: Type of action head to use
        fusion_method: Feature fusion method
        **kwargs: Additional arguments
        
    Example:
        >>> from librobot.models import create_vlm, create_vla
        >>> vlm = create_vlm("qwen2-vl-2b")
        >>> vla = create_vla("my-framework", vlm=vlm, action_dim=7)
        >>> outputs = vla(images, text, proprio, actions)
    """
    
    def __init__(
        self,
        vlm: AbstractVLM,
        action_dim: int = 7,
        state_dim: int = 14,
        action_head_type: str = "diffusion",
        fusion_method: str = "cross_attention",
        hidden_dim: int = 512,
        freeze_vlm: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.vlm = vlm
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Freeze VLM if requested
        if freeze_vlm:
            for param in self.vlm.parameters():
                param.requires_grad = False
        
        # Get VLM embedding dimension
        vlm_dim = self.vlm.get_embedding_dim()
        
        # Proprioception encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Feature fusion
        if fusion_method == "cross_attention":
            self.fusion = CrossAttentionFusion(
                vlm_dim=vlm_dim,
                proprio_dim=hidden_dim,
                output_dim=hidden_dim,
            )
        elif fusion_method == "concat":
            self.fusion = ConcatFusion(
                vlm_dim=vlm_dim,
                proprio_dim=hidden_dim,
                output_dim=hidden_dim,
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Action head
        self.action_head = create_action_head(
            action_head_type,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            **kwargs
        )
    
    def forward(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, List[str]]] = None,
        proprioception: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: Input images [B, C, H, W]
            text: Text instructions
            proprioception: Robot state [B, state_dim]
            actions: Ground truth actions [B, action_dim] (for training)
            
        Returns:
            Dictionary with:
                - loss: Training loss (if actions provided)
                - other task-specific outputs
        """
        batch_size = images.shape[0]
        
        # Encode multimodal input through VLM
        with torch.no_grad() if not self.training else torch.enable_grad():
            vlm_outputs = self.vlm(images, text)
            vlm_features = vlm_outputs["embeddings"]  # [B, seq_len, D]
        
        # Encode proprioception
        if proprioception is not None:
            proprio_features = self.proprio_encoder(proprioception)  # [B, D]
        else:
            proprio_features = torch.zeros(
                batch_size, self.hidden_dim,
                device=images.device, dtype=images.dtype
            )
        
        # Fuse features
        fused_features = self.fusion(vlm_features, proprio_features)  # [B, D]
        
        # Predict actions or compute loss
        if actions is not None:
            # Training mode: compute loss
            action_outputs = self.action_head(fused_features, actions)
            return {
                "loss": action_outputs["loss"],
                "action_loss": action_outputs["loss"],
            }
        else:
            # Inference mode: predict actions
            predicted_actions = self.action_head.predict(fused_features)
            return {
                "actions": predicted_actions,
            }
    
    def predict_action(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, List[str]]] = None,
        proprioception: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Predict actions (inference mode).
        
        Args:
            images: Input images [B, C, H, W]
            text: Text instructions
            proprioception: Robot state [B, state_dim]
            
        Returns:
            Predicted actions [B, action_dim]
        """
        outputs = self.forward(
            images=images,
            text=text,
            proprioception=proprioception,
            actions=None,
            **kwargs
        )
        return outputs["actions"]
    
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """Get number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# Helper modules

class CrossAttentionFusion(nn.Module):
    """Cross-attention based feature fusion."""
    
    def __init__(self, vlm_dim, proprio_dim, output_dim):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            batch_first=True
        )
        self.vlm_proj = nn.Linear(vlm_dim, output_dim)
        self.proprio_proj = nn.Linear(proprio_dim, output_dim)
        self.output_proj = nn.Linear(output_dim, output_dim)
    
    def forward(self, vlm_features, proprio_features):
        # Project features
        vlm_proj = self.vlm_proj(vlm_features)  # [B, seq_len, D]
        proprio_proj = self.proprio_proj(proprio_features).unsqueeze(1)  # [B, 1, D]
        
        # Cross-attention
        fused, _ = self.cross_attn(
            query=proprio_proj,
            key=vlm_proj,
            value=vlm_proj
        )
        
        # Output projection
        output = self.output_proj(fused.squeeze(1))
        return output


class ConcatFusion(nn.Module):
    """Concatenation based feature fusion."""
    
    def __init__(self, vlm_dim, proprio_dim, output_dim):
        super().__init__()
        # Pool VLM features
        self.vlm_pool = nn.AdaptiveAvgPool1d(1)
        self.fusion_proj = nn.Sequential(
            nn.Linear(vlm_dim + proprio_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, vlm_features, proprio_features):
        # Pool VLM features
        vlm_pooled = vlm_features.mean(dim=1)  # [B, D]
        
        # Concatenate
        concat = torch.cat([vlm_pooled, proprio_features], dim=-1)
        
        # Project
        output = self.fusion_proj(concat)
        return output
```

### Step 3: Create Configuration

Create `configs/model/framework/my_framework.yaml`:

```yaml
# My Custom Framework Configuration

type: "my-framework"

# VLM configuration
vlm:
  name: "qwen2_vl_2b"
  pretrained: true
  freeze: true

# Framework settings
action_dim: 7
state_dim: 14
hidden_dim: 512

# Fusion
fusion_method: "cross_attention"  # or "concat"

# Action head
action_head:
  type: "diffusion"
  action_horizon: 16
  num_diffusion_steps: 100

# Training
freeze_vlm: true
```

## Adding a New Action Head

### Step 1: Implement Action Head

Create `librobot/models/action_heads/my_action_head.py`:

```python
"""My custom action head."""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn

from librobot.models.action_heads.base import AbstractActionHead
from librobot.models.action_heads.registry import register_action_head


@register_action_head(
    name="my-action-head",
    description="My custom action prediction head"
)
class MyActionHead(AbstractActionHead):
    """
    My custom action head.
    
    Args:
        action_dim: Action space dimension
        hidden_dim: Hidden dimension
        **kwargs: Additional arguments
    """
    
    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 512,
        **kwargs
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Your custom architecture
        self.network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(
        self,
        features: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: Input features [B, hidden_dim]
            actions: Ground truth actions [B, action_dim]
            
        Returns:
            Dictionary with loss (training) or predictions (inference)
        """
        predicted_actions = self.network(features)
        
        if actions is not None:
            # Training: compute loss
            loss = nn.functional.mse_loss(predicted_actions, actions)
            return {"loss": loss, "predicted_actions": predicted_actions}
        else:
            # Inference: return predictions
            return {"actions": predicted_actions}
    
    def predict(
        self,
        features: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Predict actions."""
        with torch.no_grad():
            outputs = self.forward(features, actions=None, **kwargs)
        return outputs["actions"]
    
    def get_action_dim(self) -> int:
        """Get action dimension."""
        return self.action_dim
```

## Using External Frameworks

### Integrating HuggingFace Models

```python
from transformers import AutoModel
from librobot.models.vlm import AbstractVLM, register_vlm

@register_vlm(name="huggingface-vlm")
class HuggingFaceVLM(AbstractVLM):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.model.config.hidden_size
    
    def forward(self, images, text=None, **kwargs):
        outputs = self.model(pixel_values=images, input_ids=text)
        return {"embeddings": outputs.last_hidden_state}
    
    def encode_image(self, images, **kwargs):
        vision_outputs = self.model.vision_model(images)
        return vision_outputs.last_hidden_state
    
    def encode_text(self, text, **kwargs):
        text_outputs = self.model.text_model(text)
        return text_outputs.last_hidden_state
    
    def get_embedding_dim(self):
        return self.hidden_dim
    
    @property
    def config(self):
        return {"model_name": self.model.config.name_or_path}
```

### Integrating Diffusers

```python
from diffusers import UNet2DConditionModel
from librobot.models.action_heads import AbstractActionHead, register_action_head

@register_action_head(name="diffusers-unet")
class DiffusersActionHead(AbstractActionHead):
    def __init__(self, action_dim, **kwargs):
        super().__init__()
        self.unet = UNet2DConditionModel(
            sample_size=action_dim,
            in_channels=action_dim,
            out_channels=action_dim,
            **kwargs
        )
    
    # Implement required methods...
```

## Best Practices

### 1. Follow Interfaces

```python
# ✓ Good: Implement all required methods
class MyVLM(AbstractVLM):
    def forward(self, images, text=None, **kwargs):
        pass
    def encode_image(self, images, **kwargs):
        pass
    def encode_text(self, text, **kwargs):
        pass
    def get_embedding_dim(self):
        pass
    @property
    def config(self):
        pass

# ✗ Bad: Missing required methods
class IncompleteVLM(AbstractVLM):
    def forward(self, images, text=None, **kwargs):
        pass
    # Missing other methods!
```

### 2. Add Documentation

```python
@register_vlm(
    name="my-vlm",
    description="Clear, concise description",
    tags=["tag1", "tag2"]
)
class MyVLM(AbstractVLM):
    """
    Detailed docstring with:
    - Purpose
    - Architecture
    - Args
    - Examples
    """
    pass
```

### 3. Handle Batch Dimensions

```python
# Always support batch processing
def forward(self, images, text=None):
    # images: [B, C, H, W]
    batch_size = images.shape[0]
    
    # Process batch...
    
    return outputs  # [B, ...]
```

### 4. Support Optional Inputs

```python
def forward(self, images, text=None, proprio=None):
    # Handle None gracefully
    if text is None:
        text_features = self.get_default_text_features(batch_size)
    else:
        text_features = self.encode_text(text)
```

### 5. Add Type Hints

```python
from typing import Dict, Any, Optional, Union, List
import torch

def forward(
    self,
    images: torch.Tensor,
    text: Optional[Union[str, List[str]]] = None,
    **kwargs: Any
) -> Dict[str, torch.Tensor]:
    pass
```

## Testing Your Model

### Unit Tests

```python
import pytest
import torch
from librobot.models import create_vlm, create_vla

class TestMyModel:
    def test_vlm_creation(self):
        vlm = create_vlm("my-vlm")
        assert vlm is not None
    
    def test_vlm_forward(self):
        vlm = create_vlm("my-vlm")
        images = torch.randn(2, 3, 224, 224)
        outputs = vlm(images)
        assert "embeddings" in outputs
    
    def test_vla_creation(self):
        vlm = create_vlm("my-vlm")
        vla = create_vla("my-framework", vlm=vlm, action_dim=7)
        assert vla is not None
    
    def test_vla_training(self):
        vlm = create_vlm("my-vlm")
        vla = create_vla("my-framework", vlm=vlm, action_dim=7)
        
        images = torch.randn(2, 3, 224, 224)
        text = ["pick up", "move"]
        proprio = torch.randn(2, 14)
        actions = torch.randn(2, 7)
        
        outputs = vla(images, text, proprio, actions)
        assert "loss" in outputs
        assert outputs["loss"].requires_grad
    
    def test_vla_inference(self):
        vlm = create_vlm("my-vlm")
        vla = create_vla("my-framework", vlm=vlm, action_dim=7)
        
        images = torch.randn(2, 3, 224, 224)
        text = ["pick up", "move"]
        proprio = torch.randn(2, 14)
        
        actions = vla.predict_action(images, text, proprio)
        assert actions.shape == (2, 7)
```

## Troubleshooting

### Issue: Registration Not Working

```python
# Make sure to import your module
import librobot.models.vlm.my_vlm  # Triggers registration

# Or in __init__.py
from librobot.models.vlm.my_vlm import MyVLM
```

### Issue: Shape Mismatches

```python
# Add debug prints
def forward(self, images, text=None):
    print(f"Images shape: {images.shape}")
    features = self.encode_image(images)
    print(f"Features shape: {features.shape}")
    return features
```

### Issue: CUDA Out of Memory

```python
# Use gradient checkpointing
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    return checkpoint(self.heavy_module, x)

# Or freeze backbone
for param in self.vlm.parameters():
    param.requires_grad = False
```

### Issue: Slow Training

```python
# Use mixed precision
from torch.cuda.amp import autocast

with autocast():
    outputs = model(inputs)

# Compile model
model = torch.compile(model)
```

---

**Next Steps:**

- [Testing Guide](testing.md): Write comprehensive tests
- [Deployment Guide](deployment.md): Deploy your model
- [Contributing](../CONTRIBUTING.md): Submit your model

For technical details, see [design/COMPONENT_GUIDE.md](design/COMPONENT_GUIDE.md).
