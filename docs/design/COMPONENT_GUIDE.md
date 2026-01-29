# LibroBot VLA Component Guide

## Table of Contents
- [Overview](#overview)
- [Adding a New VLM](#adding-a-new-vlm)
- [Adding a New VLA Framework](#adding-a-new-vla-framework)
- [Adding a New Action Head](#adding-a-new-action-head)
- [Adding a New Dataset](#adding-a-new-dataset)
- [Adding a New Robot Interface](#adding-a-new-robot-interface)
- [Testing Your Component](#testing-your-component)
- [Documentation Requirements](#documentation-requirements)
- [Publishing Your Plugin](#publishing-your-plugin)

## Overview

This guide provides step-by-step instructions for extending the LibroBot VLA framework with new components. Each section includes complete code examples and best practices.

## Adding a New VLM

### Step 1: Understand the Interface

First, review the `AbstractVLM` interface:

```python
# librobot/models/vlm/base.py
class AbstractVLM(ABC, nn.Module):
    @abstractmethod
    def forward(self, images, text, attention_mask, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass returning embeddings."""
        pass
    
    @abstractmethod
    def encode_image(self, images, **kwargs) -> torch.Tensor:
        """Encode images to embeddings."""
        pass
    
    @abstractmethod
    def encode_text(self, text, **kwargs) -> torch.Tensor:
        """Encode text to embeddings."""
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get embedding dimensionality."""
        pass
    
    @property
    @abstractmethod
    def config(self) -> Dict[str, Any]:
        """Get model configuration."""
        pass
```

### Step 2: Create Implementation File

Create a new file in `librobot/models/vlm/`:

```python
# librobot/models/vlm/my_vlm.py
"""My custom VLM implementation."""

from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

from librobot.models.vlm.base import AbstractVLM
from librobot.models.vlm.registry import register_vlm
from librobot.models.components import FlashAttention, RMSNorm


@register_vlm(
    name="my-vlm-base",
    aliases=["my-vlm", "mvlm"],
    description="My custom Vision-Language Model",
    tags=["custom", "multimodal"]
)
class MyVLM(AbstractVLM):
    """
    My custom Vision-Language Model.
    
    This model combines a vision encoder and language decoder
    for multimodal understanding.
    
    Args:
        hidden_dim: Hidden dimension size
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        image_size: Input image size
        patch_size: Patch size for vision encoder
        vocab_size: Vocabulary size for text
        **kwargs: Additional arguments
    
    Example:
        >>> vlm = MyVLM(hidden_dim=768, num_layers=12)
        >>> images = torch.randn(2, 3, 224, 224)
        >>> text = ["A photo of a cat", "A dog playing"]
        >>> outputs = vlm(images, text)
        >>> print(outputs["embeddings"].shape)
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
        self.vision_encoder = VisionEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers // 2,
            num_heads=num_heads,
            image_size=image_size,
            patch_size=patch_size,
        )
        
        # Vision-language projection
        self.vision_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Text embedding
        self.text_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Language decoder
        self.language_decoder = LanguageDecoder(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # LM head
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, List[str]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_loss: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the VLM.
        
        Args:
            images: Input images [B, C, H, W]
            text: Optional text input
            attention_mask: Optional attention mask [B, seq_len]
            return_loss: Whether to compute loss
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing:
                - embeddings: Vision-language embeddings [B, seq_len, D]
                - logits: Language modeling logits (if text provided)
                - loss: Language modeling loss (if return_loss=True)
        """
        # Encode images
        vision_embeds = self.encode_image(images)  # [B, N, D]
        
        outputs = {"embeddings": vision_embeds}
        
        if text is not None:
            # Encode text
            text_embeds = self.encode_text(text)  # [B, T, D]
            
            # Concatenate vision and text
            combined_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
            
            # Decode
            decoder_output = self.language_decoder(
                combined_embeds,
                attention_mask=attention_mask
            )
            
            # Compute logits
            logits = self.lm_head(decoder_output)
            
            outputs["embeddings"] = combined_embeds
            outputs["logits"] = logits
            
            # Compute loss if needed
            if return_loss and text is not None:
                # Shift for language modeling
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = text[..., 1:].contiguous()
                
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, self.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                outputs["loss"] = loss
        
        return outputs
    
    def encode_image(self, images: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Encode images to embeddings.
        
        Args:
            images: Input images [B, C, H, W]
            **kwargs: Additional arguments
            
        Returns:
            Image embeddings [B, num_patches, hidden_dim]
        """
        vision_features = self.vision_encoder(images)
        vision_embeds = self.vision_projection(vision_features)
        return vision_embeds
    
    def encode_text(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> torch.Tensor:
        """
        Encode text to embeddings.
        
        Args:
            text: Input text (string or list of strings)
            **kwargs: Additional arguments
            
        Returns:
            Text embeddings [B, seq_len, hidden_dim]
        """
        # Convert text to token IDs (simplified)
        if isinstance(text, str):
            text = [text]
        
        # TODO: Use proper tokenizer
        # For now, create dummy token IDs
        batch_size = len(text)
        seq_len = 32
        token_ids = torch.randint(
            0, self.vocab_size,
            (batch_size, seq_len),
            device=next(self.parameters()).device
        )
        
        # Embed tokens
        text_embeds = self.text_embedding(token_ids)
        return text_embeds
    
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings."""
        return self.hidden_dim
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "name": "my-vlm",
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "vocab_size": self.vocab_size,
        }
    
    def generate(
        self,
        images: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text from images.
        
        Args:
            images: Input images [B, C, H, W]
            max_length: Maximum generation length
            temperature: Sampling temperature
            **kwargs: Additional arguments
            
        Returns:
            Generated token IDs [B, max_length]
        """
        batch_size = images.shape[0]
        device = images.device
        
        # Encode images
        vision_embeds = self.encode_image(images)
        
        # Initialize with BOS token
        generated = torch.zeros(
            batch_size, 0,
            dtype=torch.long,
            device=device
        )
        
        # Autoregressive generation
        for _ in range(max_length):
            # Get text embeddings
            if generated.shape[1] > 0:
                text_embeds = self.text_embedding(generated)
                combined = torch.cat([vision_embeds, text_embeds], dim=1)
            else:
                combined = vision_embeds
            
            # Decode
            output = self.language_decoder(combined)
            logits = self.lm_head(output[:, -1, :])  # Last token
            
            # Sample next token
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated


class VisionEncoder(nn.Module):
    """Vision encoder using patch embeddings and transformer."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        image_size: int,
        patch_size: int,
    ):
        super().__init__()
        
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, hidden_dim) * 0.02
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(hidden_dim)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Patch embedding
        x = self.patch_embed(images)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.norm(x)
        return x


class LanguageDecoder(nn.Module):
    """Language decoder transformer."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        x = self.norm(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with attention and FFN."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.attention = FlashAttention(hidden_dim, num_heads)
        self.norm1 = RMSNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = RMSNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Attention
        x = x + self.attention(self.norm1(x), attention_mask=attention_mask)
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x
```

### Step 3: Register Your VLM

The `@register_vlm` decorator automatically registers your VLM. Now you can use it:

```python
from librobot.models.vlm import create_vlm, list_vlms

# List available VLMs
print(list_vlms())  # Includes "my-vlm-base"

# Create your VLM
vlm = create_vlm("my-vlm", hidden_dim=1024)
vlm = create_vlm("mvlm", hidden_dim=1024)  # Using alias

# Use it
images = torch.randn(2, 3, 224, 224)
outputs = vlm(images)
```

### Step 4: Add to `__init__.py`

Update `librobot/models/vlm/__init__.py`:

```python
from .my_vlm import MyVLM

__all__ = [
    # ... existing exports ...
    'MyVLM',
]
```

## Adding a New VLA Framework

### Step 1: Understand the Interface

Review the `AbstractVLA` interface:

```python
# librobot/models/frameworks/base.py
class AbstractVLA(ABC, nn.Module):
    @abstractmethod
    def forward(self, images, text, proprioception, actions, **kwargs) -> Dict:
        """Forward pass with loss computation."""
        pass
    
    @abstractmethod
    def predict_action(self, images, text, proprioception, **kwargs) -> torch.Tensor:
        """Predict actions for inference."""
        pass
    
    @abstractmethod
    def compute_loss(self, predictions, targets, **kwargs) -> Dict:
        """Compute training losses."""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get framework configuration."""
        pass
```

### Step 2: Create Implementation

Create `librobot/models/frameworks/my_framework.py`:

```python
"""My custom VLA framework."""

from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn

from librobot.models.frameworks.base import AbstractVLA
from librobot.models.frameworks.registry import register_vla
from librobot.models.vlm.base import AbstractVLM
from librobot.models.action_heads.base import AbstractActionHead


@register_vla(
    name="my-framework",
    aliases=["my-vla", "mf"],
    description="My custom VLA framework",
    tags=["custom", "robot-learning"]
)
class MyFramework(AbstractVLA):
    """
    My custom VLA framework.
    
    This framework combines VLM, state encoding, and action prediction
    in a novel architecture.
    
    Args:
        vlm: Vision-Language Model backbone
        action_head: Action prediction head
        state_dim: Proprioception state dimension
        action_dim: Action dimension
        hidden_dim: Hidden dimension
        **kwargs: Additional arguments
    
    Example:
        >>> from librobot.models.vlm import create_vlm
        >>> vlm = create_vlm("qwen2-vl-2b")
        >>> vla = MyFramework(vlm=vlm, action_dim=7, state_dim=14)
        >>> 
        >>> images = torch.randn(2, 3, 224, 224)
        >>> text = ["pick up the cup", "move forward"]
        >>> proprioception = torch.randn(2, 14)
        >>> actions = torch.randn(2, 7)
        >>> 
        >>> outputs = vla(images, text, proprioception, actions)
        >>> print(outputs["loss"])
    """
    
    def __init__(
        self,
        vlm: AbstractVLM,
        action_head: Optional[AbstractActionHead] = None,
        state_dim: int = 14,
        action_dim: int = 7,
        hidden_dim: int = 512,
        freeze_vlm: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.vlm = vlm
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Freeze VLM if requested
        if freeze_vlm:
            self.freeze_backbone()
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Fusion layer
        vlm_dim = vlm.get_embedding_dim()
        self.fusion = nn.Sequential(
            nn.Linear(vlm_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Action head
        if action_head is None:
            self.action_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
            )
        else:
            self.action_head = action_head
    
    def forward(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, List[str]]] = None,
        proprioception: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the VLA framework.
        
        Args:
            images: Input images [B, C, H, W]
            text: Optional text instructions
            proprioception: Optional state [B, state_dim]
            actions: Optional ground truth actions [B, action_dim]
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing:
                - actions: Predicted actions [B, action_dim]
                - loss: Training loss (if actions provided)
        """
        batch_size = images.shape[0]
        
        # Encode vision-language
        vlm_outputs = self.vlm(images, text)
        vlm_features = vlm_outputs["embeddings"]
        
        # Pool VLM features (mean pooling over sequence)
        vlm_pooled = vlm_features.mean(dim=1)  # [B, vlm_dim]
        
        # Encode state if provided
        if proprioception is not None:
            state_features = self.state_encoder(proprioception)
        else:
            state_features = torch.zeros(
                batch_size, self.hidden_dim,
                device=images.device
            )
        
        # Fuse features
        combined = torch.cat([vlm_pooled, state_features], dim=1)
        fused_features = self.fusion(combined)
        
        # Predict actions
        predicted_actions = self.action_head(fused_features)
        
        outputs = {"actions": predicted_actions}
        
        # Compute loss if ground truth actions provided
        if actions is not None:
            loss = self.compute_loss(
                {"actions": predicted_actions},
                {"actions": actions}
            )
            outputs.update(loss)
        
        return outputs
    
    def predict_action(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, List[str]]] = None,
        proprioception: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Predict actions for inference.
        
        Args:
            images: Input images [B, C, H, W]
            text: Optional text instructions
            proprioception: Optional state [B, state_dim]
            **kwargs: Additional arguments
            
        Returns:
            Predicted actions [B, action_dim]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images, text, proprioception)
            return outputs["actions"]
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing losses
        """
        # MSE loss on actions
        action_loss = nn.functional.mse_loss(
            predictions["actions"],
            targets["actions"]
        )
        
        return {
            "loss": action_loss,
            "action_loss": action_loss,
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get framework configuration."""
        return {
            "name": "my-framework",
            "vlm_config": self.vlm.config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
        }
    
    def freeze_backbone(self) -> None:
        """Freeze VLM parameters."""
        for param in self.vlm.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze VLM parameters."""
        for param in self.vlm.parameters():
            param.requires_grad = True
```

### Step 3: Use Your Framework

```python
from librobot.models.vlm import create_vlm
from librobot.models.frameworks import create_vla

# Create VLM
vlm = create_vlm("qwen2-vl-2b", pretrained=True)

# Create your framework
vla = create_vla("my-framework", vlm=vlm, action_dim=7, state_dim=14)

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

## Adding a New Action Head

Action heads implement different action prediction mechanisms (MLP, diffusion, flow matching, etc.).

### Step 1: Create Implementation

Create `librobot/models/action_heads/my_action_head.py`:

```python
"""My custom action head."""

from typing import Dict, Optional
import torch
import torch.nn as nn

from librobot.models.action_heads.base import AbstractActionHead
from librobot.models.action_heads.registry import register_action_head


@register_action_head(name="my-action-head", aliases=["mah"])
class MyActionHead(AbstractActionHead):
    """
    My custom action prediction head.
    
    Args:
        input_dim: Input feature dimension
        action_dim: Action dimension
        hidden_dim: Hidden dimension
        num_layers: Number of MLP layers
    """
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Build MLP
        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, action_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        features: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: Input features [B, input_dim]
            actions: Optional ground truth actions [B, action_dim]
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with predicted actions and optional loss
        """
        predicted_actions = self.mlp(features)
        
        outputs = {"actions": predicted_actions}
        
        if actions is not None:
            loss = nn.functional.mse_loss(predicted_actions, actions)
            outputs["loss"] = loss
        
        return outputs
    
    def predict(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        """Predict actions."""
        with torch.no_grad():
            outputs = self.forward(features)
            return outputs["actions"]
```

## Adding a New Dataset

Datasets load and preprocess training/evaluation data.

### Step 1: Create Implementation

Create `librobot/data/datasets/my_dataset.py`:

```python
"""My custom dataset."""

from typing import Dict, Optional
import torch
from torch.utils.data import Dataset

from librobot.data.datasets.base import AbstractDataset
from librobot.data.datasets.registry import register_dataset


@register_dataset(name="my-dataset", aliases=["md"])
class MyDataset(AbstractDataset):
    """
    My custom dataset.
    
    Args:
        data_path: Path to dataset
        split: Dataset split ('train', 'val', 'test')
        transform: Optional image transforms
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform: Optional[callable] = None,
    ):
        super().__init__()
        
        self.data_path = data_path
        self.split = split
        self.transform = transform
        
        # Load dataset
        self.data = self._load_data()
    
    def _load_data(self):
        """Load dataset from disk."""
        # TODO: Implement data loading
        # For example, load from HDF5, RLDS, or custom format
        pass
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single data sample.
        
        Returns:
            Dictionary containing:
                - images: Image tensor [C, H, W]
                - text: Text instruction (string)
                - proprioception: State [state_dim]
                - actions: Action [action_dim]
        """
        sample = self.data[idx]
        
        # Extract fields
        image = sample["image"]
        text = sample["text"]
        proprioception = sample["proprioception"]
        action = sample["action"]
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
        
        return {
            "images": image,
            "text": text,
            "proprioception": torch.tensor(proprioception, dtype=torch.float32),
            "actions": torch.tensor(action, dtype=torch.float32),
        }
```

### Step 2: Use Your Dataset

```python
from librobot.data.datasets import create_dataset
from torch.utils.data import DataLoader

# Create dataset
dataset = create_dataset(
    "my-dataset",
    data_path="/path/to/data",
    split="train"
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Use in training
for batch in dataloader:
    images = batch["images"]
    text = batch["text"]
    proprioception = batch["proprioception"]
    actions = batch["actions"]
    
    outputs = model(images, text, proprioception, actions)
    loss = outputs["loss"]
    loss.backward()
```

## Adding a New Robot Interface

Robot interfaces abstract hardware communication.

### Step 1: Create Implementation

Create `librobot/robots/my_robot.py`:

```python
"""My custom robot interface."""

from typing import Dict, List, Optional
import numpy as np

from librobot.robots.base import AbstractRobot
from librobot.robots.registry import register_robot


@register_robot(name="my-robot", aliases=["mr"])
class MyRobot(AbstractRobot):
    """
    Interface for my custom robot.
    
    Args:
        ip_address: Robot IP address
        port: Communication port
        control_freq: Control frequency in Hz
    """
    
    def __init__(
        self,
        ip_address: str = "192.168.1.100",
        port: int = 5000,
        control_freq: int = 20,
    ):
        super().__init__()
        
        self.ip_address = ip_address
        self.port = port
        self.control_freq = control_freq
        
        # Connect to robot
        self._connect()
    
    def _connect(self):
        """Establish connection to robot."""
        # TODO: Implement connection logic
        pass
    
    def get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get current robot observation.
        
        Returns:
            Dictionary containing:
                - images: Camera images [num_cameras, H, W, C]
                - proprioception: Joint positions/velocities [state_dim]
        """
        # TODO: Implement observation retrieval
        pass
    
    def execute_action(self, action: np.ndarray) -> bool:
        """
        Execute an action on the robot.
        
        Args:
            action: Action to execute [action_dim]
            
        Returns:
            Success status
        """
        # TODO: Implement action execution
        pass
    
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset robot to initial state."""
        # TODO: Implement reset logic
        pass
    
    def close(self):
        """Close connection to robot."""
        # TODO: Implement cleanup
        pass
```

## Testing Your Component

### Unit Tests

Create tests for your component:

```python
# tests/test_my_vlm.py
import pytest
import torch
from librobot.models.vlm import create_vlm


class TestMyVLM:
    """Tests for MyVLM."""
    
    def test_initialization(self):
        """Test model initialization."""
        vlm = create_vlm("my-vlm", hidden_dim=768)
        assert vlm.get_embedding_dim() == 768
    
    def test_encode_image(self):
        """Test image encoding."""
        vlm = create_vlm("my-vlm")
        images = torch.randn(2, 3, 224, 224)
        embeddings = vlm.encode_image(images)
        
        assert embeddings.shape[0] == 2
        assert embeddings.dim() == 3
    
    def test_forward_pass(self):
        """Test forward pass."""
        vlm = create_vlm("my-vlm")
        images = torch.randn(2, 3, 224, 224)
        
        outputs = vlm(images)
        assert "embeddings" in outputs
        assert outputs["embeddings"].requires_grad
    
    def test_registry(self):
        """Test registry integration."""
        from librobot.models.vlm import list_vlms
        assert "my-vlm-base" in list_vlms()
        assert "my-vlm" in list_vlms()  # Alias
```

### Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_my_vlm.py

# Run with coverage
pytest --cov=librobot tests/
```

## Documentation Requirements

### Docstring Format

Use Google-style docstrings:

```python
def my_function(arg1: int, arg2: str) -> bool:
    """
    Short description of the function.
    
    Longer description with more details about what the function
    does and how it works.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When arg1 is negative
    
    Example:
        >>> result = my_function(42, "hello")
        >>> print(result)
        True
    """
    pass
```

### README for Your Component

Create a README explaining your component:

```markdown
# My VLM

My custom Vision-Language Model for robot learning.

## Features

- Feature 1
- Feature 2
- Feature 3

## Usage

\`\`\`python
from librobot.models.vlm import create_vlm

vlm = create_vlm("my-vlm", hidden_dim=768)
images = torch.randn(2, 3, 224, 224)
outputs = vlm(images)
\`\`\`

## Architecture

Description of your architecture...

## Performance

Benchmarks and performance metrics...

## Citation

\`\`\`bibtex
@article{yourpaper2024,
  title={Your Paper Title},
  author={Your Name},
  year={2024}
}
\`\`\`
```

## Publishing Your Plugin

### Package Structure

```
my-librobot-plugin/
├── my_plugin/
│   ├── __init__.py
│   └── my_component.py
├── tests/
│   └── test_my_component.py
├── README.md
├── LICENSE
├── setup.py
└── requirements.txt
```

### setup.py

```python
from setuptools import setup, find_packages

setup(
    name="my-librobot-plugin",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "librobot>=0.1.0",
        "torch>=2.0.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="My custom plugin for LibroBot VLA",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/my-librobot-plugin",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
```

### Installation

Users can install your plugin:

```bash
pip install my-librobot-plugin
```

Then use it:

```python
import my_plugin  # Registers components automatically
from librobot.models.vlm import create_vlm

vlm = create_vlm("my-vlm")
```

## Conclusion

This guide provided comprehensive instructions for extending LibroBot VLA with new components. Follow these patterns for consistency and quality.

For more information:
- [Architecture](./ARCHITECTURE.md) - System architecture
- [Design Principles](./DESIGN_PRINCIPLES.md) - Design patterns
- [API Contracts](./API_CONTRACTS.md) - Interface definitions
