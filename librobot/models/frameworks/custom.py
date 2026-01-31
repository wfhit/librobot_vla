"""Custom VLA framework template for user-defined architectures."""

from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..action_heads.base import AbstractActionHead
from ..encoders.base import AbstractEncoder
from ..vlm.base import AbstractVLM
from .base import AbstractVLA


class CustomVLA(AbstractVLA):
    """
    Custom VLA framework template for user-defined architectures.

    This class provides a flexible template for building custom VLA frameworks
    by composing different components:
        - Vision-Language Models (VLMs)
        - State encoders
        - Fusion modules
        - Action heads
        - Custom processing layers

    Features:
        - Flexible component composition
        - Mix-and-match support for different modules
        - Easy customization and experimentation
        - Inheritance-based extension

    Usage:
        1. Direct instantiation with components:
           ```python
           custom_vla = CustomVLA(
               vlm=my_vlm,
               action_head=my_action_head,
               state_encoder=my_state_encoder,
               ...
           )
           ```

        2. Subclass for more complex architectures:
           ```python
           class MyVLA(CustomVLA):
               def __init__(self, ...):
                   super().__init__(...)
                   # Add custom components
                   self.my_custom_layer = ...

               def _process_features(self, features):
                   # Override processing
                   return self.my_custom_layer(features)
           ```

    Args:
        vlm: Vision-Language Model (optional if using vision_encoder)
        vision_encoder: Vision encoder (alternative to VLM)
        text_encoder: Text encoder (alternative to VLM)
        action_head: Action prediction head
        state_encoder: State/proprioception encoder
        fusion_module: Module for fusing multiple modalities
        action_dim: Dimension of action space
        state_dim: Dimension of proprioceptive state
        hidden_dim: Hidden dimension for processing
        use_vlm: Whether to use VLM (vs separate encoders)
        freeze_backbone: Whether to freeze vision/language backbones
    """

    def __init__(
        self,
        action_dim: int,
        vlm: Optional[AbstractVLM] = None,
        vision_encoder: Optional[nn.Module] = None,
        text_encoder: Optional[nn.Module] = None,
        action_head: Optional[AbstractActionHead] = None,
        state_encoder: Optional[AbstractEncoder] = None,
        fusion_module: Optional[nn.Module] = None,
        state_dim: Optional[int] = None,
        hidden_dim: int = 512,
        use_vlm: bool = True,
        freeze_backbone: bool = False,
        custom_config: Optional[dict[str, Any]] = None,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.use_vlm = use_vlm
        self.custom_config = custom_config or {}

        # ============================================
        # Vision-Language Processing
        # ============================================

        if use_vlm:
            # Use unified VLM
            assert vlm is not None, "VLM must be provided if use_vlm=True"
            self.vlm = vlm
            self.vision_encoder = None
            self.text_encoder = None
            self.vlm_dim = vlm.get_embedding_dim()

            if freeze_backbone:
                self.freeze_backbone()
        else:
            # Use separate encoders
            self.vlm = None
            self.vision_encoder = vision_encoder
            self.text_encoder = text_encoder

            if freeze_backbone:
                if self.vision_encoder is not None:
                    for param in self.vision_encoder.parameters():
                        param.requires_grad = False
                if self.text_encoder is not None:
                    for param in self.text_encoder.parameters():
                        param.requires_grad = False

        # ============================================
        # State Encoding
        # ============================================

        self.state_encoder = state_encoder

        # ============================================
        # Feature Fusion
        # ============================================

        self.fusion_module = fusion_module

        # If no fusion module provided, use simple concatenation
        if self.fusion_module is None and not use_vlm:
            # Build default fusion
            self.fusion_module = self._build_default_fusion()

        # ============================================
        # Action Prediction
        # ============================================

        if action_head is None:
            # Build default action head (simple MLP)
            from ..action_heads.mlp_oft import MLPActionHead

            action_head = MLPActionHead(
                input_dim=hidden_dim,
                action_dim=action_dim,
                hidden_dims=[hidden_dim, hidden_dim // 2],
            )

        self.action_head = action_head

        # ============================================
        # Feature Processing
        # ============================================

        # Projection layer to unified feature space
        if use_vlm:
            input_dim = self.vlm_dim
        else:
            input_dim = hidden_dim

        if state_encoder is not None:
            input_dim += state_encoder.output_dim

        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def _build_default_fusion(self) -> nn.Module:
        """Build default fusion module for separate encoders."""
        return nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),  # Assuming vision + text
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )

    def _get_vision_dim(self) -> int:
        """Get vision encoder output dimension."""
        if self.vision_encoder is None:
            return 0
        if hasattr(self.vision_encoder, "output_dim"):
            return self.vision_encoder.output_dim
        elif hasattr(self.vision_encoder, "embed_dim"):
            return self.vision_encoder.embed_dim
        else:
            return self.hidden_dim

    def _encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """Encode visual inputs."""
        if self.use_vlm:
            # VLM handles vision
            return None

        if self.vision_encoder is None:
            raise ValueError("No vision encoder available")

        features = self.vision_encoder(images)

        # Handle different output formats
        if features.dim() == 4:  # [batch, C, H, W]
            features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
        elif features.dim() == 3:  # [batch, seq, dim]
            features = features.mean(dim=1)

        return features

    def _encode_text(self, text: Union[str, list[str]]) -> torch.Tensor:
        """Encode text inputs."""
        if self.use_vlm:
            # VLM handles text
            return None

        if self.text_encoder is None:
            # No text encoder, return dummy
            return None

        return self.text_encoder(text)

    def _process_features(
        self,
        vision_features: Optional[torch.Tensor],
        text_features: Optional[torch.Tensor],
        state_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Process and combine features from different modalities.

        Override this method in subclasses for custom feature processing.

        Args:
            vision_features: Vision features
            text_features: Text features
            state_features: State features

        Returns:
            Combined features
        """
        features_list = []

        if vision_features is not None:
            features_list.append(vision_features)

        if text_features is not None:
            features_list.append(text_features)

        # Fuse vision and text if both present
        if len(features_list) > 1 and self.fusion_module is not None:
            fused = self.fusion_module(torch.cat(features_list, dim=-1))
        elif len(features_list) == 1:
            fused = features_list[0]
        else:
            raise ValueError("No features available for processing")

        # Add state features
        if state_features is not None:
            combined = torch.cat([fused, state_features], dim=-1)
        else:
            combined = fused

        return combined

    def forward(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, list[str]]] = None,
        proprioception: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of custom VLA.

        Args:
            images: Input images [batch_size, C, H, W]
            text: Optional text instructions
            proprioception: Optional proprioceptive state
            actions: Optional action targets for training
            **kwargs: Additional arguments

        Returns:
            Dictionary containing:
                - 'actions': Predicted actions
                - 'loss': Training loss (if actions provided)
                - Other outputs
        """
        # Encode vision and language
        if self.use_vlm:
            vlm_outputs = self.vlm(images, text=text, **kwargs)
            vlm_embeddings = vlm_outputs["embeddings"]

            # Pool if needed
            if vlm_embeddings.dim() == 3:
                vlm_embeddings = vlm_embeddings.mean(dim=1)

            vision_features = vlm_embeddings
            text_features = None
        else:
            vision_features = self._encode_vision(images)
            text_features = self._encode_text(text) if text is not None else None

        # Encode state
        if proprioception is not None and self.state_encoder is not None:
            state_features = self.state_encoder(proprioception)
        else:
            state_features = None

        # Process and combine features
        combined_features = self._process_features(vision_features, text_features, state_features)

        # Project to action head input
        action_input = self.feature_proj(combined_features)

        # Predict actions
        action_outputs = self.action_head(action_input)

        if actions is not None:
            # Training mode: compute loss
            loss = self.action_head.compute_loss(action_outputs, actions)

            return {
                "actions": action_outputs.get("actions", action_outputs.get("logits")),
                "loss": loss,
                "action_input": action_input,
                **action_outputs,
            }
        else:
            # Inference mode
            # Try to sample if available, otherwise use direct output
            if hasattr(self.action_head, "sample"):
                predicted_actions = self.action_head.sample(action_input)
            else:
                predicted_actions = action_outputs.get("actions", action_outputs.get("logits"))

            return {
                "actions": predicted_actions,
                "action_input": action_input,
                **action_outputs,
            }

    def predict_action(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, list[str]]] = None,
        proprioception: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predict actions for inference.

        Args:
            images: Input images
            text: Optional text instructions
            proprioception: Optional proprioceptive state
            **kwargs: Additional arguments

        Returns:
            Predicted actions [batch_size, action_dim]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                images=images, text=text, proprioception=proprioception, actions=None, **kwargs
            )
            return outputs["actions"]

    def compute_loss(
        self, predictions: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Compute losses for training.

        Args:
            predictions: Model predictions from forward()
            targets: Ground truth targets
            **kwargs: Additional loss computation arguments

        Returns:
            Dictionary containing losses
        """
        losses = {}

        if "loss" in predictions:
            losses["action_loss"] = predictions["loss"]
            losses["total_loss"] = predictions["loss"]

        return losses

    def get_config(self) -> dict[str, Any]:
        """
        Get framework configuration.

        Returns:
            Dictionary containing configuration
        """
        config = {
            "type": "CustomVLA",
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "hidden_dim": self.hidden_dim,
            "use_vlm": self.use_vlm,
        }

        # Add VLM config if available
        if self.use_vlm and self.vlm is not None:
            config["vlm_config"] = self.vlm.config

        # Add custom config
        config.update(self.custom_config)

        return config

    def add_component(self, name: str, component: nn.Module) -> None:
        """
        Add a custom component to the framework.

        Args:
            name: Component name
            component: PyTorch module
        """
        setattr(self, name, component)

    def get_component(self, name: str) -> Optional[nn.Module]:
        """
        Get a component by name.

        Args:
            name: Component name

        Returns:
            Component module or None if not found
        """
        return getattr(self, name, None)
