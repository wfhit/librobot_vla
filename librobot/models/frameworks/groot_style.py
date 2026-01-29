"""GR00T-style VLA framework implementation."""

from typing import Dict, List, Optional

import torch

from librobot.models.frameworks.base import BaseVLAFramework
from librobot.utils.registry import register_framework


@register_framework("groot_style", aliases=["groot"])
class GR00TStyleFramework(BaseVLAFramework):
    """GR00T-style VLA framework.

    Architecture:
    - VLM processes images + instructions (frozen)
    - State encoder processes robot state separately (bypasses VLM)
    - Features concatenated and fed to action head
    - Optional history encoder for temporal context
    """

    def __init__(
        self,
        vlm,
        action_head,
        state_encoder=None,
        history_encoder=None,
        fusion_method: str = "concat",
        **kwargs,
    ):
        """Initialize GR00T-style framework.

        Args:
            vlm: Vision-Language Model
            action_head: Action prediction head
            state_encoder: State encoder
            history_encoder: Optional history encoder
            fusion_method: How to fuse features ('concat', 'add', 'cross_attention')
        """
        super().__init__(vlm, action_head, state_encoder, history_encoder)
        self.fusion_method = fusion_method

        # Determine fusion dimension
        fusion_dim = vlm.hidden_dim
        if state_encoder is not None:
            if fusion_method == "concat":
                fusion_dim += state_encoder.output_dim
            elif fusion_method == "add":
                assert state_encoder.output_dim == vlm.hidden_dim
        if history_encoder is not None:
            if fusion_method == "concat":
                fusion_dim += history_encoder.output_dim

        # Project to action head input dim if needed
        if fusion_dim != action_head.input_dim:
            self.fusion_proj = torch.nn.Linear(fusion_dim, action_head.input_dim)
        else:
            self.fusion_proj = torch.nn.Identity()

    def forward(
        self,
        images: torch.Tensor,
        instruction: List[str],
        state: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            images: Input images [batch_size, num_images, C, H, W]
            instruction: Text instructions
            state: Robot state [batch_size, state_dim]
            history: Action/observation history [batch_size, history_len, history_dim]
            actions: Ground truth actions [batch_size, action_horizon, action_dim]

        Returns:
            Dictionary with predictions and loss
        """
        # Get VLM features
        vlm_output = self.vlm(images, instruction)
        vlm_features = vlm_output["hidden_states"]

        # Pool VLM features (use mean pooling)
        if vlm_features.ndim == 3:
            vlm_features = vlm_features.mean(dim=1)  # [batch_size, hidden_dim]

        # Encode state
        features = [vlm_features]
        if state is not None and self.state_encoder is not None:
            state_features = self.state_encoder(state)
            features.append(state_features)

        # Encode history
        if history is not None and self.history_encoder is not None:
            history_features = self.history_encoder(history)
            if history_features.ndim == 3:
                history_features = history_features.mean(dim=1)
            features.append(history_features)

        # Fuse features
        if self.fusion_method == "concat":
            fused_features = torch.cat(features, dim=-1)
        elif self.fusion_method == "add":
            fused_features = sum(features)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        # Project to action head input dim
        fused_features = self.fusion_proj(fused_features)

        # Predict actions
        output = self.action_head(fused_features, actions=actions, **kwargs)

        return output

    def predict(
        self,
        images: torch.Tensor,
        instruction: List[str],
        state: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Predict actions at inference time.

        Args:
            images: Input images [batch_size, num_images, C, H, W]
            instruction: Text instructions
            state: Robot state [batch_size, state_dim]
            history: Action/observation history
            num_samples: Number of samples to generate

        Returns:
            Predicted actions [batch_size, num_samples, action_horizon, action_dim]
        """
        # Get VLM features
        vlm_output = self.vlm(images, instruction)
        vlm_features = vlm_output["hidden_states"]

        # Pool VLM features
        if vlm_features.ndim == 3:
            vlm_features = vlm_features.mean(dim=1)

        # Encode state
        features = [vlm_features]
        if state is not None and self.state_encoder is not None:
            state_features = self.state_encoder(state)
            features.append(state_features)

        # Encode history
        if history is not None and self.history_encoder is not None:
            history_features = self.history_encoder(history)
            if history_features.ndim == 3:
                history_features = history_features.mean(dim=1)
            features.append(history_features)

        # Fuse features
        if self.fusion_method == "concat":
            fused_features = torch.cat(features, dim=-1)
        elif self.fusion_method == "add":
            fused_features = sum(features)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        # Project to action head input dim
        fused_features = self.fusion_proj(fused_features)

        # Predict actions
        actions = self.action_head.predict(fused_features, num_samples=num_samples, **kwargs)

        return actions
