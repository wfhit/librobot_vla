"""Base VLA framework interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class BaseVLAFramework(nn.Module, ABC):
    """Base class for VLA (Vision-Language-Action) frameworks.

    A VLA framework orchestrates the interaction between:
    - Vision-Language Models (VLM)
    - State/History Encoders
    - Action Heads
    """

    def __init__(
        self,
        vlm: nn.Module,
        action_head: nn.Module,
        state_encoder: Optional[nn.Module] = None,
        history_encoder: Optional[nn.Module] = None,
        **kwargs,
    ):
        """Initialize VLA framework.

        Args:
            vlm: Vision-Language Model
            action_head: Action prediction head
            state_encoder: Optional state encoder
            history_encoder: Optional history encoder
            **kwargs: Framework-specific arguments
        """
        super().__init__()
        self.vlm = vlm
        self.action_head = action_head
        self.state_encoder = state_encoder
        self.history_encoder = history_encoder

    @abstractmethod
    def forward(
        self,
        images: torch.Tensor,
        instruction: List[str],
        state: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through VLA framework.

        Args:
            images: Input images [batch_size, num_images, C, H, W]
            instruction: Text instructions
            state: Robot state [batch_size, state_dim]
            history: Action/observation history [batch_size, history_len, history_dim]
            actions: Ground truth actions for training [batch_size, action_horizon, action_dim]
            **kwargs: Framework-specific arguments

        Returns:
            Dictionary containing:
                - actions: Predicted actions
                - loss: Training loss (if actions provided)
                - (optional) other framework-specific outputs
        """
        pass

    @abstractmethod
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
            history: Action/observation history [batch_size, history_len, history_dim]
            num_samples: Number of action samples to generate
            **kwargs: Framework-specific arguments

        Returns:
            Predicted actions [batch_size, num_samples, action_horizon, action_dim]
        """
        pass

    def freeze_vlm(self) -> None:
        """Freeze VLM parameters."""
        if hasattr(self.vlm, "freeze_parameters"):
            self.vlm.freeze_parameters()
        else:
            for param in self.vlm.parameters():
                param.requires_grad = False

    def unfreeze_vlm(self) -> None:
        """Unfreeze VLM parameters."""
        if hasattr(self.vlm, "unfreeze_parameters"):
            self.vlm.unfreeze_parameters()
        else:
            for param in self.vlm.parameters():
                param.requires_grad = True

    def get_trainable_parameters(self) -> Dict[str, int]:
        """Get number of trainable parameters by component."""
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        params = {
            "vlm": count_params(self.vlm),
            "action_head": count_params(self.action_head),
            "total": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

        if self.state_encoder is not None:
            params["state_encoder"] = count_params(self.state_encoder)
        if self.history_encoder is not None:
            params["history_encoder"] = count_params(self.history_encoder)

        return params
