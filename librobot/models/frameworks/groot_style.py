"""NVIDIA GR00T-style VLA framework implementation."""

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AbstractVLA
from ..vlm.base import AbstractVLM
from ..encoders.state.mlp_encoder import MLPStateEncoder
from ..encoders.fusion.film import FiLMFusion
from ..action_heads.diffusion.ddpm import DDPMActionHead


class GR00TVLA(AbstractVLA):
    """
    NVIDIA GR00T-style Vision-Language-Action framework.

    Architecture:
        - Frozen VLM backbone for visual-language understanding
        - State encoder (MLP/Transformer) for proprioception
        - FiLM conditioning to modulate visual features with VLM embeddings
        - Diffusion action head for continuous action prediction
        - Multi-camera support via feature fusion

    Key Features:
        - Frozen VLM: Pre-trained VLM is frozen for stability
        - FiLM Modulation: VLM features condition the action prediction
        - Diffusion Policy: DDPM-based action generation
        - Multi-camera: Supports multiple camera inputs

    Args:
        vlm: Pre-trained VLM backbone
        action_dim: Dimension of action space
        state_dim: Dimension of proprioceptive state
        hidden_dim: Hidden dimension for encoders
        num_cameras: Number of camera views
        diffusion_steps: Number of diffusion timesteps
        freeze_vlm: Whether to freeze VLM backbone
        state_encoder_layers: Number of layers in state encoder
    """

    def __init__(
        self,
        vlm: AbstractVLM,
        action_dim: int,
        state_dim: Optional[int] = None,
        hidden_dim: int = 512,
        num_cameras: int = 1,
        diffusion_steps: int = 100,
        freeze_vlm: bool = True,
        state_encoder_layers: int = 2,
    ):
        super().__init__()

        self.vlm = vlm
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_cameras = num_cameras
        self.diffusion_steps = diffusion_steps

        # Get VLM embedding dimension
        self.vlm_dim = vlm.get_embedding_dim()

        # Freeze VLM if requested
        if freeze_vlm:
            self.freeze_backbone()

        # State encoder (if proprioception is used)
        if state_dim is not None:
            self.state_encoder = MLPStateEncoder(
                input_dim=state_dim,
                output_dim=hidden_dim,
                hidden_dims=[hidden_dim] * state_encoder_layers,
            )
        else:
            self.state_encoder = None

        # Multi-camera fusion (average pooling for simplicity)
        if num_cameras > 1:
            self.camera_fusion = nn.Sequential(
                nn.Linear(self.vlm_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.camera_fusion = None

        # FiLM conditioning layer
        self.film_fusion = FiLMFusion(
            feature_dim=hidden_dim,
            context_dim=self.vlm_dim,
            use_residual=True,
        )

        # Projection to action head input
        input_dim = hidden_dim
        if state_dim is not None:
            input_dim += hidden_dim  # Concatenate state embeddings

        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Diffusion action head
        self.action_head = DDPMActionHead(
            input_dim=hidden_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_timesteps=diffusion_steps,
            beta_schedule='cosine',
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
        Forward pass of GR00T VLA.

        Args:
            images: Input images [batch_size, num_cameras, C, H, W] or [batch_size, C, H, W]
            text: Optional text instructions
            proprioception: Optional proprioceptive state [batch_size, state_dim]
            actions: Optional action targets for training
            **kwargs: Additional arguments

        Returns:
            Dictionary containing:
                - 'actions': Predicted actions [batch_size, action_dim]
                - 'loss': Training loss (if actions provided)
                - 'vlm_embeddings': VLM embeddings
        """
        batch_size = images.size(0)

        # Handle multi-camera inputs
        if images.dim() == 5:  # [batch, num_cameras, C, H, W]
            num_cameras = images.size(1)
            images = images.view(batch_size * num_cameras, *images.shape[2:])
            multi_camera = True
        else:
            multi_camera = False

        # Extract VLM features
        with torch.set_grad_enabled(not self.training or self.vlm.training):
            vlm_outputs = self.vlm(images, text=text, **kwargs)
            vlm_embeddings = vlm_outputs['embeddings']

        # Pool VLM embeddings if sequence
        if vlm_embeddings.dim() == 3:
            vlm_embeddings = vlm_embeddings.mean(dim=1)  # [batch * num_cameras, vlm_dim]

        # Fuse multi-camera features
        if multi_camera and self.camera_fusion is not None:
            vlm_embeddings = vlm_embeddings.view(batch_size, num_cameras, -1)
            vlm_embeddings = vlm_embeddings.mean(dim=1)  # Average pooling
            visual_features = self.camera_fusion(vlm_embeddings)
        else:
            visual_features = vlm_embeddings
            if visual_features.size(-1) != self.hidden_dim:
                visual_features = F.adaptive_avg_pool1d(
                    visual_features.unsqueeze(1), self.hidden_dim
                ).squeeze(1)

        # Apply FiLM conditioning
        if multi_camera:
            context_for_film = vlm_embeddings
        else:
            context_for_film = vlm_embeddings.view(batch_size, -1, self.vlm_dim)[:, 0, :]

        visual_features_modulated = self.film_fusion(visual_features, context_for_film)

        # Encode proprioception
        if proprioception is not None and self.state_encoder is not None:
            state_features = self.state_encoder(proprioception)
            # Concatenate state features
            combined_features = torch.cat([visual_features_modulated, state_features], dim=-1)
        else:
            combined_features = visual_features_modulated

        # Project features
        action_input = self.feature_proj(combined_features)

        # Predict actions
        if actions is not None:
            # Training mode: compute loss
            action_pred = self.action_head(action_input)
            loss = self.action_head.compute_loss(action_pred, actions)

            return {
                'actions': self.action_head.sample(action_input),
                'loss': loss,
                'vlm_embeddings': vlm_embeddings,
                'action_input': action_input,
            }
        else:
            # Inference mode: sample actions
            predicted_actions = self.action_head.sample(action_input)

            return {
                'actions': predicted_actions,
                'vlm_embeddings': vlm_embeddings,
                'action_input': action_input,
            }

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
                images=images,
                text=text,
                proprioception=proprioception,
                actions=None,
                **kwargs
            )
            return outputs['actions']

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
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

        # Main action loss (already computed in forward)
        if 'loss' in predictions:
            losses['action_loss'] = predictions['loss']
            losses['total_loss'] = predictions['loss']

        return losses

    def get_config(self) -> Dict[str, Any]:
        """
        Get framework configuration.

        Returns:
            Dictionary containing configuration
        """
        return {
            'type': 'GR00TVLA',
            'vlm_config': self.vlm.config,
            'action_dim': self.action_dim,
            'state_dim': self.state_dim,
            'hidden_dim': self.hidden_dim,
            'num_cameras': self.num_cameras,
            'diffusion_steps': self.diffusion_steps,
        }
