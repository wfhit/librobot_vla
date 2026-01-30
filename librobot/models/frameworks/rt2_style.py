"""Google RT-2 style VLA framework implementation."""

from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..vlm.base import AbstractVLM
from .base import AbstractVLA


class RT2VLA(AbstractVLA):
    """
    Google RT-2 (Robotics Transformer 2) style Vision-Language-Action framework.

    Architecture:
        - VLM backbone (e.g., PaLI, PaLM-E) for vision-language understanding
        - Discretized action space: Continuous actions → 256 bins per dimension
        - Token-based action prediction: Predict action tokens autoregressively
        - Language conditioning: Natural language instructions
        - Co-fine-tuning: Train on web data + robotics data

    Key Features:
        - Action Discretization: Bins continuous actions into 256 discrete values
        - Autoregressive Prediction: Predict action dimensions sequentially
        - Language Conditioning: Instructions guide action generation
        - Transfer Learning: Leverage pre-trained VLM knowledge
        - Token Vocabulary: Actions as special tokens in VLM vocabulary

    Args:
        vlm: Pre-trained VLM backbone
        action_dim: Dimension of action space
        num_bins: Number of discretization bins per action dimension
        action_bounds: Bounds for action space [(min, max), ...]
        fine_tune_vlm: Whether to fine-tune VLM
        temperature: Sampling temperature for action tokens
    """

    def __init__(
        self,
        vlm: AbstractVLM,
        action_dim: int,
        num_bins: int = 256,
        action_bounds: Optional[list[tuple]] = None,
        fine_tune_vlm: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.vlm = vlm
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.temperature = temperature

        # Get VLM embedding dimension
        self.vlm_dim = vlm.get_embedding_dim()

        # Configure VLM training
        if not fine_tune_vlm:
            self.freeze_backbone()

        # Action bounds for discretization
        if action_bounds is None:
            # Default: assume normalized actions [-1, 1]
            action_bounds = [(-1.0, 1.0)] * action_dim
        self.action_bounds = action_bounds

        # Create bin edges for discretization
        self._create_discretization_bins()

        # Action token embeddings (one head per action dimension)
        # Each dimension is predicted as a classification problem over bins
        self.action_token_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.vlm_dim, self.vlm_dim // 2),
                nn.LayerNorm(self.vlm_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.vlm_dim // 2, num_bins),
            )
            for _ in range(action_dim)
        ])

        # Action queries (learned tokens for each action dimension)
        self.action_queries = nn.Parameter(
            torch.randn(1, action_dim, self.vlm_dim)
        )

        # Transformer decoder for autoregressive action prediction
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.vlm_dim,
            nhead=8,
            dim_feedforward=self.vlm_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.action_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

    def _create_discretization_bins(self):
        """Create bin edges for action discretization."""
        bins_list = []
        for low, high in self.action_bounds:
            bins = torch.linspace(low, high, self.num_bins + 1)
            bins_list.append(bins)

        # Register as buffer (not trainable)
        self.register_buffer('bins', torch.stack(bins_list))  # [action_dim, num_bins + 1]

    def discretize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Discretize continuous actions to bin indices.

        Args:
            actions: Continuous actions [batch, action_dim]

        Returns:
            Discretized actions [batch, action_dim] (bin indices)
        """
        actions.size(0)
        discretized = torch.zeros_like(actions, dtype=torch.long)

        for i in range(self.action_dim):
            # Use torch.bucketize for efficient discretization
            discretized[:, i] = torch.bucketize(
                actions[:, i],
                self.bins[i],
                right=False
            ).clamp(0, self.num_bins - 1)

        return discretized

    def undiscretize_actions(self, action_indices: torch.Tensor) -> torch.Tensor:
        """
        Convert bin indices back to continuous actions.

        Args:
            action_indices: Discretized actions [batch, action_dim] (bin indices)

        Returns:
            Continuous actions [batch, action_dim]
        """
        batch_size = action_indices.size(0)
        continuous = torch.zeros(
            batch_size, self.action_dim,
            device=action_indices.device,
            dtype=torch.float32
        )

        for i in range(self.action_dim):
            # Get bin centers
            indices = action_indices[:, i].clamp(0, self.num_bins - 1)
            bin_centers = (self.bins[i, :-1] + self.bins[i, 1:]) / 2
            continuous[:, i] = bin_centers[indices]

        return continuous

    def forward(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, list[str]]] = None,
        proprioception: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of RT-2 VLA.

        Args:
            images: Input images [batch_size, C, H, W]
            text: Text instructions
            proprioception: Optional proprioceptive state (encoded in text)
            actions: Optional action targets for training [batch_size, action_dim]
            **kwargs: Additional arguments

        Returns:
            Dictionary containing:
                - 'actions': Predicted actions [batch_size, action_dim]
                - 'loss': Training loss (if actions provided)
                - 'logits': Action token logits
        """
        batch_size = images.size(0)

        # Process through VLM
        vlm_outputs = self.vlm(images, text=text, **kwargs)
        vlm_embeddings = vlm_outputs['embeddings']  # [batch, seq_len, vlm_dim]

        # Pool VLM embeddings to use as memory for decoder
        memory = vlm_embeddings  # [batch, seq_len, vlm_dim]

        # Action queries
        action_queries = self.action_queries.expand(batch_size, -1, -1)

        # Decode action tokens autoregressively
        decoded_actions = self.action_decoder(action_queries, memory)
        # [batch, action_dim, vlm_dim]

        # Predict bin for each action dimension
        action_logits = []
        for i in range(self.action_dim):
            logits = self.action_token_heads[i](decoded_actions[:, i, :])
            action_logits.append(logits)

        action_logits = torch.stack(action_logits, dim=1)  # [batch, action_dim, num_bins]

        if actions is not None:
            # Training mode: compute classification loss
            discretized_actions = self.discretize_actions(actions)

            # Cross-entropy loss for each action dimension
            loss = 0.0
            for i in range(self.action_dim):
                loss += F.cross_entropy(
                    action_logits[:, i, :],
                    discretized_actions[:, i],
                    reduction='mean'
                )
            loss = loss / self.action_dim

            # Also return predicted actions
            predicted_bins = action_logits.argmax(dim=-1)
            predicted_actions = self.undiscretize_actions(predicted_bins)

            return {
                'actions': predicted_actions,
                'loss': loss,
                'logits': action_logits,
                'discretized_actions': discretized_actions,
            }
        else:
            # Inference mode: sample from logits
            predicted_bins = self._sample_actions(action_logits)
            predicted_actions = self.undiscretize_actions(predicted_bins)

            return {
                'actions': predicted_actions,
                'logits': action_logits,
                'discretized_actions': predicted_bins,
            }

    def _sample_actions(
        self,
        logits: torch.Tensor,
        temperature: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Sample action bins from logits.

        Args:
            logits: Action logits [batch, action_dim, num_bins]
            temperature: Sampling temperature (None uses self.temperature)

        Returns:
            Sampled bin indices [batch, action_dim]
        """
        if temperature is None:
            temperature = self.temperature

        batch_size = logits.size(0)
        sampled_bins = torch.zeros(
            batch_size, self.action_dim,
            device=logits.device,
            dtype=torch.long
        )

        for i in range(self.action_dim):
            # Apply temperature
            scaled_logits = logits[:, i, :] / temperature
            probs = F.softmax(scaled_logits, dim=-1)

            # Sample from categorical distribution
            sampled_bins[:, i] = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return sampled_bins

    def predict_action(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, list[str]]] = None,
        proprioception: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Predict actions for inference.

        Args:
            images: Input images
            text: Text instructions
            proprioception: Optional proprioceptive state
            temperature: Sampling temperature
            **kwargs: Additional arguments

        Returns:
            Predicted actions [batch_size, action_dim]
        """
        self.eval()

        if temperature is not None:
            original_temp = self.temperature
            self.temperature = temperature

        with torch.no_grad():
            outputs = self.forward(
                images=images,
                text=text,
                proprioception=proprioception,
                actions=None,
                **kwargs
            )
            result = outputs['actions']

        if temperature is not None:
            self.temperature = original_temp

        return result

    def compute_loss(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        **kwargs
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

        if 'loss' in predictions:
            losses['action_loss'] = predictions['loss']
            losses['total_loss'] = predictions['loss']

        return losses

    def get_config(self) -> dict[str, Any]:
        """
        Get framework configuration.

        Returns:
            Dictionary containing configuration
        """
        return {
            'type': 'RT2VLA',
            'vlm_config': self.vlm.config,
            'action_dim': self.action_dim,
            'num_bins': self.num_bins,
            'action_bounds': self.action_bounds,
            'temperature': self.temperature,
        }
