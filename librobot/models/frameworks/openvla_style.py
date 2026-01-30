"""Berkeley OpenVLA-style VLA framework implementation."""

from typing import Any, Optional, Union

import torch
import torch.nn as nn

from ..action_heads.mlp_oft import MLPActionHead
from ..vlm.base import AbstractVLM
from .base import AbstractVLA


class OpenVLA(AbstractVLA):
    """
    Berkeley OpenVLA-style Vision-Language-Action framework.

    Architecture:
        - Fine-tune open-source VLM (e.g., LLaVA, Prismatic) end-to-end
        - Action tokens: Add special tokens to VLM vocabulary for actions
        - Output-from-Tokens (OFT): MLP head extracts actions from token embeddings
        - Instruction following: Natural language task specification
        - Co-training: Train on vision-language AND robotics data

    Key Features:
        - VLM Fine-tuning: End-to-end training of VLM backbone
        - Action Tokens: Actions represented as special tokens in sequence
        - MLP OFT Head: Extract continuous actions from discrete token space
        - Instruction Following: Natural language interface
        - Open-source: Built on accessible VLMs (LLaVA, Prismatic, etc.)

    Args:
        vlm: Pre-trained open-source VLM backbone
        action_dim: Dimension of action space
        hidden_dim: Hidden dimension for action head
        num_action_tokens: Number of special action tokens
        fine_tune_vlm: Whether to fine-tune VLM (recommended)
        freeze_vision_encoder: Whether to freeze vision encoder only
        action_token_pattern: Pattern for action token output
    """

    def __init__(
        self,
        vlm: AbstractVLM,
        action_dim: int,
        hidden_dim: int = 512,
        num_action_tokens: int = 1,
        fine_tune_vlm: bool = True,
        freeze_vision_encoder: bool = False,
        action_token_pattern: str = 'last',  # 'last', 'first', 'mean', 'special'
    ):
        super().__init__()

        self.vlm = vlm
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_action_tokens = num_action_tokens
        self.action_token_pattern = action_token_pattern

        # Get VLM embedding dimension
        self.vlm_dim = vlm.get_embedding_dim()

        # Configure VLM freezing
        if not fine_tune_vlm:
            self.freeze_backbone()
        elif freeze_vision_encoder:
            # Freeze only vision encoder, fine-tune language model
            if hasattr(self.vlm, 'vision_encoder'):
                for param in self.vlm.vision_encoder.parameters():
                    param.requires_grad = False

        # Special action tokens (if using 'special' pattern)
        if action_token_pattern == 'special':
            self.action_tokens = nn.Parameter(
                torch.randn(num_action_tokens, self.vlm_dim)
            )
        else:
            self.action_tokens = None

        # Output-from-Tokens (OFT) MLP head
        # Takes token embeddings and produces continuous actions
        self.action_head = MLPActionHead(
            input_dim=self.vlm_dim * num_action_tokens,
            action_dim=action_dim,
            hidden_dims=[hidden_dim, hidden_dim],
        )

        # Optional: Action token position projection
        if action_token_pattern in ['special', 'last', 'first']:
            self.token_proj = nn.Identity()
        else:
            self.token_proj = nn.Sequential(
                nn.Linear(self.vlm_dim, self.vlm_dim),
                nn.LayerNorm(self.vlm_dim),
                nn.GELU(),
            )

    def _extract_action_tokens(
        self,
        embeddings: torch.Tensor,
        pattern: str = None,
    ) -> torch.Tensor:
        """
        Extract action tokens from VLM output embeddings.

        Args:
            embeddings: VLM output embeddings [batch, seq_len, dim]
            pattern: Token extraction pattern

        Returns:
            Action token embeddings [batch, num_action_tokens * dim]
        """
        if pattern is None:
            pattern = self.action_token_pattern

        batch_size = embeddings.size(0)

        if pattern == 'last':
            # Use last N tokens
            action_tokens = embeddings[:, -self.num_action_tokens:, :]
        elif pattern == 'first':
            # Use first N tokens
            action_tokens = embeddings[:, :self.num_action_tokens, :]
        elif pattern == 'mean':
            # Mean pool all tokens, then expand
            pooled = embeddings.mean(dim=1, keepdim=True)
            action_tokens = pooled.expand(-1, self.num_action_tokens, -1)
        elif pattern == 'special':
            # Use learned special tokens (append and process)
            # This would be done during forward pass with VLM
            action_tokens = embeddings[:, -self.num_action_tokens:, :]
        else:
            raise ValueError(f"Unknown action token pattern: {pattern}")

        # Flatten action tokens
        action_tokens_flat = action_tokens.reshape(batch_size, -1)

        return action_tokens_flat

    def forward(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, list[str]]] = None,
        proprioception: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of OpenVLA.

        Args:
            images: Input images [batch_size, C, H, W]
            text: Text instructions (natural language task specification)
            proprioception: Optional proprioceptive state (can be encoded in text)
            actions: Optional action targets for training [batch_size, action_dim]
            **kwargs: Additional arguments

        Returns:
            Dictionary containing:
                - 'actions': Predicted actions [batch_size, action_dim]
                - 'loss': Training loss (if actions provided)
                - 'vlm_outputs': Raw VLM outputs
        """
        batch_size = images.size(0)

        # Optionally encode proprioception into text prompt
        if proprioception is not None and text is not None:
            # Convert state to string and append to instruction
            if isinstance(text, str):
                text = [text] * batch_size

            # Add state information to prompt
            enhanced_text = []
            for i, txt in enumerate(text):
                state_str = ', '.join([f'{x:.3f}' for x in proprioception[i].tolist()])
                enhanced_text.append(f"{txt} [Current state: {state_str}]")
            text = enhanced_text

        # Process through VLM
        vlm_outputs = self.vlm(images, text=text, **kwargs)
        vlm_embeddings = vlm_outputs['embeddings']  # [batch, seq_len, vlm_dim]

        # Optionally add special action tokens to sequence
        if self.action_tokens is not None:
            # Append special action tokens
            action_token_batch = self.action_tokens.unsqueeze(0).expand(
                batch_size, -1, -1
            )
            vlm_embeddings = torch.cat([vlm_embeddings, action_token_batch], dim=1)

        # Extract action tokens from embeddings
        action_token_features = self._extract_action_tokens(vlm_embeddings)

        # Optional projection
        if self.token_proj is not None and not isinstance(self.token_proj, nn.Identity):
            action_token_features = self.token_proj(
                action_token_features.view(batch_size, self.num_action_tokens, -1)
            ).view(batch_size, -1)

        # Predict actions through MLP OFT head
        action_outputs = self.action_head(action_token_features)
        predicted_actions = action_outputs['actions']

        if actions is not None:
            # Training mode: compute loss
            loss = self.action_head.compute_loss(action_outputs, actions)

            return {
                'actions': predicted_actions,
                'loss': loss,
                'vlm_outputs': vlm_outputs,
                'action_token_features': action_token_features,
            }
        else:
            # Inference mode
            return {
                'actions': predicted_actions,
                'vlm_outputs': vlm_outputs,
                'action_token_features': action_token_features,
            }

    def predict_action(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, list[str]]] = None,
        proprioception: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Predict actions for inference.

        Args:
            images: Input images
            text: Text instructions
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

        # Optional: Add language modeling loss if co-training
        if 'lm_loss' in predictions:
            losses['lm_loss'] = predictions['lm_loss']
            losses['total_loss'] = losses['total_loss'] + 0.1 * predictions['lm_loss']

        return losses

    def get_config(self) -> dict[str, Any]:
        """
        Get framework configuration.

        Returns:
            Dictionary containing configuration
        """
        return {
            'type': 'OpenVLA',
            'vlm_config': self.vlm.config,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim,
            'num_action_tokens': self.num_action_tokens,
            'action_token_pattern': self.action_token_pattern,
        }

    def get_trainable_params(self) -> list[str]:
        """
        Get names of trainable parameters.

        Returns:
            List of trainable parameter names
        """
        return [name for name, param in self.named_parameters() if param.requires_grad]
