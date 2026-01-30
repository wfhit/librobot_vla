"""Berkeley Octo-style VLA framework implementation."""

from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoders.history.transformer_encoder import TransformerHistoryEncoder
from ..encoders.state.transformer_encoder import TransformerStateEncoder
from .base import AbstractVLA


class OctoVLA(AbstractVLA):
    """
    Berkeley Octo-style Vision-Language-Action framework.

    Architecture:
        - Unified transformer architecture for all modalities
        - Task conditioning via learned task embeddings
        - Multi-task learning across diverse datasets
        - Flexible observation and action spaces
        - History-aware action prediction

    Key Features:
        - Unified Architecture: Single transformer processes all inputs
        - Task Conditioning: Explicit task embeddings for multi-task learning
        - Flexible Spaces: Handles variable observation/action dimensions
        - History Integration: Temporal context for better predictions
        - Multi-dataset Training: Designed for diverse robot data

    Args:
        vision_encoder: Visual encoder (e.g., ResNet, ViT)
        action_dim: Dimension of action space
        state_dim: Dimension of proprioceptive state
        hidden_dim: Hidden dimension for transformer
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        history_length: Length of history to condition on
        num_tasks: Number of tasks (for task embedding)
        dropout: Dropout rate
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        action_dim: int,
        state_dim: Optional[int] = None,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        history_length: int = 1,
        num_tasks: int = 100,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.history_length = history_length
        self.num_tasks = num_tasks

        # Task embeddings for multi-task learning
        self.task_embedding = nn.Embedding(num_tasks, hidden_dim)

        # Modality-specific projections to unified embedding space
        # Vision projection
        self.vision_proj = nn.Sequential(
            nn.Linear(self._get_vision_dim(), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # State encoder (if proprioception is used)
        if state_dim is not None:
            self.state_encoder = TransformerStateEncoder(
                input_dim=state_dim,
                output_dim=hidden_dim,
                num_layers=2,
                num_heads=4,
                ffn_dim=hidden_dim * 4,
            )
        else:
            self.state_encoder = None

        # History encoder
        if history_length > 1:
            self.history_encoder = TransformerHistoryEncoder(
                input_dim=hidden_dim,
                output_dim=hidden_dim,
                num_layers=2,
                num_heads=4,
                max_length=history_length,
            )
        else:
            self.history_encoder = None

        # Positional embeddings for sequence positions
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, hidden_dim))  # Max sequence length

        # Unified transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Action head - simple MLP for continuous actions
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        # Action query token
        self.action_query = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def _get_vision_dim(self) -> int:
        """Get output dimension of vision encoder."""
        # Try to infer from encoder
        if hasattr(self.vision_encoder, "output_dim"):
            return self.vision_encoder.output_dim
        elif hasattr(self.vision_encoder, "embed_dim"):
            return self.vision_encoder.embed_dim
        else:
            # Default assumption
            return 512

    def forward(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, list[str]]] = None,
        proprioception: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
        history_images: Optional[torch.Tensor] = None,
        history_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of Octo VLA.

        Args:
            images: Input images [batch_size, C, H, W]
            text: Optional text instructions (task description)
            proprioception: Optional proprioceptive state [batch_size, state_dim]
            actions: Optional action targets for training [batch_size, action_dim]
            task_id: Optional task ID for multi-task learning [batch_size]
            history_images: Optional history images [batch_size, history_len, C, H, W]
            history_states: Optional history states [batch_size, history_len, state_dim]
            **kwargs: Additional arguments

        Returns:
            Dictionary containing:
                - 'actions': Predicted actions [batch_size, action_dim]
                - 'loss': Training loss (if actions provided)
        """
        batch_size = images.size(0)

        # Default task ID if not provided
        if task_id is None:
            task_id = torch.zeros(batch_size, dtype=torch.long, device=images.device)

        # Get task embeddings
        task_emb = self.task_embedding(task_id).unsqueeze(1)  # [batch, 1, hidden]

        # Encode visual observations
        vision_features = self.vision_encoder(images)
        if vision_features.dim() == 4:  # [batch, C, H, W]
            vision_features = F.adaptive_avg_pool2d(vision_features, (1, 1)).squeeze(-1).squeeze(-1)
        elif vision_features.dim() == 3:  # [batch, seq, dim]
            vision_features = vision_features.mean(dim=1)

        vision_emb = self.vision_proj(vision_features).unsqueeze(1)  # [batch, 1, hidden]

        # Encode proprioceptive state
        if proprioception is not None and self.state_encoder is not None:
            # TransformerStateEncoder expects 3D input [batch, seq, dim]
            state_emb = self.state_encoder(proprioception.unsqueeze(1))  # [batch, 1, hidden]
        else:
            state_emb = None

        # Build token sequence: [task, vision, state?, history?, action_query]
        token_list = [task_emb, vision_emb]

        if state_emb is not None:
            token_list.append(state_emb)

        # Add history if provided
        if history_images is not None and self.history_encoder is not None:
            # Encode history
            hist_len = history_images.size(1)
            hist_imgs = history_images.view(batch_size * hist_len, *history_images.shape[2:])
            hist_vision = self.vision_encoder(hist_imgs)
            if hist_vision.dim() == 4:
                hist_vision = F.adaptive_avg_pool2d(hist_vision, (1, 1)).squeeze(-1).squeeze(-1)
            elif hist_vision.dim() == 3:
                hist_vision = hist_vision.mean(dim=1)
            hist_vision = hist_vision.view(batch_size, hist_len, -1)
            hist_vision = self.vision_proj(hist_vision)

            # Encode with history encoder
            hist_emb = self.history_encoder(hist_vision)  # [batch, hidden]
            token_list.append(hist_emb.unsqueeze(1))

        # Add action query
        action_queries = self.action_query.expand(batch_size, -1, -1)
        token_list.append(action_queries)

        # Concatenate all tokens
        tokens = torch.cat(token_list, dim=1)  # [batch, seq_len, hidden]
        seq_len = tokens.size(1)

        # Add positional embeddings
        tokens = tokens + self.pos_embedding[:, :seq_len, :]

        # Process through transformer
        transformer_out = self.transformer(tokens)

        # Extract action token (last token)
        action_token = transformer_out[:, -1, :]  # [batch, hidden]

        # Predict actions
        predicted_actions = self.action_head(action_token)

        if actions is not None:
            # Training mode: compute loss
            loss = F.mse_loss(predicted_actions, actions)

            return {
                "actions": predicted_actions,
                "loss": loss,
                "action_token": action_token,
            }
        else:
            # Inference mode
            return {
                "actions": predicted_actions,
                "action_token": action_token,
            }

    def predict_action(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, list[str]]] = None,
        proprioception: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
        history_images: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predict actions for inference.

        Args:
            images: Input images
            text: Optional text instructions
            proprioception: Optional proprioceptive state
            task_id: Optional task ID
            history_images: Optional history images
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
                task_id=task_id,
                history_images=history_images,
                actions=None,
                **kwargs,
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
        return {
            "type": "OctoVLA",
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "history_length": self.history_length,
            "num_tasks": self.num_tasks,
        }
