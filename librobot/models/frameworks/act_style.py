"""ALOHA ACT (Action Chunking Transformer) style VLA framework implementation."""

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AbstractVLA


class ACTVLA(AbstractVLA):
    """
    ALOHA ACT (Action Chunking with Transformers) style VLA framework.

    Architecture:
        - Vision encoder: ResNet/ViT for image encoding
        - Transformer encoder: Encodes visual observations and state
        - CVAE latent variable model: Captures multi-modal action distributions
        - Transformer decoder: Decodes latent + observations to action chunks
        - Temporal consistency: Predicts sequences of actions (chunks)

    Key Features:
        - Action Chunking: Predict sequences of K future actions
        - CVAE: Conditional VAE for multi-modal action distributions
        - Transformer Architecture: Encoder-decoder with cross-attention
        - Bi-manual Support: Handles dual-arm robot control
        - Temporal Consistency: Smooth action sequences

    Args:
        vision_encoder: Vision encoder module
        action_dim: Dimension of action space
        state_dim: Dimension of proprioceptive state
        chunk_size: Number of actions to predict per forward pass
        hidden_dim: Hidden dimension for transformer
        latent_dim: Dimension of CVAE latent space
        num_encoder_layers: Number of transformer encoder layers
        num_decoder_layers: Number of transformer decoder layers
        num_heads: Number of attention heads
        kl_weight: Weight for KL divergence loss
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        action_dim: int,
        state_dim: Optional[int] = None,
        chunk_size: int = 10,
        hidden_dim: int = 512,
        latent_dim: int = 32,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        kl_weight: float = 0.0001,
    ):
        super().__init__()

        self.vision_encoder = vision_encoder
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        # Vision feature projection
        self.vision_proj = nn.Sequential(
            nn.Linear(self._get_vision_dim(), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # State encoder
        if state_dim is not None:
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.state_encoder = None

        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, hidden_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # CVAE components
        # Encoder: (obs, action_sequence) -> latent
        self.cvae_encoder = nn.Sequential(
            nn.Linear(hidden_dim + action_dim * chunk_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.cvae_mean = nn.Linear(hidden_dim, latent_dim)
        self.cvae_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder: (obs, latent) -> action_sequence
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)

        # Action queries (learned)
        self.action_queries = nn.Parameter(torch.randn(1, chunk_size, hidden_dim))

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

    def _get_vision_dim(self) -> int:
        """Get output dimension of vision encoder."""
        if hasattr(self.vision_encoder, 'output_dim'):
            return self.vision_encoder.output_dim
        elif hasattr(self.vision_encoder, 'embed_dim'):
            return self.vision_encoder.embed_dim
        else:
            return 512

    def encode_observations(
        self,
        images: torch.Tensor,
        proprioception: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode observations (images + state).

        Args:
            images: Input images [batch, C, H, W] or [batch, num_views, C, H, W]
            proprioception: Optional proprioceptive state [batch, state_dim]

        Returns:
            Encoded observation features [batch, seq_len, hidden_dim]
        """
        batch_size = images.size(0)

        # Handle multi-view images
        if images.dim() == 5:
            num_views = images.size(1)
            images = images.view(batch_size * num_views, *images.shape[2:])
            multi_view = True
        else:
            multi_view = False

        # Encode vision
        vision_features = self.vision_encoder(images)
        if vision_features.dim() == 4:  # [batch, C, H, W]
            vision_features = F.adaptive_avg_pool2d(vision_features, (1, 1))
            vision_features = vision_features.squeeze(-1).squeeze(-1)
        elif vision_features.dim() == 3:  # [batch, seq, dim]
            vision_features = vision_features.mean(dim=1)

        # Handle multi-view
        if multi_view:
            vision_features = vision_features.view(batch_size, num_views, -1)
        else:
            vision_features = vision_features.unsqueeze(1)

        vision_features = self.vision_proj(vision_features)

        # Encode state
        if proprioception is not None and self.state_encoder is not None:
            state_features = self.state_encoder(proprioception).unsqueeze(1)
            # Concatenate state with vision features
            obs_features = torch.cat([state_features, vision_features], dim=1)
        else:
            obs_features = vision_features

        # Add positional embeddings
        seq_len = obs_features.size(1)
        obs_features = obs_features + self.pos_embedding[:, :seq_len, :]

        # Encode through transformer
        encoded = self.encoder(obs_features)

        return encoded

    def cvae_encode(
        self,
        obs_encoding: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple:
        """
        Encode observations and actions to latent distribution.

        Args:
            obs_encoding: Encoded observations [batch, seq_len, hidden_dim]
            actions: Action sequence [batch, chunk_size, action_dim]

        Returns:
            Tuple of (mean, logvar)
        """
        # Pool observation encoding
        obs_pooled = obs_encoding.mean(dim=1)  # [batch, hidden_dim]

        # Flatten actions
        actions_flat = actions.view(actions.size(0), -1)  # [batch, chunk_size * action_dim]

        # Concatenate and encode
        cvae_input = torch.cat([obs_pooled, actions_flat], dim=-1)
        cvae_hidden = self.cvae_encoder(cvae_input)

        mean = self.cvae_mean(cvae_hidden)
        logvar = self.cvae_logvar(cvae_hidden)

        return mean, logvar

    def cvae_reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode_actions(
        self,
        obs_encoding: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode latent and observations to action sequence.

        Args:
            obs_encoding: Encoded observations [batch, seq_len, hidden_dim]
            latent: Latent code [batch, latent_dim]

        Returns:
            Action sequence [batch, chunk_size, action_dim]
        """
        batch_size = obs_encoding.size(0)

        # Project latent
        latent_features = self.latent_proj(latent).unsqueeze(1)  # [batch, 1, hidden_dim]

        # Action queries
        queries = self.action_queries.expand(batch_size, -1, -1)  # [batch, chunk_size, hidden_dim]

        # Add latent to queries
        queries = queries + latent_features

        # Decode
        decoded = self.decoder(queries, obs_encoding)  # [batch, chunk_size, hidden_dim]

        # Predict actions
        actions = self.action_head(decoded)  # [batch, chunk_size, action_dim]

        return actions

    def forward(
        self,
        images: torch.Tensor,
        text: Optional[Union[str, List[str]]] = None,
        proprioception: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of ACT VLA.

        Args:
            images: Input images [batch_size, C, H, W]
            text: Optional text instructions (not used in original ACT)
            proprioception: Optional proprioceptive state [batch_size, state_dim]
            actions: Optional action targets [batch_size, chunk_size, action_dim]
            **kwargs: Additional arguments

        Returns:
            Dictionary containing:
                - 'actions': Predicted action chunks [batch_size, chunk_size, action_dim]
                - 'loss': Training loss (if actions provided)
                - 'kl_loss': KL divergence loss
                - 'recon_loss': Reconstruction loss
        """
        # Encode observations
        obs_encoding = self.encode_observations(images, proprioception)

        if actions is not None:
            # Training mode: use CVAE
            # Reshape actions if needed
            if actions.dim() == 2:
                # Single action provided, expand to chunk
                actions = actions.unsqueeze(1).expand(-1, self.chunk_size, -1)

            # Encode to latent
            mean, logvar = self.cvae_encode(obs_encoding, actions)

            # Reparameterize
            latent = self.cvae_reparameterize(mean, logvar)

            # Decode to actions
            predicted_actions = self.decode_actions(obs_encoding, latent)

            # Compute losses
            # Reconstruction loss
            recon_loss = F.mse_loss(predicted_actions, actions)

            # KL divergence loss
            kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

            # Total loss
            total_loss = recon_loss + self.kl_weight * kl_loss

            return {
                'actions': predicted_actions,
                'loss': total_loss,
                'recon_loss': recon_loss,
                'kl_loss': kl_loss,
                'mean': mean,
                'logvar': logvar,
            }
        else:
            # Inference mode: sample from prior
            batch_size = obs_encoding.size(0)
            latent = torch.randn(
                batch_size, self.latent_dim,
                device=obs_encoding.device
            )

            # Decode to actions
            predicted_actions = self.decode_actions(obs_encoding, latent)

            return {
                'actions': predicted_actions,
                'latent': latent,
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
            Predicted action chunks [batch_size, chunk_size, action_dim]
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

        if 'recon_loss' in predictions:
            losses['recon_loss'] = predictions['recon_loss']

        if 'kl_loss' in predictions:
            losses['kl_loss'] = predictions['kl_loss']

        if 'loss' in predictions:
            losses['total_loss'] = predictions['loss']

        return losses

    def get_config(self) -> Dict[str, Any]:
        """
        Get framework configuration.

        Returns:
            Dictionary containing configuration
        """
        return {
            'type': 'ACTVLA',
            'action_dim': self.action_dim,
            'state_dim': self.state_dim,
            'chunk_size': self.chunk_size,
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim,
            'kl_weight': self.kl_weight,
        }
