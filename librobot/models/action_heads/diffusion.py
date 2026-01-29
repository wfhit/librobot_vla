"""Diffusion-based action head."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from librobot.models.action_heads.base import BaseActionHead
from librobot.utils.registry import register_action_head


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


@register_action_head("diffusion_transformer", aliases=["diffusion"])
class DiffusionTransformerHead(BaseActionHead):
    """Diffusion-based action head with transformer denoiser.

    Uses DDPM for training and DDIM for fast inference.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        action_horizon: int = 10,
        num_diffusion_steps: int = 100,
        num_inference_steps: int = 10,
        dropout: float = 0.1,
        **kwargs,
    ):
        """Initialize diffusion transformer head.

        Args:
            input_dim: Input feature dimension
            action_dim: Action space dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            action_horizon: Number of action steps to predict
            num_diffusion_steps: Number of diffusion steps for training
            num_inference_steps: Number of denoising steps for inference
            dropout: Dropout probability
        """
        super().__init__(input_dim, action_dim, hidden_dim)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.action_horizon = action_horizon
        self.num_diffusion_steps = num_diffusion_steps
        self.num_inference_steps = num_inference_steps

        # Time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Project action to hidden dim
        self.action_proj = nn.Linear(action_dim, hidden_dim)

        # Project condition features
        self.cond_proj = nn.Linear(input_dim, hidden_dim)

        # Transformer denoiser
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, action_dim)

        # Noise schedule (linear for simplicity)
        self.register_buffer(
            "betas",
            torch.linspace(0.0001, 0.02, num_diffusion_steps),
        )
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(self.alphas, dim=0))

    def forward(
        self,
        features: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training.

        Args:
            features: Input features [batch_size, input_dim]
            actions: Ground truth actions [batch_size, action_horizon, action_dim]

        Returns:
            Dictionary with 'loss'
        """
        if actions is None:
            raise ValueError("Actions required for diffusion training")

        batch_size = features.shape[0]

        # Sample random timesteps
        t = torch.randint(
            0, self.num_diffusion_steps, (batch_size,), device=features.device
        ).long()

        # Add noise to actions
        noise = torch.randn_like(actions)
        alpha_t = self.alphas_cumprod[t].view(-1, 1, 1)
        noisy_actions = torch.sqrt(alpha_t) * actions + torch.sqrt(1 - alpha_t) * noise

        # Predict noise
        predicted_noise = self._denoise(noisy_actions, features, t)

        # Compute loss
        loss = nn.functional.mse_loss(predicted_noise, noise)

        return {"loss": loss, "predicted_noise": predicted_noise}

    def predict(
        self,
        features: torch.Tensor,
        num_samples: int = 1,
        **kwargs,
    ) -> torch.Tensor:
        """Predict actions using DDIM sampling.

        Args:
            features: Input features [batch_size, input_dim]
            num_samples: Number of samples to generate

        Returns:
            Predicted actions [batch_size, num_samples, action_horizon, action_dim]
        """
        batch_size = features.shape[0]
        device = features.device

        # Initialize with noise
        actions = torch.randn(
            batch_size, num_samples, self.action_horizon, self.action_dim,
            device=device,
        )

        # DDIM sampling
        timesteps = torch.linspace(
            self.num_diffusion_steps - 1, 0, self.num_inference_steps, device=device
        ).long()

        for i, t in enumerate(timesteps):
            t_batch = t.repeat(batch_size)

            # Denoise for each sample
            for s in range(num_samples):
                predicted_noise = self._denoise(
                    actions[:, s], features, t_batch
                )

                alpha_t = self.alphas_cumprod[t]
                alpha_t_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)

                # DDIM update
                pred_x0 = (actions[:, s] - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
                direction = torch.sqrt(1 - alpha_t_prev) * predicted_noise
                actions[:, s] = torch.sqrt(alpha_t_prev) * pred_x0 + direction

        return actions

    def _denoise(
        self,
        noisy_actions: torch.Tensor,
        features: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Denoise actions at timestep t.

        Args:
            noisy_actions: Noisy actions [batch_size, action_horizon, action_dim]
            features: Conditioning features [batch_size, input_dim]
            t: Timesteps [batch_size]

        Returns:
            Predicted noise [batch_size, action_horizon, action_dim]
        """
        # Time embedding
        time_emb = self.time_emb(t.float())  # [batch_size, hidden_dim]

        # Project actions
        action_tokens = self.action_proj(noisy_actions)  # [batch_size, action_horizon, hidden_dim]

        # Add time embedding
        action_tokens = action_tokens + time_emb.unsqueeze(1)

        # Project and add condition
        cond_token = self.cond_proj(features).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        tokens = torch.cat([cond_token, action_tokens], dim=1)  # [batch_size, 1+action_horizon, hidden_dim]

        # Transformer
        denoised = self.transformer(tokens)

        # Remove condition token and project to action space
        action_denoised = denoised[:, 1:]  # [batch_size, action_horizon, hidden_dim]
        predicted_noise = self.output_proj(action_denoised)

        return predicted_noise
