"""DDPM (Denoising Diffusion Probabilistic Models) action head."""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import AbstractActionHead


class DDPMActionHead(AbstractActionHead):
    """
    DDPM action head for diffusion-based action prediction.

    Implements DDPM scheduler with cosine noise schedule.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_timesteps: int = 100,
        beta_schedule: str = 'cosine',
    ):
        super().__init__(input_dim, action_dim)
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps

        # Denoiser network
        self.denoiser = nn.Sequential(
            nn.Linear(action_dim + input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Noise schedule
        if beta_schedule == 'linear':
            betas = torch.linspace(1e-4, 0.02, num_timesteps)
        elif beta_schedule == 'cosine':
            betas = self._cosine_beta_schedule(num_timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def forward(self, embeddings: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        return {'embeddings': embeddings}

    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: torch.Tensor, **kwargs) -> torch.Tensor:
        embeddings = predictions['embeddings']
        batch_size = embeddings.size(0)

        # Sample timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=embeddings.device)

        # Add noise
        noise = torch.randn_like(targets)
        noisy_actions = self.sqrt_alphas_cumprod[t].view(-1, 1) * targets + \
                       self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1) * noise

        # Predict noise
        t_normalized = t.float() / self.num_timesteps
        model_input = torch.cat([noisy_actions, embeddings, t_normalized.unsqueeze(1)], dim=-1)
        predicted_noise = self.denoiser(model_input)

        return F.mse_loss(predicted_noise, noise)

    def sample(self, embeddings: torch.Tensor, temperature: float = 1.0, **kwargs) -> torch.Tensor:
        batch_size = embeddings.size(0)
        actions = torch.randn(batch_size, self.action_dim, device=embeddings.device) * temperature

        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=embeddings.device)
            t_normalized = t_batch.float() / self.num_timesteps

            model_input = torch.cat([actions, embeddings, t_normalized.unsqueeze(1)], dim=-1)
            predicted_noise = self.denoiser(model_input)

            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]

            if t > 0:
                noise = torch.randn_like(actions)
            else:
                noise = torch.zeros_like(actions)

            actions = (actions - beta / torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha)
            actions = actions + torch.sqrt(beta) * noise

        return actions
