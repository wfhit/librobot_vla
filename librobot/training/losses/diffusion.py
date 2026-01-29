"""Diffusion-specific loss functions."""

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .base import AbstractLoss


class DiffusionLoss(AbstractLoss):
    """Loss for diffusion-based action prediction (e.g., GR00T style)."""
    
    def __init__(
        self,
        weight: float = 1.0,
        prediction_type: str = "epsilon",
        loss_type: str = "mse",
        snr_gamma: Optional[float] = None,
    ):
        """
        Args:
            weight: Loss weight
            prediction_type: What model predicts ("epsilon", "v_prediction", "sample")
            loss_type: Base loss type ("mse", "l1", "smooth_l1")
            snr_gamma: SNR weighting factor (for min-SNR weighting)
        """
        super().__init__(weight=weight)
        self.prediction_type = prediction_type
        self.loss_type = loss_type
        self.snr_gamma = snr_gamma
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """Compute diffusion loss."""
        pred = predictions.get('noise_pred', predictions.get('pred'))
        target = targets.get('noise', targets.get('target'))
        timesteps = targets.get('timesteps')
        
        if pred is None or target is None:
            return torch.tensor(0.0)
        
        # Compute base loss
        if self.loss_type == "mse":
            loss = F.mse_loss(pred, target, reduction='none')
        elif self.loss_type == "l1":
            loss = F.l1_loss(pred, target, reduction='none')
        else:
            loss = F.smooth_l1_loss(pred, target, reduction='none')
        
        # Average over non-batch dimensions
        loss = loss.mean(dim=list(range(1, loss.dim())))
        
        # Apply SNR weighting if specified
        if self.snr_gamma is not None and timesteps is not None:
            snr = self._compute_snr(timesteps)
            snr_weight = torch.minimum(
                snr,
                torch.ones_like(snr) * self.snr_gamma
            ) / snr
            loss = loss * snr_weight
        
        return loss.mean()
    
    def _compute_snr(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute signal-to-noise ratio for timesteps."""
        # Assuming linear beta schedule
        alphas_cumprod = self._get_alphas_cumprod(timesteps.max().item() + 1)
        alphas_cumprod = alphas_cumprod.to(timesteps.device)[timesteps]
        return alphas_cumprod / (1 - alphas_cumprod)
    
    def _get_alphas_cumprod(self, num_timesteps: int) -> torch.Tensor:
        """Get cumulative alpha values for diffusion."""
        betas = torch.linspace(0.0001, 0.02, num_timesteps)
        alphas = 1 - betas
        return torch.cumprod(alphas, dim=0)


class DDPMLoss(AbstractLoss):
    """DDPM-style denoising loss."""
    
    def __init__(
        self,
        weight: float = 1.0,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule: str = "linear",
    ):
        """
        Args:
            weight: Loss weight
            num_timesteps: Number of diffusion steps
            beta_start: Starting beta
            beta_end: Ending beta
            schedule: Beta schedule type
        """
        super().__init__(weight=weight)
        self.num_timesteps = num_timesteps
        
        # Compute beta schedule
        if schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule == "cosine":
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine beta schedule."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        noise_pred = predictions.get('noise_pred')
        noise = targets.get('noise')
        
        if noise_pred is None or noise is None:
            return torch.tensor(0.0)
        
        return F.mse_loss(noise_pred, noise)


class ScoreMatchingLoss(AbstractLoss):
    """Score matching loss for score-based diffusion."""
    
    def __init__(
        self,
        weight: float = 1.0,
        sigma_min: float = 0.01,
        sigma_max: float = 50.0,
    ):
        super().__init__(weight=weight)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        score_pred = predictions.get('score')
        noise = targets.get('noise')
        sigma = targets.get('sigma')
        
        if score_pred is None or noise is None:
            return torch.tensor(0.0)
        
        if sigma is None:
            sigma = torch.ones_like(noise[..., :1])
        
        # Score should be -noise/sigma^2
        target_score = -noise / (sigma ** 2 + 1e-8)
        
        # Weight by sigma^2
        loss = ((score_pred - target_score) ** 2 * sigma ** 2).mean()
        
        return loss


__all__ = [
    'DiffusionLoss',
    'DDPMLoss',
    'ScoreMatchingLoss',
]
