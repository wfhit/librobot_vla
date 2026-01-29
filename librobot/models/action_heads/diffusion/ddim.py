"""DDIM (Denoising Diffusion Implicit Models) action head."""

from typing import Any, Dict
import torch
import torch.nn as nn
from .ddpm import DDPMActionHead


class DDIMActionHead(DDPMActionHead):
    """DDIM action head with faster sampling."""
    
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256,
                 num_timesteps: int = 100, inference_steps: int = 10):
        super().__init__(input_dim, action_dim, hidden_dim, num_timesteps)
        self.inference_steps = inference_steps
    
    def sample(self, embeddings: torch.Tensor, temperature: float = 1.0, eta: float = 0.0, **kwargs) -> torch.Tensor:
        batch_size = embeddings.size(0)
        actions = torch.randn(batch_size, self.action_dim, device=embeddings.device) * temperature
        
        # DDIM sampling with fewer steps
        timesteps = torch.linspace(self.num_timesteps - 1, 0, self.inference_steps, dtype=torch.long, device=embeddings.device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=embeddings.device)
            t_normalized = t_batch.float() / self.num_timesteps
            
            model_input = torch.cat([actions, embeddings, t_normalized.unsqueeze(1)], dim=-1)
            predicted_noise = self.denoiser(model_input)
            
            alpha_cumprod_t = self.alphas_cumprod[t]
            
            if i < len(timesteps) - 1:
                alpha_cumprod_t_prev = self.alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_cumprod_t_prev = torch.tensor(1.0, device=embeddings.device)
            
            pred_x0 = (actions - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            
            sigma_t = eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))
            
            dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigma_t ** 2) * predicted_noise
            noise = torch.randn_like(actions) if i < len(timesteps) - 1 else torch.zeros_like(actions)
            
            actions = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + dir_xt + sigma_t * noise
        
        return actions
