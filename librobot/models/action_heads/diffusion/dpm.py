"""DPM++ Solver action head."""

import torch

from .ddpm import DDPMActionHead


class DPMActionHead(DDPMActionHead):
    """DPM++ Solver for fast sampling."""

    def sample(self, embeddings: torch.Tensor, temperature: float = 1.0, **kwargs) -> torch.Tensor:
        batch_size = embeddings.size(0)
        x = torch.randn(batch_size, self.action_dim, device=embeddings.device) * temperature
        for t in reversed(range(0, self.num_timesteps, 5)):
            t_batch = torch.full((batch_size,), t, device=embeddings.device)
            t_norm = t_batch.float() / self.num_timesteps
            inp = torch.cat([x, embeddings, t_norm.unsqueeze(1)], dim=-1)
            noise = self.denoiser(inp)
            alpha = self.alphas_cumprod[t]
            x = (x - torch.sqrt(1 - alpha) * noise) / torch.sqrt(alpha)
        return x
