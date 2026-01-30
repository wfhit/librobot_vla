"""Base flow matching model."""

import torch
import torch.nn as nn

from ..base import AbstractActionHead


class FlowMatchingHead(AbstractActionHead):
    """Flow matching action head."""

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__(input_dim, action_dim)
        self.velocity_net = nn.Sequential(
            nn.Linear(action_dim + input_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, embeddings: torch.Tensor, **kwargs) -> dict:
        return {"embeddings": embeddings}

    def compute_loss(self, predictions: dict, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        emb = predictions["embeddings"]
        t = torch.rand(targets.size(0), 1, device=targets.device)
        x0 = torch.randn_like(targets)
        xt = (1 - t) * x0 + t * targets
        velocity = targets - x0
        inp = torch.cat([xt, emb, t], dim=-1)
        pred_v = self.velocity_net(inp)
        return torch.nn.functional.mse_loss(pred_v, velocity)

    def sample(self, embeddings: torch.Tensor, steps: int = 50, **kwargs) -> torch.Tensor:
        x = torch.randn(embeddings.size(0), self.action_dim, device=embeddings.device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((embeddings.size(0), 1), i * dt, device=embeddings.device)
            inp = torch.cat([x, embeddings, t], dim=-1)
            v = self.velocity_net(inp)
            x = x + v * dt
        return x
