"""Consistency Models for fast generation."""
import torch
import torch.nn as nn
from ..base import AbstractActionHead

class ConsistencyActionHead(AbstractActionHead):
    """Consistency model for single-step generation."""
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__(input_dim, action_dim)
        self.model = nn.Sequential(
            nn.Linear(action_dim + input_dim, hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), nn.Linear(hidden_dim, action_dim))

    def forward(self, embeddings: torch.Tensor, **kwargs) -> dict:
        return {'embeddings': embeddings}

    def compute_loss(self, predictions: dict, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        emb = predictions['embeddings']
        noise = torch.randn_like(targets) * 0.1
        inp = torch.cat([targets + noise, emb], dim=-1)
        pred = self.model(inp)
        return torch.nn.functional.mse_loss(pred, targets)

    def sample(self, embeddings: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size = embeddings.size(0)
        noise = torch.randn(batch_size, self.action_dim, device=embeddings.device)
        inp = torch.cat([noise, embeddings], dim=-1)
        return self.model(inp)
