"""Autoregressive FAST tokenized actions."""

import torch
import torch.nn as nn

from .base import AbstractActionHead


class FASTActionHead(AbstractActionHead):
    """FAST autoregressive discrete action head."""

    def __init__(self, input_dim: int, action_dim: int, num_bins: int = 256, num_steps: int = 10):
        super().__init__(input_dim, action_dim)
        self.num_bins = num_bins
        self.num_steps = num_steps
        self.decoder = nn.GRU(input_dim + num_bins, input_dim, batch_first=True)
        self.action_head = nn.Linear(input_dim, num_bins * action_dim)

    def forward(self, embeddings: torch.Tensor, **kwargs) -> dict:
        return {"embeddings": embeddings}

    def compute_loss(self, predictions: dict, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        # Simplified - discretize and predict
        emb = predictions["embeddings"]
        targets_discrete = ((targets + 1) * (self.num_bins // 2)).long().clamp(0, self.num_bins - 1)
        logits = self.action_head(emb).view(-1, self.action_dim, self.num_bins)
        return torch.nn.functional.cross_entropy(logits, targets_discrete)

    def sample(self, embeddings: torch.Tensor, **kwargs) -> torch.Tensor:
        logits = self.action_head(embeddings).view(-1, self.action_dim, self.num_bins)
        indices = logits.argmax(dim=-1)
        return (indices.float() / (self.num_bins // 2)) - 1
