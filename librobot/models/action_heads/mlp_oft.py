"""MLP action head for OpenVLA-OFT."""
import torch
import torch.nn as nn
from .base import AbstractActionHead

class MLPActionHead(AbstractActionHead):
    """Simple MLP for action regression."""
    def __init__(self, input_dim: int, action_dim: int, hidden_dims: list = [512, 512]):
        super().__init__(input_dim, action_dim)
        layers = []
        dims = [input_dim] + hidden_dims + [action_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor, **kwargs) -> dict:
        actions = self.mlp(embeddings)
        return {'actions': actions, 'logits': actions}

    def compute_loss(self, predictions: dict, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.nn.functional.mse_loss(predictions['actions'], targets)

    def sample(self, embeddings: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.mlp(embeddings)
