"""ACT (Action Chunking Transformer) decoder."""
import torch
import torch.nn as nn
from .base import AbstractActionHead

class ACTActionHead(AbstractActionHead):
    """Transformer decoder for action chunking."""
    def __init__(self, input_dim: int, action_dim: int, chunk_size: int = 10, num_layers: int = 4):
        super().__init__(input_dim, action_dim)
        self.chunk_size = chunk_size
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(input_dim, 8, input_dim*4, batch_first=True),
            num_layers)
        self.action_proj = nn.Linear(input_dim, action_dim)
        self.action_queries = nn.Parameter(torch.randn(1, chunk_size, input_dim))

    def forward(self, embeddings: torch.Tensor, **kwargs) -> dict:
        queries = self.action_queries.expand(embeddings.size(0), -1, -1)
        memory = embeddings.unsqueeze(1)
        decoded = self.decoder(queries, memory)
        actions = self.action_proj(decoded)
        return {'actions': actions[:, 0], 'action_sequence': actions}

    def compute_loss(self, predictions: dict, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.nn.functional.mse_loss(predictions['actions'], targets)

    def sample(self, embeddings: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.forward(embeddings)['actions']
