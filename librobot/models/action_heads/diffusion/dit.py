"""Diffusion Transformer (DiT)."""
import torch
import torch.nn as nn


class DiT(nn.Module):
    """Diffusion Transformer denoiser."""
    def __init__(self, dim: int, num_layers: int = 4, num_heads: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(dim, num_heads, dim*4, batch_first=True) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # Add seq dim
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(1)
