"""1D UNet for diffusion denoising."""
import torch
import torch.nn as nn


class UNet1D(nn.Module):
    """1D UNet denoiser."""
    def __init__(self, input_dim: int, channels: list = [64, 128, 256]):
        super().__init__()
        self.input_dim = input_dim
        self.down = nn.ModuleList([nn.Conv1d(input_dim if i==0 else channels[i-1], channels[i], 3, 2, 1) for i in range(len(channels))])
        self.up = nn.ModuleList([nn.ConvTranspose1d(channels[i], channels[i-1] if i>0 else input_dim, 4, 2, 1) for i in range(len(channels)-1, -1, -1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(2)  # Add seq dim
        skips = []
        for down in self.down:
            x = torch.relu(down(x))
            skips.append(x)
        for up in self.up:
            x = torch.relu(up(x))
            if skips: x = x + skips.pop()
        return x.squeeze(2)
