"""Attention sink for streaming inference."""

import torch
import torch.nn as nn


class AttentionSink(nn.Module):
    """Attention sink tokens for streaming."""

    def __init__(self, num_sink_tokens: int = 4, window_size: int = 1024):
        super().__init__()
        self.num_sink_tokens = num_sink_tokens
        self.window_size = window_size

    def forward(self, k: torch.Tensor, v: torch.Tensor, start_pos: int = 0) -> tuple:
        """Apply attention sink mechanism."""
        if start_pos + k.size(2) <= self.window_size + self.num_sink_tokens:
            return k, v

        # Keep sink tokens and recent window
        k_sink = k[:, :, : self.num_sink_tokens]
        k_window = k[:, :, -(self.window_size) :]
        k = torch.cat([k_sink, k_window], dim=2)

        v_sink = v[:, :, : self.num_sink_tokens]
        v_window = v[:, :, -(self.window_size) :]
        v = torch.cat([v_sink, v_window], dim=2)

        return k, v
