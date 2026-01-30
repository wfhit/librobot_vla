"""LSTM encoder for temporal history."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..base import AbstractEncoder


class LSTMHistoryEncoder(AbstractEncoder):
    """
    LSTM encoder for temporal history.

    Processes sequential history using LSTM for temporal modeling.
    Captures long-term dependencies in robot trajectories.

    Args:
        input_dim: Input dimension per timestep
        output_dim: Output embedding dimension
        num_layers: Number of LSTM layers
        hidden_dim: Hidden dimension of LSTM
        dropout: Dropout rate between LSTM layers
        bidirectional: Whether to use bidirectional LSTM
        pooling: Output pooling method ('last', 'mean', 'max')
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        bidirectional: bool = False,
        pooling: str = 'last',
    ):
        super().__init__(output_dim)
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        self.bidirectional = bidirectional
        self.pooling_method = pooling

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Output projection
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.output_proj = nn.Linear(lstm_output_dim, output_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode history with LSTM.

        Args:
            inputs: History tensor [batch_size, seq_len, input_dim]
            lengths: Optional sequence lengths [batch_size]
            **kwargs: Additional arguments

        Returns:
            Encoded embeddings [batch_size, output_dim]
        """
        batch_size, seq_len, _ = inputs.shape

        # Pack sequences if lengths provided
        if lengths is not None:
            # Ensure lengths are on CPU for pack_padded_sequence
            lengths_cpu = lengths.cpu()
            packed_input = pack_padded_sequence(
                inputs,
                lengths_cpu,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, (hidden, cell) = self.lstm(packed_input)
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            output, (hidden, cell) = self.lstm(inputs)

        # Pool LSTM outputs
        if self.pooling_method == 'last':
            if lengths is not None:
                # Get last valid timestep for each sequence
                indices = (lengths - 1).long().unsqueeze(1).unsqueeze(2)
                indices = indices.expand(-1, -1, output.size(2))
                pooled = output.gather(1, indices).squeeze(1)
            else:
                pooled = output[:, -1]

        elif self.pooling_method == 'mean':
            if lengths is not None:
                # Masked mean
                mask = torch.arange(seq_len, device=inputs.device).unsqueeze(0) < lengths.unsqueeze(1)
                pooled = (output * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(1)
            else:
                pooled = output.mean(dim=1)

        elif self.pooling_method == 'max':
            if lengths is not None:
                mask = torch.arange(seq_len, device=inputs.device).unsqueeze(0) < lengths.unsqueeze(1)
                output = output.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            pooled = output.max(dim=1)[0]

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")

        # Project to output dimension
        output = self.output_proj(pooled)

        return output

    def get_output_shape(self, input_shape: tuple) -> tuple:
        """Get output shape for given input shape."""
        if len(input_shape) == 3:
            return (input_shape[0], self.output_dim)
        else:
            raise ValueError(f"Expected 3D input, got shape: {input_shape}")

    @property
    def config(self) -> Dict[str, Any]:
        """Get encoder configuration."""
        return {
            'type': 'LSTMHistoryEncoder',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout_rate,
            'bidirectional': self.bidirectional,
            'pooling': self.pooling_method,
        }
