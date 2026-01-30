"""Gated fusion mechanism."""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism.

    Learns gating weights to combine multiple modalities adaptively.
    Can handle variable number of inputs with learned importance.

    Args:
        input_dims: List of input dimensions for each modality
        output_dim: Output dimension
        gate_activation: Activation for gate ('sigmoid', 'softmax', 'tanh')
        use_context: Whether to use context for computing gates
        context_dim: Context dimension (if use_context=True)
    """

    def __init__(
        self,
        input_dims: List[int],
        output_dim: int,
        gate_activation: str = 'sigmoid',
        use_context: bool = False,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.gate_activation_name = gate_activation
        self.use_context = use_context
        self.context_dim = context_dim
        self.num_modalities = len(input_dims)

        # Input projections
        self.input_projs = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])

        # Gate network
        if use_context:
            if context_dim is None:
                raise ValueError("context_dim must be specified when use_context=True")
            gate_input_dim = context_dim
        else:
            gate_input_dim = sum(input_dims)

        self.gate_net = nn.Sequential(
            nn.Linear(gate_input_dim, self.num_modalities * 2),
            nn.ReLU(),
            nn.Linear(self.num_modalities * 2, self.num_modalities),
        )

        # Gate activation
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'softmax':
            self.gate_activation = nn.Softmax(dim=-1)
        elif gate_activation == 'tanh':
            self.gate_activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown gate activation: {gate_activation}")

        # Layer normalization
        self.norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        *embeddings: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Fuse embeddings with learned gates.

        Args:
            *embeddings: Variable number of embedding tensors [batch_size, dim_i]
            context: Optional context for gate computation [batch_size, context_dim]
            **kwargs: Additional arguments

        Returns:
            Fused embeddings [batch_size, output_dim]
        """
        if len(embeddings) != self.num_modalities:
            raise ValueError(
                f"Expected {self.num_modalities} embeddings, got {len(embeddings)}"
            )

        # Project all inputs to common dimension
        projected = [proj(emb) for proj, emb in zip(self.input_projs, embeddings)]

        # Compute gates
        if self.use_context:
            if context is None:
                raise ValueError("context must be provided when use_context=True")
            if context.dim() == 3:
                context = context.mean(dim=1)
            gate_input = context
        else:
            gate_input = torch.cat(embeddings, dim=-1)

        gates = self.gate_net(gate_input)
        gates = self.gate_activation(gates)

        # Apply gates and sum
        output = torch.zeros_like(projected[0])
        for i, proj_emb in enumerate(projected):
            gate_weight = gates[:, i:i+1]
            output = output + gate_weight * proj_emb

        # Normalize
        output = self.norm(output)

        return output

    def get_config(self) -> Dict[str, Any]:
        """Get fusion configuration."""
        return {
            'type': 'GatedFusion',
            'input_dims': self.input_dims,
            'output_dim': self.output_dim,
            'gate_activation': self.gate_activation_name,
            'use_context': self.use_context,
            'context_dim': self.context_dim,
        }
