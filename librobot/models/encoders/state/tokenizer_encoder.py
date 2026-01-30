"""π0-style state tokenizer encoder."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..base import AbstractEncoder


class TokenizerStateEncoder(AbstractEncoder):
    """
    State tokenizer encoder as used in π0.

    Tokenizes continuous state into discrete tokens using learned codebook,
    then embeds tokens for processing by transformer models.

    Args:
        input_dim: Input state dimension
        output_dim: Output embedding dimension
        num_tokens: Number of tokens in codebook
        hidden_dim: Hidden dimension for encoding
        commitment_cost: Weight for commitment loss
        use_ema: Whether to use exponential moving average for codebook updates
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_tokens: int = 1024,
        hidden_dim: int = 256,
        commitment_cost: float = 0.25,
        use_ema: bool = True,
    ):
        super().__init__(output_dim)
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.hidden_dim = hidden_dim
        self.commitment_cost = commitment_cost
        self.use_ema = use_ema

        # Pre-quantization encoder
        self.pre_quant = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Vector quantization codebook
        self.codebook = nn.Embedding(num_tokens, hidden_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_tokens, 1.0 / num_tokens)

        # Post-quantization projection
        self.post_quant = nn.Linear(hidden_dim, output_dim)

        # EMA parameters for codebook updates
        if use_ema:
            self.register_buffer('ema_cluster_size', torch.zeros(num_tokens))
            self.register_buffer('ema_weight', self.codebook.weight.data.clone())
            self.ema_decay = 0.99
            self.ema_epsilon = 1e-5

    def forward(
        self,
        inputs: torch.Tensor,
        return_indices: bool = False,
        **kwargs
    ) -> torch.Tensor | tuple:
        """
        Encode state through tokenization.

        Args:
            inputs: State tensor [batch_size, input_dim] or [batch_size, seq_len, input_dim]
            return_indices: Whether to return token indices
            **kwargs: Additional arguments

        Returns:
            Encoded embeddings [batch_size, output_dim] or [batch_size, seq_len, output_dim]
            If return_indices=True, returns (embeddings, indices, quantization_loss)
        """
        original_shape = inputs.shape
        flatten = inputs.dim() == 3

        if flatten:
            batch_size, seq_len, _ = inputs.shape
            inputs = inputs.reshape(batch_size * seq_len, -1)

        # Pre-quantization encoding
        z = self.pre_quant(inputs)

        # Quantize
        z_q, indices, vq_loss = self._quantize(z)

        # Post-quantization projection
        output = self.post_quant(z_q)

        # Reshape if needed
        if flatten:
            output = output.reshape(batch_size, seq_len, -1)
            indices = indices.reshape(batch_size, seq_len)

        if return_indices:
            return output, indices, vq_loss

        return output

    def _quantize(
        self,
        z: torch.Tensor,
    ) -> tuple:
        """
        Vector quantization using learned codebook.

        Args:
            z: Continuous embeddings [batch_size, hidden_dim]

        Returns:
            Tuple of (quantized_embeddings, indices, vq_loss)
        """
        # Flatten
        z_flattened = z.view(-1, self.hidden_dim)

        # Calculate distances to codebook vectors
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self.codebook.weight.t())
        )

        # Find closest codebook entry
        indices = torch.argmin(distances, dim=1)
        z_q = self.codebook(indices).view(z.shape)

        # Compute VQ loss
        if self.training:
            # Commitment loss
            commitment_loss = self.commitment_cost * torch.mean((z_q.detach() - z) ** 2)

            # Codebook loss (only if not using EMA)
            if not self.use_ema:
                codebook_loss = torch.mean((z_q - z.detach()) ** 2)
                vq_loss = codebook_loss + commitment_loss
            else:
                vq_loss = commitment_loss

                # Update EMA
                self._update_ema(z_flattened, indices)
        else:
            vq_loss = torch.tensor(0.0, device=z.device)

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, indices, vq_loss

    def _update_ema(self, z: torch.Tensor, indices: torch.Tensor):
        """Update EMA statistics for codebook."""
        with torch.no_grad():
            # Update cluster sizes
            encodings = torch.zeros(indices.shape[0], self.num_tokens, device=z.device)
            encodings.scatter_(1, indices.unsqueeze(1), 1)

            self.ema_cluster_size = self.ema_cluster_size * self.ema_decay + \
                                   (1 - self.ema_decay) * torch.sum(encodings, 0)

            # Laplace smoothing
            n = torch.sum(self.ema_cluster_size)
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.ema_epsilon)
                / (n + self.num_tokens * self.ema_epsilon) * n
            )

            # Update embeddings
            dw = torch.matmul(encodings.t(), z)
            self.ema_weight = self.ema_weight * self.ema_decay + (1 - self.ema_decay) * dw

            self.codebook.weight.data = self.ema_weight / self.ema_cluster_size.unsqueeze(1)

    def get_output_shape(self, input_shape: tuple) -> tuple:
        """Get output shape for given input shape."""
        if len(input_shape) == 2:
            return (input_shape[0], self.output_dim)
        elif len(input_shape) == 3:
            return (input_shape[0], input_shape[1], self.output_dim)
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")

    @property
    def config(self) -> Dict[str, Any]:
        """Get encoder configuration."""
        return {
            'type': 'TokenizerStateEncoder',
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'num_tokens': self.num_tokens,
            'hidden_dim': self.hidden_dim,
            'commitment_cost': self.commitment_cost,
            'use_ema': self.use_ema,
        }
