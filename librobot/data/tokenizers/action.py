"""Action tokenizer for robot actions."""

from typing import Optional, Union

import numpy as np


class ActionTokenizer:
    """
    Tokenizer for robot actions.

    Converts continuous action vectors into discrete tokens for
    autoregressive prediction in VLA models.
    """

    def __init__(
        self,
        action_dim: int,
        num_bins: int = 256,
        min_val: Union[float, np.ndarray] = -1.0,
        max_val: Union[float, np.ndarray] = 1.0,
        tokenize_method: str = "uniform",
        action_horizon: int = 1,
        special_tokens: bool = True,
    ):
        """
        Initialize action tokenizer.

        Args:
            action_dim: Dimension of action vector
            num_bins: Number of discretization bins per dimension
            min_val: Minimum value(s) for normalization
            max_val: Maximum value(s) for normalization
            tokenize_method: Tokenization method ("uniform", "mu_law", "vq")
            action_horizon: Number of future actions to predict
            special_tokens: Whether to add special tokens
        """
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.tokenize_method = tokenize_method
        self.action_horizon = action_horizon
        self.special_tokens = special_tokens

        # Handle per-dimension bounds
        self.min_val = np.asarray(min_val)
        self.max_val = np.asarray(max_val)
        if self.min_val.ndim == 0:
            self.min_val = np.full(action_dim, float(self.min_val))
        if self.max_val.ndim == 0:
            self.max_val = np.full(action_dim, float(self.max_val))

        # Token IDs
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.action_token_offset = 3 if special_tokens else 0

        # Vocab size
        self.vocab_size = self.action_token_offset + (num_bins * action_dim)

        # Normalization stats
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

        # VQ codebook (if using VQ tokenization)
        self._codebook: Optional[np.ndarray] = None

    def fit(
        self,
        actions: np.ndarray,
        update_bounds: bool = True
    ) -> 'ActionTokenizer':
        """
        Fit tokenizer to data.

        Args:
            actions: Action data [N, action_dim]
            update_bounds: Whether to update min/max from data

        Returns:
            Self for chaining
        """
        self.mean = np.mean(actions, axis=0)
        self.std = np.std(actions, axis=0) + 1e-6

        if update_bounds:
            self.min_val = np.percentile(actions, 1, axis=0)
            self.max_val = np.percentile(actions, 99, axis=0)

        if self.tokenize_method == "vq":
            self._fit_vq_codebook(actions)

        return self

    def _fit_vq_codebook(self, actions: np.ndarray) -> None:
        """Fit VQ codebook using k-means."""
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.num_bins, random_state=42)
            kmeans.fit(actions)
            self._codebook = kmeans.cluster_centers_
        except ImportError:
            # Fallback: uniform codebook
            self._codebook = np.linspace(
                self.min_val, self.max_val, self.num_bins
            )

    def encode(
        self,
        action: Union[np.ndarray, list[float]],
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Encode action to token IDs.

        Args:
            action: Action vector [action_dim] or [B, action_dim] or [B, T, action_dim]
            normalize: Whether to normalize before encoding

        Returns:
            Token IDs with same batch dimensions
        """
        action = np.asarray(action, dtype=np.float32)
        original_shape = action.shape

        # Flatten to 2D for processing
        if action.ndim == 1:
            action = action[np.newaxis, :]
            squeeze_output = True
        elif action.ndim == 3:
            batch_size, seq_len = action.shape[:2]
            action = action.reshape(-1, self.action_dim)
            squeeze_output = False
        else:
            squeeze_output = False

        # Normalize if requested
        if normalize and self.mean is not None:
            action = (action - self.mean) / self.std

        if self.tokenize_method == "uniform":
            tokens = self._encode_uniform(action)
        elif self.tokenize_method == "mu_law":
            tokens = self._encode_mu_law(action)
        elif self.tokenize_method == "vq":
            tokens = self._encode_vq(action)
        else:
            tokens = self._encode_uniform(action)

        # Reshape back
        if squeeze_output:
            tokens = tokens[0]
        elif len(original_shape) == 3:
            tokens = tokens.reshape(batch_size, seq_len, self.action_dim)

        return tokens

    def _encode_uniform(self, action: np.ndarray) -> np.ndarray:
        """Uniform discretization encoding."""
        # Clip to valid range
        action = np.clip(action, self.min_val, self.max_val)

        # Normalize to [0, 1]
        normalized = (action - self.min_val) / (self.max_val - self.min_val + 1e-8)

        # Discretize
        bin_indices = (normalized * (self.num_bins - 1)).astype(np.int64)
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)

        # Convert to tokens
        tokens = np.zeros_like(bin_indices)
        for d in range(self.action_dim):
            tokens[:, d] = self.action_token_offset + d * self.num_bins + bin_indices[:, d]

        return tokens

    def _encode_mu_law(self, action: np.ndarray, mu: float = 255.0) -> np.ndarray:
        """Mu-law companding encoding for better resolution near zero."""
        # Normalize to [-1, 1]
        normalized = 2 * (action - self.min_val) / (self.max_val - self.min_val + 1e-8) - 1

        # Apply mu-law companding
        compressed = np.sign(normalized) * np.log1p(mu * np.abs(normalized)) / np.log1p(mu)

        # Scale to [0, num_bins-1]
        bin_indices = ((compressed + 1) / 2 * (self.num_bins - 1)).astype(np.int64)
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)

        # Convert to tokens
        tokens = np.zeros_like(bin_indices)
        for d in range(self.action_dim):
            tokens[:, d] = self.action_token_offset + d * self.num_bins + bin_indices[:, d]

        return tokens

    def _encode_vq(self, action: np.ndarray) -> np.ndarray:
        """Vector quantization encoding."""
        if self._codebook is None:
            return self._encode_uniform(action)

        # Find nearest codebook entry for each action
        distances = np.sum(
            (action[:, np.newaxis, :] - self._codebook[np.newaxis, :, :]) ** 2,
            axis=2
        )
        tokens = np.argmin(distances, axis=1) + self.action_token_offset
        return tokens[:, np.newaxis].repeat(self.action_dim, axis=1)

    def decode(
        self,
        tokens: Union[np.ndarray, list[int]],
        denormalize: bool = False,
    ) -> np.ndarray:
        """
        Decode token IDs to action vector.

        Args:
            tokens: Token IDs [action_dim] or [B, action_dim] or [B, T, action_dim]
            denormalize: Whether to denormalize after decoding

        Returns:
            Action vector with same batch dimensions
        """
        tokens = np.asarray(tokens)
        original_shape = tokens.shape

        if tokens.ndim == 1:
            tokens = tokens[np.newaxis, :]
            squeeze_output = True
        elif tokens.ndim == 3:
            batch_size, seq_len = tokens.shape[:2]
            tokens = tokens.reshape(-1, self.action_dim)
            squeeze_output = False
        else:
            squeeze_output = False

        if self.tokenize_method == "uniform":
            action = self._decode_uniform(tokens)
        elif self.tokenize_method == "mu_law":
            action = self._decode_mu_law(tokens)
        elif self.tokenize_method == "vq":
            action = self._decode_vq(tokens)
        else:
            action = self._decode_uniform(tokens)

        # Denormalize
        if denormalize and self.mean is not None:
            action = action * self.std + self.mean

        # Reshape back
        if squeeze_output:
            action = action[0]
        elif len(original_shape) == 3:
            action = action.reshape(batch_size, seq_len, self.action_dim)

        return action

    def _decode_uniform(self, tokens: np.ndarray) -> np.ndarray:
        """Decode uniform discretization."""
        action = np.zeros((tokens.shape[0], self.action_dim), dtype=np.float32)

        for d in range(min(self.action_dim, tokens.shape[1])):
            bin_indices = tokens[:, d] - self.action_token_offset - d * self.num_bins
            bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
            normalized = bin_indices / (self.num_bins - 1)
            action[:, d] = normalized * (self.max_val[d] - self.min_val[d]) + self.min_val[d]

        return action

    def _decode_mu_law(self, tokens: np.ndarray, mu: float = 255.0) -> np.ndarray:
        """Decode mu-law companding."""
        action = np.zeros((tokens.shape[0], self.action_dim), dtype=np.float32)

        for d in range(min(self.action_dim, tokens.shape[1])):
            bin_indices = tokens[:, d] - self.action_token_offset - d * self.num_bins
            bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)

            # Reverse mu-law
            compressed = 2 * bin_indices / (self.num_bins - 1) - 1
            expanded = np.sign(compressed) * (np.power(1 + mu, np.abs(compressed)) - 1) / mu

            # Scale back to original range
            action[:, d] = (expanded + 1) / 2 * (self.max_val[d] - self.min_val[d]) + self.min_val[d]

        return action

    def _decode_vq(self, tokens: np.ndarray) -> np.ndarray:
        """Decode vector quantization."""
        if self._codebook is None:
            return self._decode_uniform(tokens)

        indices = tokens[:, 0] - self.action_token_offset
        indices = np.clip(indices, 0, len(self._codebook) - 1)
        return self._codebook[indices]

    def get_token_for_action_dim(self, dim: int) -> tuple[int, int]:
        """
        Get token range for a specific action dimension.

        Args:
            dim: Action dimension index

        Returns:
            Tuple of (start_token_id, end_token_id)
        """
        start = self.action_token_offset + dim * self.num_bins
        end = start + self.num_bins
        return start, end

    def __call__(
        self,
        action: Union[np.ndarray, list[float]],
        **kwargs
    ) -> np.ndarray:
        """Encode action (alias)."""
        return self.encode(action, **kwargs)


__all__ = ['ActionTokenizer']
