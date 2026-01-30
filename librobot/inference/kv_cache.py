"""KV cache management for efficient transformer inference."""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from librobot.utils import get_logger


logger = get_logger(__name__)


class KVCache:
    """
    Key-Value cache for transformer-based models.

    Stores past key-value pairs to avoid recomputing attention
    for previously processed tokens, significantly speeding up
    autoregressive generation.
    """

    def __init__(
        self,
        max_length: Optional[int] = None,
        num_layers: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize KV cache.

        Args:
            max_length: Maximum sequence length to cache
            num_layers: Number of transformer layers
            device: Device to store cache tensors
        """
        self.max_length = max_length
        self.num_layers = num_layers
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Cache storage: List of (key, value) tuples for each layer
        self._cache: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self._current_length = 0

    def get(self) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Get cached key-value pairs.

        Returns:
            List of (key, value) tuples for each layer, or None if empty
        """
        return self._cache

    def update(
        self,
        new_kv: Optional[Union[List[Tuple[torch.Tensor, torch.Tensor]], Tuple]],
    ) -> None:
        """
        Update cache with new key-value pairs.

        Args:
            new_kv: New key-value pairs from model forward pass
                Can be:
                - List of (key, value) tuples (one per layer)
                - Single tuple of (keys, values) across all layers
                - None (cache remains unchanged)
        """
        if new_kv is None:
            return

        # Handle different input formats
        if isinstance(new_kv, tuple) and len(new_kv) == 2:
            # Single tuple format: convert to list
            if isinstance(new_kv[0], (list, tuple)):
                new_kv = list(new_kv)
            else:
                new_kv = [new_kv]

        # Initialize cache if first update
        if self._cache is None:
            self._cache = new_kv
            if new_kv and len(new_kv) > 0:
                self._current_length = new_kv[0][0].shape[-2]  # seq_len dimension
        else:
            # Concatenate with existing cache
            updated_cache = []
            for i, (new_k, new_v) in enumerate(new_kv):
                if i < len(self._cache):
                    cached_k, cached_v = self._cache[i]
                    # Concatenate along sequence dimension (typically dim=-2)
                    concat_k = torch.cat([cached_k, new_k], dim=-2)
                    concat_v = torch.cat([cached_v, new_v], dim=-2)

                    # Truncate if exceeds max_length
                    if self.max_length is not None:
                        if concat_k.shape[-2] > self.max_length:
                            concat_k = concat_k[..., -self.max_length:, :]
                            concat_v = concat_v[..., -self.max_length:, :]

                    updated_cache.append((concat_k, concat_v))
                else:
                    updated_cache.append((new_k, new_v))

            self._cache = updated_cache
            if updated_cache and len(updated_cache) > 0:
                self._current_length = updated_cache[0][0].shape[-2]

        logger.debug(f"KV cache updated: length={self._current_length}")

    def clear(self) -> None:
        """Clear the cache."""
        self._cache = None
        self._current_length = 0
        logger.debug("KV cache cleared")

    def get_length(self) -> int:
        """
        Get current cache sequence length.

        Returns:
            Current cached sequence length
        """
        return self._current_length

    def is_empty(self) -> bool:
        """
        Check if cache is empty.

        Returns:
            True if cache is empty
        """
        return self._cache is None or self._current_length == 0

    def to(self, device: Union[str, torch.device]) -> "KVCache":
        """
        Move cache to device.

        Args:
            device: Target device

        Returns:
            Self for chaining
        """
        self.device = torch.device(device) if isinstance(device, str) else device

        if self._cache is not None:
            self._cache = [
                (k.to(self.device), v.to(self.device))
                for k, v in self._cache
            ]

        return self

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage statistics.

        Returns:
            Dictionary with memory usage in MB
        """
        if self._cache is None:
            return {"total_mb": 0.0, "per_layer_mb": 0.0}

        total_bytes = 0
        for k, v in self._cache:
            total_bytes += k.element_size() * k.nelement()
            total_bytes += v.element_size() * v.nelement()

        total_mb = total_bytes / (1024 * 1024)
        per_layer_mb = total_mb / len(self._cache) if self._cache else 0.0

        return {
            "total_mb": total_mb,
            "per_layer_mb": per_layer_mb,
            "num_layers": len(self._cache),
            "sequence_length": self._current_length,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"KVCache(length={self._current_length}, "
            f"num_layers={len(self._cache) if self._cache else 0}, "
            f"device={self.device})"
        )


class MultiHeadKVCache:
    """
    KV cache with support for multiple attention heads.

    Provides more fine-grained control over caching with
    per-head management.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize multi-head KV cache.

        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads per layer
            head_dim: Dimension of each attention head
            max_length: Maximum sequence length
            device: Device to store cache
            dtype: Data type for cache tensors
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_length = max_length
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float32

        # Pre-allocate cache tensors for efficiency
        if max_length is not None:
            self._preallocate_cache()
        else:
            self._cache_keys = None
            self._cache_values = None

        self._current_length = 0

    def _preallocate_cache(self) -> None:
        """Pre-allocate cache tensors with maximum length."""
        # TODO: Implement pre-allocation for better memory efficiency
        # Shape: (num_layers, batch_size, num_heads, max_length, head_dim)
        # Note: batch_size can be handled dynamically
        self._cache_keys = None
        self._cache_values = None

    def get(self, layer_idx: Optional[int] = None) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get cached key-value pairs.

        Args:
            layer_idx: Optional layer index. If None, return all layers.

        Returns:
            Tuple of (keys, values) for specified layer(s)
        """
        # TODO: Implement per-layer retrieval
        return None

    def update(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """
        Update cache for specific layer.

        Args:
            layer_idx: Layer index to update
            keys: New key tensor
            values: New value tensor
        """
        # TODO: Implement per-layer update with efficient tensor operations
        pass

    def clear(self, layer_idx: Optional[int] = None) -> None:
        """
        Clear cache.

        Args:
            layer_idx: Optional layer index. If None, clear all layers.
        """
        if layer_idx is None:
            self._cache_keys = None
            self._cache_values = None
            self._current_length = 0
        else:
            # TODO: Implement per-layer clearing
            pass


class StaticKVCache:
    """
    Static KV cache for fixed-size contexts.

    Optimized for scenarios where context length is known in advance,
    using pre-allocated tensors for maximum efficiency.
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_heads: int,
        max_length: int,
        head_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize static KV cache.

        Args:
            num_layers: Number of transformer layers
            batch_size: Batch size
            num_heads: Number of attention heads
            max_length: Maximum sequence length
            head_dim: Dimension of each head
            device: Device to store cache
            dtype: Data type for cache
        """
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_length = max_length
        self.head_dim = head_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.float16

        # Pre-allocate cache tensors
        self._keys = torch.zeros(
            num_layers, batch_size, num_heads, max_length, head_dim,
            device=self.device, dtype=self.dtype
        )
        self._values = torch.zeros(
            num_layers, batch_size, num_heads, max_length, head_dim,
            device=self.device, dtype=self.dtype
        )

        self._current_pos = 0

    def get_slice(
        self,
        layer_idx: int,
        start_pos: int,
        end_pos: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cache slice for specific layer and position range.

        Args:
            layer_idx: Layer index
            start_pos: Start position
            end_pos: End position

        Returns:
            Tuple of (keys, values) slice
        """
        return (
            self._keys[layer_idx, :, :, start_pos:end_pos, :],
            self._values[layer_idx, :, :, start_pos:end_pos, :],
        )

    def set_slice(
        self,
        layer_idx: int,
        position: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        """
        Set cache values at specific position.

        Args:
            layer_idx: Layer index
            position: Position to set
            keys: Key tensor to store
            values: Value tensor to store
        """
        seq_len = keys.shape[-2]
        self._keys[layer_idx, :, :, position:position+seq_len, :] = keys
        self._values[layer_idx, :, :, position:position+seq_len, :] = values
        self._current_pos = max(self._current_pos, position + seq_len)

    def clear(self) -> None:
        """Clear cache by resetting to zeros."""
        self._keys.zero_()
        self._values.zero_()
        self._current_pos = 0

    def get_current_length(self) -> int:
        """Get current cache length."""
        return self._current_pos
