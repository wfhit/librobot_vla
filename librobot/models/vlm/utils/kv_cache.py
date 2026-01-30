"""KV cache for efficient inference."""
import torch
from typing import Optional, Tuple

class KVCache:
    """Key-Value cache for transformer inference."""
    def __init__(self, max_batch_size: int, max_seq_len: int, num_heads: int, head_dim: int, device: str = 'cuda'):
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.cache_k = torch.zeros(max_batch_size, num_heads, max_seq_len, head_dim, device=device)
        self.cache_v = torch.zeros(max_batch_size, num_heads, max_seq_len, head_dim, device=device)
        self.seq_len = 0

    def update(self, k: torch.Tensor, v: torch.Tensor, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new key-value pairs."""
        batch_size, num_heads, seq_len, head_dim = k.shape
        self.cache_k[:batch_size, :, start_pos:start_pos+seq_len] = k
        self.cache_v[:batch_size, :, start_pos:start_pos+seq_len] = v
        self.seq_len = start_pos + seq_len
        return self.cache_k[:batch_size, :, :self.seq_len], self.cache_v[:batch_size, :, :self.seq_len]

    def reset(self):
        """Reset cache."""
        self.seq_len = 0
