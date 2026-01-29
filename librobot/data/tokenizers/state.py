"""State tokenizer for proprioceptive data."""

from typing import Any, Dict, List, Optional, Union
import numpy as np


class StateTokenizer:
    """
    Tokenizer for robot proprioceptive state.
    
    Converts continuous state vectors into discrete tokens for
    transformer-based models.
    """
    
    def __init__(
        self,
        state_dim: int,
        num_bins: int = 256,
        min_val: float = -1.0,
        max_val: float = 1.0,
        tokenize_method: str = "uniform",
        special_tokens: bool = True,
    ):
        """
        Initialize state tokenizer.
        
        Args:
            state_dim: Dimension of state vector
            num_bins: Number of discretization bins per dimension
            min_val: Minimum value for normalization
            max_val: Maximum value for normalization
            tokenize_method: Method for tokenization ("uniform", "learned")
            special_tokens: Whether to add special tokens
        """
        self.state_dim = state_dim
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val
        self.tokenize_method = tokenize_method
        self.special_tokens = special_tokens
        
        # Token IDs
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.state_token_offset = 3 if special_tokens else 0
        
        # Vocab size: special tokens + (num_bins * state_dim)
        self.vocab_size = self.state_token_offset + (num_bins * state_dim)
        
        # Normalization stats (can be set from data)
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
    
    def fit(self, states: np.ndarray) -> 'StateTokenizer':
        """
        Fit tokenizer to data for computing normalization statistics.
        
        Args:
            states: State data [N, state_dim]
            
        Returns:
            Self for chaining
        """
        self.mean = np.mean(states, axis=0)
        self.std = np.std(states, axis=0) + 1e-6
        return self
    
    def encode(
        self,
        state: Union[np.ndarray, List[float]],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode state to token IDs.
        
        Args:
            state: State vector [state_dim] or batch [B, state_dim]
            normalize: Whether to normalize before encoding
            
        Returns:
            Token IDs [state_dim] or [B, state_dim]
        """
        state = np.asarray(state)
        is_batch = state.ndim == 2
        
        if not is_batch:
            state = state[np.newaxis, :]
        
        # Normalize
        if normalize and self.mean is not None:
            state = (state - self.mean) / self.std
        
        # Clip to valid range
        state = np.clip(state, self.min_val, self.max_val)
        
        # Discretize to bins
        normalized = (state - self.min_val) / (self.max_val - self.min_val)
        bin_indices = (normalized * (self.num_bins - 1)).astype(np.int64)
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
        
        # Convert to token IDs (each dimension has its own token space)
        tokens = np.zeros_like(bin_indices)
        for d in range(self.state_dim):
            tokens[:, d] = self.state_token_offset + d * self.num_bins + bin_indices[:, d]
        
        if not is_batch:
            tokens = tokens[0]
        
        return tokens
    
    def decode(
        self,
        tokens: Union[np.ndarray, List[int]],
        denormalize: bool = True,
    ) -> np.ndarray:
        """
        Decode token IDs to state vector.
        
        Args:
            tokens: Token IDs [state_dim] or [B, state_dim]
            denormalize: Whether to denormalize after decoding
            
        Returns:
            State vector [state_dim] or [B, state_dim]
        """
        tokens = np.asarray(tokens)
        is_batch = tokens.ndim == 2
        
        if not is_batch:
            tokens = tokens[np.newaxis, :]
        
        # Convert from token IDs to bin indices
        bin_indices = np.zeros_like(tokens, dtype=np.float32)
        for d in range(min(self.state_dim, tokens.shape[1])):
            dim_tokens = tokens[:, d] - self.state_token_offset - d * self.num_bins
            bin_indices[:, d] = np.clip(dim_tokens, 0, self.num_bins - 1)
        
        # Convert from bins to continuous values
        normalized = bin_indices / (self.num_bins - 1)
        state = normalized * (self.max_val - self.min_val) + self.min_val
        
        # Denormalize
        if denormalize and self.mean is not None:
            state = state * self.std + self.mean
        
        if not is_batch:
            state = state[0]
        
        return state
    
    def batch_encode(
        self,
        states: np.ndarray,
        add_special_tokens: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Batch encode states with attention mask.
        
        Args:
            states: States [B, state_dim]
            add_special_tokens: Add BOS/EOS tokens
            
        Returns:
            Dict with 'input_ids' and 'attention_mask'
        """
        tokens = self.encode(states)
        
        if add_special_tokens and self.special_tokens:
            # Add BOS at start, EOS at end
            batch_size = tokens.shape[0]
            bos = np.full((batch_size, 1), self.bos_token_id)
            eos = np.full((batch_size, 1), self.eos_token_id)
            tokens = np.concatenate([bos, tokens, eos], axis=1)
        
        attention_mask = np.ones_like(tokens)
        
        return {
            'input_ids': tokens,
            'attention_mask': attention_mask,
        }
    
    def __call__(
        self,
        state: Union[np.ndarray, List[float]],
        **kwargs
    ) -> np.ndarray:
        """Encode state (alias for encode)."""
        return self.encode(state, **kwargs)


__all__ = ['StateTokenizer']
