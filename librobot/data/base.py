"""Abstract base classes for datasets and tokenizers."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Optional, Union

import numpy as np


class AbstractDataset(ABC):
    """
    Abstract base class for robot learning datasets.

    Provides a unified interface for loading and processing robot
    demonstration data for VLA model training.
    """

    def __init__(
        self,
        data_path: str,
        split: str = "train",
        transform: Optional[Any] = None,
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to dataset root directory
            split: Dataset split ("train", "val", "test")
            transform: Optional transform to apply to samples
        """
        self.data_path = data_path
        self.split = split
        self.transform = transform

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - 'images': Image tensor(s) [C, H, W] or [N_cameras, C, H, W]
                - 'text': Text instruction string
                - 'actions': Action tensor [action_dim] or [T, action_dim]
                - 'proprioception': Proprioceptive state [state_dim]
                - Optional additional fields
        """
        pass

    @abstractmethod
    def get_statistics(self) -> dict[str, Any]:
        """
        Get dataset statistics for normalization.

        Returns:
            Dictionary containing:
                - 'action_mean': Mean of actions
                - 'action_std': Std of actions
                - 'state_mean': Mean of proprioceptive states
                - 'state_std': Std of proprioceptive states
        """
        pass

    def get_action_dim(self) -> int:
        """
        Get action dimension.

        Returns:
            Action dimension
        """
        sample = self[0]
        actions = sample.get('actions')
        if actions is None:
            raise ValueError("Dataset samples must contain 'actions' field")
        if hasattr(actions, 'shape'):
            return actions.shape[-1]
        return len(actions)

    def get_state_dim(self) -> int:
        """
        Get proprioceptive state dimension.

        Returns:
            State dimension
        """
        sample = self[0]
        state = sample.get('proprioception')
        if state is None:
            return 0
        if hasattr(state, 'shape'):
            return state.shape[-1]
        return len(state)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over the dataset."""
        for i in range(len(self)):
            yield self[i]


class AbstractTokenizer(ABC):
    """
    Abstract base class for text tokenizers.

    Provides a unified interface for tokenizing text instructions
    for VLA models.
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        max_length: int = 77,
        padding: str = "max_length",
        truncation: bool = True,
    ):
        """
        Initialize tokenizer.

        Args:
            vocab_size: Vocabulary size
            max_length: Maximum sequence length
            padding: Padding strategy ("max_length", "longest", "do_not_pad")
            truncation: Whether to truncate sequences
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

    @abstractmethod
    def encode(
        self,
        text: Union[str, list[str]],
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> dict[str, Any]:
        """
        Encode text to token IDs.

        Args:
            text: Input text or list of texts
            return_tensors: Return tensor type ("pt", "np", None)
            **kwargs: Additional tokenization arguments

        Returns:
            Dictionary containing:
                - 'input_ids': Token IDs
                - 'attention_mask': Attention mask
        """
        pass

    @abstractmethod
    def decode(
        self,
        token_ids: Union[list[int], np.ndarray, Any],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional decoding arguments

        Returns:
            Decoded text string
        """
        pass

    @abstractmethod
    def batch_encode(
        self,
        texts: list[str],
        return_tensors: Optional[str] = None,
        **kwargs
    ) -> dict[str, Any]:
        """
        Batch encode multiple texts.

        Args:
            texts: List of input texts
            return_tensors: Return tensor type
            **kwargs: Additional tokenization arguments

        Returns:
            Dictionary containing batched token IDs and attention masks
        """
        pass

    @abstractmethod
    def batch_decode(
        self,
        token_ids_batch: Union[list[list[int]], np.ndarray, Any],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> list[str]:
        """
        Batch decode multiple token ID sequences.

        Args:
            token_ids_batch: Batch of token IDs
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional decoding arguments

        Returns:
            List of decoded text strings
        """
        pass

    def __call__(
        self,
        text: Union[str, list[str]],
        **kwargs
    ) -> dict[str, Any]:
        """
        Tokenize text (alias for encode).

        Args:
            text: Input text or list of texts
            **kwargs: Additional arguments

        Returns:
            Tokenization output
        """
        if isinstance(text, list):
            return self.batch_encode(text, **kwargs)
        return self.encode(text, **kwargs)

    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        pass

    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        """Get end-of-sequence token ID."""
        pass

    @property
    @abstractmethod
    def bos_token_id(self) -> int:
        """Get beginning-of-sequence token ID."""
        pass


__all__ = [
    'AbstractDataset',
    'AbstractTokenizer',
]
