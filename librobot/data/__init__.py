"""Data package for LibroBot VLA framework.

This package provides abstract base classes and utilities for working with
robot learning datasets and text tokenizers.

Modules:
--------
- base: Abstract base classes (AbstractDataset, AbstractTokenizer)
- registry: Registration system for datasets and tokenizers

Usage:
------
>>> from librobot.data import AbstractDataset, register_dataset
>>> 
>>> @register_dataset(name="my_dataset")
>>> class MyDataset(AbstractDataset):
...     def __len__(self):
...         return 1000
...     
...     def __getitem__(self, idx):
...         return {'images': ..., 'text': ..., 'actions': ...}
...     
...     def get_statistics(self):
...         return {'action_mean': ..., 'action_std': ...}
"""

from .base import AbstractDataset, AbstractTokenizer
from .registry import (
    DATASET_REGISTRY,
    TOKENIZER_REGISTRY,
    register_dataset,
    get_dataset,
    create_dataset,
    list_datasets,
    register_tokenizer,
    get_tokenizer,
    create_tokenizer,
    list_tokenizers,
)

__all__ = [
    # Base classes
    'AbstractDataset',
    'AbstractTokenizer',
    # Dataset registry
    'DATASET_REGISTRY',
    'register_dataset',
    'get_dataset',
    'create_dataset',
    'list_datasets',
    # Tokenizer registry
    'TOKENIZER_REGISTRY',
    'register_tokenizer',
    'get_tokenizer',
    'create_tokenizer',
    'list_tokenizers',
]
