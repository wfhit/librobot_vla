"""Data package for LibroBot VLA framework.

This package provides abstract base classes and utilities for working with
robot learning datasets and text tokenizers.

Modules:
--------
- base: Abstract base classes (AbstractDataset, AbstractTokenizer)
- registry: Registration system for datasets and tokenizers
- datasets: Dataset loaders (LeRobot, RLDS, HDF5, Zarr, WebDataset)
- tokenizers: Tokenizers for state, action, image data
- transforms: Data transforms and augmentations

Usage:
------
>>> from librobot.data import AbstractDataset, register_dataset
>>> from librobot.data.datasets import LeRobotDataset, HDF5Dataset
>>> from librobot.data.tokenizers import ActionTokenizer, StateTokenizer
>>> from librobot.data.transforms import Compose, ActionNormalize
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

# Import submodules
from . import datasets
from . import tokenizers
from . import transforms

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
    # Submodules
    'datasets',
    'tokenizers',
    'transforms',
]
