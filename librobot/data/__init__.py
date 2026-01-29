"""Data module for LibroBot VLA.

This module provides datasets, tokenizers, and transforms for training
and evaluating VLA models. It supports multiple dataset formats including
LeRobot, RLDS, and HDF5.

See docs/design/data_pipeline.md for detailed design documentation.
"""

from .datasets import (
    AbstractDataset,
    register_dataset,
    get_dataset,
    create_dataset,
    list_datasets,
)

__all__ = [
    # Datasets
    'AbstractDataset',
    'register_dataset',
    'get_dataset',
    'create_dataset',
    'list_datasets',
]
