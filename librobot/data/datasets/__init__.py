"""Dataset implementations for LibroBot VLA.

This package provides dataset implementations for various robotics data formats:
- LeRobot: HuggingFace-based robotics datasets
- RLDS: TensorFlow Datasets format for robotics
- HDF5: Custom HDF5 format for robotics data

See docs/design/data_pipeline.md for detailed design documentation.
"""

from .base import AbstractDataset
from .registry import (
    DATASET_REGISTRY,
    register_dataset,
    get_dataset,
    create_dataset,
    list_datasets,
)

# Import dataset implementations to register them
from . import lerobot_dataset
from . import rlds_dataset
from . import hdf5_dataset

__all__ = [
    'AbstractDataset',
    'DATASET_REGISTRY',
    'register_dataset',
    'get_dataset',
    'create_dataset',
    'list_datasets',
    # Dataset implementations
    'lerobot_dataset',
    'rlds_dataset',
    'hdf5_dataset',
]
