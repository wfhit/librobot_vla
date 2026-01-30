"""File I/O utilities for various data formats."""

from __future__ import annotations

import json
import pickle
import yaml
from pathlib import Path
from typing import Any, Dict, Union, Optional
import torch


def save_json(data: Any, path: Union[str, Path], indent: int = 2, **kwargs) -> None:
    """
    Save data to JSON file.

    Args:
        data: Data to save
        path: Output file path
        indent: JSON indentation level
        **kwargs: Additional arguments for json.dump
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, **kwargs)


def load_json(path: Union[str, Path], **kwargs) -> Any:
    """
    Load data from JSON file.

    Args:
        path: Input file path
        **kwargs: Additional arguments for json.load

    Returns:
        Loaded data
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f, **kwargs)


def save_yaml(data: Any, path: Union[str, Path], **kwargs) -> None:
    """
    Save data to YAML file.

    Args:
        data: Data to save
        path: Output file path
        **kwargs: Additional arguments for yaml.dump
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, **kwargs)


def load_yaml(path: Union[str, Path], **kwargs) -> Any:
    """
    Load data from YAML file.

    Args:
        path: Input file path
        **kwargs: Additional arguments for yaml.safe_load

    Returns:
        Loaded data
    """
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f, **kwargs)


def save_pickle(data: Any, path: Union[str, Path], **kwargs) -> None:
    """
    Save data to pickle file.

    Args:
        data: Data to save
        path: Output file path
        **kwargs: Additional arguments for pickle.dump
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(data, f, **kwargs)


def load_pickle(path: Union[str, Path], **kwargs) -> Any:
    """
    Load data from pickle file.

    Args:
        path: Input file path
        **kwargs: Additional arguments for pickle.load

    Returns:
        Loaded data
    """
    with open(path, 'rb') as f:
        return pickle.load(f, **kwargs)


def save_torch(data: Any, path: Union[str, Path], **kwargs) -> None:
    """
    Save data using torch.save.

    Args:
        data: Data to save (tensor, model state dict, etc.)
        path: Output file path
        **kwargs: Additional arguments for torch.save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path, **kwargs)


def load_torch(path: Union[str, Path], map_location: Optional[str] = None, **kwargs) -> Any:
    """
    Load data using torch.load.

    Args:
        path: Input file path
        map_location: Device to map tensors to
        **kwargs: Additional arguments for torch.load

    Returns:
        Loaded data

    Note:
        This uses weights_only=False to support loading complex objects.
        Only load checkpoints from trusted sources.
    """
    return torch.load(path, map_location=map_location, weights_only=False, **kwargs)


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path: Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_text(path: Union[str, Path], encoding: str = 'utf-8') -> str:
    """
    Read text from file.

    Args:
        path: Input file path
        encoding: Text encoding

    Returns:
        str: File contents
    """
    with open(path, 'r', encoding=encoding) as f:
        return f.read()


def write_text(text: str, path: Union[str, Path], encoding: str = 'utf-8') -> None:
    """
    Write text to file.

    Args:
        text: Text to write
        path: Output file path
        encoding: Text encoding
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding=encoding) as f:
        f.write(text)


def read_lines(path: Union[str, Path], encoding: str = 'utf-8', strip: bool = True) -> list[str]:
    """
    Read lines from file.

    Args:
        path: Input file path
        encoding: Text encoding
        strip: If True, strips whitespace from each line

    Returns:
        List of lines
    """
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()

    if strip:
        lines = [line.strip() for line in lines]

    return lines


def write_lines(lines: list[str], path: Union[str, Path], encoding: str = 'utf-8') -> None:
    """
    Write lines to file.

    Args:
        lines: Lines to write
        path: Output file path
        encoding: Text encoding
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding=encoding) as f:
        for line in lines:
            f.write(line)
            if not line.endswith('\n'):
                f.write('\n')
