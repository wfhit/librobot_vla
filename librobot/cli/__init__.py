"""CLI commands for LibroBot VLA."""

from .collect import collect_cli
from .evaluate import evaluate_cli
from .serve import serve_cli
from .train import train_cli

__all__ = [
    'train_cli',
    'evaluate_cli',
    'serve_cli',
    'collect_cli',
]
