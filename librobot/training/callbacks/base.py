"""Abstract base class for training callbacks."""

from abc import ABC
from typing import Any, Optional


class AbstractCallback(ABC):
    """
    Abstract base class for training callbacks.

    Callbacks allow for custom behavior at different stages of training.
    """

    def __init__(self):
        """Initialize callback."""
        self.trainer: Optional[Any] = None

    def set_trainer(self, trainer: Any) -> None:
        """
        Set reference to trainer.

        Args:
            trainer: Trainer instance
        """
        self.trainer = trainer

    def on_train_begin(self, logs: Optional[dict[str, Any]] = None) -> None:
        """
        Called at the beginning of training.

        Args:
            logs: Dictionary of training information
        """
        pass

    def on_train_end(self, logs: Optional[dict[str, Any]] = None) -> None:
        """
        Called at the end of training.

        Args:
            logs: Dictionary of training information
        """
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> None:
        """
        Called at the beginning of an epoch.

        Args:
            epoch: Current epoch number
            logs: Dictionary of training information
        """
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[dict[str, Any]] = None) -> None:
        """
        Called at the end of an epoch.

        Args:
            epoch: Current epoch number
            logs: Dictionary of training information
        """
        pass

    def on_batch_begin(self, batch: int, logs: Optional[dict[str, Any]] = None) -> None:
        """
        Called at the beginning of a training batch.

        Args:
            batch: Current batch number
            logs: Dictionary of training information
        """
        pass

    def on_batch_end(self, batch: int, logs: Optional[dict[str, Any]] = None) -> None:
        """
        Called at the end of a training batch.

        Args:
            batch: Current batch number
            logs: Dictionary of training information
        """
        pass

    def on_validation_begin(self, logs: Optional[dict[str, Any]] = None) -> None:
        """
        Called at the beginning of validation.

        Args:
            logs: Dictionary of validation information
        """
        pass

    def on_validation_end(self, logs: Optional[dict[str, Any]] = None) -> None:
        """
        Called at the end of validation.

        Args:
            logs: Dictionary of validation information
        """
        pass

    def get_config(self) -> dict[str, Any]:
        """
        Get callback configuration.

        Returns:
            Dictionary containing configuration
        """
        return {"type": self.__class__.__name__}
