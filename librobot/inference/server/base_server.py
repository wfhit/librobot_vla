"""Abstract base class for inference servers."""

from abc import ABC, abstractmethod
from typing import Any


class AbstractServer(ABC):
    """
    Abstract base class for inference servers.

    Provides a unified interface for serving VLA models.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        """
        Initialize server.

        Args:
            host: Server host address
            port: Server port
        """
        self.host = host
        self.port = port
        self._is_running = False

    @abstractmethod
    async def start(self) -> None:
        """Start the server."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the server."""
        pass

    @abstractmethod
    async def predict(self, request: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Handle prediction request.

        Args:
            request: Prediction request data
            **kwargs: Additional arguments

        Returns:
            Dictionary containing prediction results
        """
        pass

    @abstractmethod
    def load_model(self, model_path: str, **kwargs) -> None:
        """
        Load model for inference.

        Args:
            model_path: Path to model weights
            **kwargs: Additional loading arguments
        """
        pass

    @abstractmethod
    def get_server_info(self) -> dict[str, Any]:
        """
        Get server information.

        Returns:
            Dictionary containing server status and configuration
        """
        pass

    @property
    def is_running(self) -> bool:
        """Check if server is running."""
        return self._is_running

    def get_address(self) -> str:
        """
        Get server address.

        Returns:
            str: Server address (host:port)
        """
        return f"{self.host}:{self.port}"

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
