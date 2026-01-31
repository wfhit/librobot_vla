"""Thread-safe data buffer for efficient data storage."""

import threading
from collections import deque
from typing import Any, Optional

import numpy as np


class DataBuffer:
    """
    Thread-safe data buffer for efficient storage of episode data.

    Supports multiple data streams (images, actions, proprioception, etc.)
    with memory-efficient chunked storage.
    """

    def __init__(
        self,
        max_size: Optional[int] = None,
        chunk_size: int = 100,
    ):
        """
        Initialize data buffer.

        Args:
            max_size: Maximum buffer size (None for unlimited)
            chunk_size: Size of data chunks for efficient storage
        """
        self.max_size = max_size
        self.chunk_size = chunk_size
        self._lock = threading.RLock()
        self._buffers: dict[str, deque] = {}
        self._metadata: dict[str, Any] = {}

    def create_stream(self, name: str, dtype: Optional[np.dtype] = None) -> None:
        """
        Create a new data stream.

        Args:
            name: Stream name
            dtype: Optional data type for the stream
        """
        with self._lock:
            if name not in self._buffers:
                self._buffers[name] = deque(maxlen=self.max_size)
                if dtype is not None:
                    self._metadata[name] = {"dtype": dtype}

    def append(self, stream_name: str, data: Any) -> None:
        """
        Append data to a stream.

        Args:
            stream_name: Name of the stream
            data: Data to append
        """
        with self._lock:
            if stream_name not in self._buffers:
                self.create_stream(stream_name)

            self._buffers[stream_name].append(data)

    def get_stream(self, stream_name: str) -> list[Any]:
        """
        Get all data from a stream.

        Args:
            stream_name: Name of the stream

        Returns:
            List of data in the stream
        """
        with self._lock:
            if stream_name not in self._buffers:
                return []
            return list(self._buffers[stream_name])

    def get_last(self, stream_name: str, n: int = 1) -> list[Any]:
        """
        Get last n items from a stream.

        Args:
            stream_name: Name of the stream
            n: Number of items to retrieve

        Returns:
            List of last n items
        """
        with self._lock:
            if stream_name not in self._buffers:
                return []

            buffer = self._buffers[stream_name]
            return list(buffer)[-n:] if len(buffer) >= n else list(buffer)

    def clear_stream(self, stream_name: str) -> None:
        """
        Clear a specific stream.

        Args:
            stream_name: Name of the stream to clear
        """
        with self._lock:
            if stream_name in self._buffers:
                self._buffers[stream_name].clear()

    def clear_all(self) -> None:
        """Clear all streams."""
        with self._lock:
            for buffer in self._buffers.values():
                buffer.clear()

    def get_stream_length(self, stream_name: str) -> int:
        """
        Get length of a stream.

        Args:
            stream_name: Name of the stream

        Returns:
            Length of the stream
        """
        with self._lock:
            if stream_name not in self._buffers:
                return 0
            return len(self._buffers[stream_name])

    def get_all_streams(self) -> dict[str, list[Any]]:
        """
        Get all data from all streams.

        Returns:
            Dictionary mapping stream names to data lists
        """
        with self._lock:
            return {name: list(buffer) for name, buffer in self._buffers.items()}

    def list_streams(self) -> list[str]:
        """
        List all stream names.

        Returns:
            List of stream names
        """
        with self._lock:
            return list(self._buffers.keys())

    def __len__(self) -> int:
        """Get number of streams."""
        with self._lock:
            return len(self._buffers)

    def __contains__(self, stream_name: str) -> bool:
        """Check if stream exists."""
        with self._lock:
            return stream_name in self._buffers


__all__ = ["DataBuffer"]
