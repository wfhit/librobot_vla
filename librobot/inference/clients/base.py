"""Inference clients for connecting to servers."""

from typing import Any, Dict, Optional, List
import asyncio


class BaseClient:
    """Base client for inference servers."""

    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self._connected = False

    def connect(self) -> bool:
        """Connect to server."""
        self._connected = True
        return True

    def disconnect(self) -> None:
        """Disconnect from server."""
        self._connected = False

    def predict(
        self,
        images: Any,
        proprioception: Optional[Any] = None,
        instruction: str = "",
    ) -> Dict[str, Any]:
        """Get prediction from server."""
        raise NotImplementedError

    @property
    def is_connected(self) -> bool:
        return self._connected


class RESTClient(BaseClient):
    """REST API client."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        timeout: float = 10.0,
    ):
        super().__init__(host, port)
        self.timeout = timeout
        self.base_url = f"http://{host}:{port}"

    def predict(
        self,
        images: Any,
        proprioception: Optional[Any] = None,
        instruction: str = "",
    ) -> Dict[str, Any]:
        """Send prediction request."""
        try:
            import requests
            import numpy as np

            payload = {
                "images": np.asarray(images).tolist(),
                "instruction": instruction,
            }
            if proprioception is not None:
                payload["proprioception"] = np.asarray(proprioception).tolist()

            response = requests.post(
                f"{self.base_url}/predict",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()
            result["actions"] = np.array(result["actions"])
            return result

        except ImportError:
            return {"actions": [], "error": "requests required"}
        except Exception as e:
            return {"actions": [], "error": str(e)}

    def health_check(self) -> bool:
        """Check server health."""
        try:
            import requests
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False


class WebSocketClient(BaseClient):
    """WebSocket client for real-time inference."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
    ):
        super().__init__(host, port)
        self.uri = f"ws://{host}:{port}"
        self._websocket = None

    async def connect_async(self) -> bool:
        """Connect to WebSocket server."""
        try:
            import websockets
            self._websocket = await websockets.connect(self.uri)
            self._connected = True
            return True
        except ImportError:
            return False

    async def predict_async(
        self,
        images: Any,
        proprioception: Optional[Any] = None,
        instruction: str = "",
    ) -> Dict[str, Any]:
        """Send prediction request asynchronously."""
        import json
        import numpy as np

        if not self._websocket:
            return {"actions": [], "error": "Not connected"}

        payload = {
            "images": np.asarray(images).tolist(),
            "instruction": instruction,
        }
        if proprioception is not None:
            payload["proprioception"] = np.asarray(proprioception).tolist()

        await self._websocket.send(json.dumps(payload))
        response = await self._websocket.recv()

        result = json.loads(response)
        result["actions"] = np.array(result.get("actions", []))
        return result

    def predict(
        self,
        images: Any,
        proprioception: Optional[Any] = None,
        instruction: str = "",
    ) -> Dict[str, Any]:
        """Synchronous prediction wrapper."""
        return asyncio.get_event_loop().run_until_complete(
            self.predict_async(images, proprioception, instruction)
        )

    async def disconnect_async(self) -> None:
        """Disconnect from server."""
        if self._websocket:
            await self._websocket.close()
        self._connected = False


class GRPCClient(BaseClient):
    """gRPC client for low-latency inference."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50051,
    ):
        super().__init__(host, port)
        self._channel = None
        self._stub = None

    def connect(self) -> bool:
        """Connect to gRPC server."""
        try:
            import grpc
            self._channel = grpc.insecure_channel(f"{self.host}:{self.port}")
            self._connected = True
            return True
        except ImportError:
            return False

    def predict(
        self,
        images: Any,
        proprioception: Optional[Any] = None,
        instruction: str = "",
    ) -> Dict[str, Any]:
        """Send prediction request."""
        import numpy as np

        # In real implementation, would use generated stub
        return {"actions": np.zeros(7)}

    def disconnect(self) -> None:
        """Disconnect from server."""
        if self._channel:
            self._channel.close()
        self._connected = False


__all__ = [
    'BaseClient',
    'RESTClient',
    'WebSocketClient',
    'GRPCClient',
]
