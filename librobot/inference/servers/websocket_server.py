"""WebSocket server for real-time VLA inference."""

import asyncio
import json
from typing import Any, Optional

from ..server.base_server import AbstractServer


class WebSocketServer(AbstractServer):
    """WebSocket server for real-time streaming inference."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        model: Optional[Any] = None,
    ):
        """
        Args:
            host: Server host
            port: Server port
            model: VLA model or policy
        """
        super().__init__(host=host, port=port)
        self.model = model
        self._clients: set = set()
        self._server = None

    async def start(self) -> None:
        """Start WebSocket server."""
        try:
            import websockets

            self._server = await websockets.serve(
                self._handle_client,
                self.host,
                self.port,
            )
            self._is_running = True

            await self._server.wait_closed()

        except ImportError:
            raise ImportError("websockets required for WebSocket server")

    async def _handle_client(self, websocket, path) -> None:
        """Handle WebSocket client connection."""
        self._clients.add(websocket)
        try:
            async for message in websocket:
                request = json.loads(message)
                response = await self.predict(request)
                await websocket.send(json.dumps(response))
        except Exception as e:
            await websocket.send(json.dumps({"error": str(e)}))
        finally:
            self._clients.discard(websocket)

    async def stop(self) -> None:
        """Stop WebSocket server."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        self._is_running = False

    async def predict(
        self,
        request: dict[str, Any],
        **kwargs
    ) -> dict[str, Any]:
        """Handle prediction request."""
        import numpy as np

        if self.model is None:
            return {"actions": [], "error": "No model loaded"}

        observation = {
            "images": np.array(request.get("images", [])),
            "proprioception": np.array(request.get("proprioception", [])),
        }
        instruction = request.get("instruction", "")

        if hasattr(self.model, 'get_action'):
            action = self.model.get_action(observation, instruction)
        else:
            action = np.zeros(7)

        return {"actions": action.tolist(), "timestamp": request.get("timestamp")}

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Broadcast message to all connected clients."""
        if self._clients:
            await asyncio.gather(
                *[client.send(json.dumps(message)) for client in self._clients]
            )

    def load_model(self, model_path: str, **kwargs) -> None:
        """Load model."""
        try:
            import torch
            self.model = torch.load(model_path)
        except ImportError:
            pass

    def get_server_info(self) -> dict[str, Any]:
        return {
            "type": "WebSocket",
            "host": self.host,
            "port": self.port,
            "is_running": self._is_running,
            "connected_clients": len(self._clients),
        }


__all__ = ['WebSocketServer']
