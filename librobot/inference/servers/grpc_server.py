"""gRPC server for VLA inference."""

from typing import Any, Dict, Optional
import asyncio

from ..server.base_server import AbstractServer


class GRPCServer(AbstractServer):
    """gRPC server for low-latency VLA inference."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 50051,
        model: Optional[Any] = None,
        max_workers: int = 10,
    ):
        """
        Args:
            host: Server host
            port: Server port
            model: VLA model or policy
            max_workers: Maximum worker threads
        """
        super().__init__(host=host, port=port)
        self.model = model
        self.max_workers = max_workers
        self._server = None

    async def start(self) -> None:
        """Start gRPC server."""
        try:
            import grpc
            from concurrent import futures

            self._server = grpc.aio.server(
                futures.ThreadPoolExecutor(max_workers=self.max_workers)
            )

            # Add service
            self._add_servicer()

            self._server.add_insecure_port(f'{self.host}:{self.port}')
            await self._server.start()
            self._is_running = True

            await self._server.wait_for_termination()

        except ImportError:
            raise ImportError("grpcio required for gRPC server")

    def _add_servicer(self) -> None:
        """Add gRPC service implementation."""
        # In a real implementation, this would add a generated servicer
        pass

    async def stop(self) -> None:
        """Stop gRPC server."""
        if self._server:
            await self._server.stop(grace=5)
        self._is_running = False

    async def predict(
        self,
        request: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Handle prediction request."""
        import numpy as np

        if self.model is None:
            return {"actions": []}

        observation = {
            "images": np.array(request.get("images", [])),
            "proprioception": np.array(request.get("proprioception", [])),
        }
        instruction = request.get("instruction", "")

        if hasattr(self.model, 'get_action'):
            action = self.model.get_action(observation, instruction)
        else:
            action = np.zeros(7)

        return {"actions": action.tolist()}

    def load_model(self, model_path: str, **kwargs) -> None:
        """Load model."""
        try:
            import torch
            self.model = torch.load(model_path)
        except ImportError:
            pass

    def get_server_info(self) -> Dict[str, Any]:
        return {
            "type": "gRPC",
            "host": self.host,
            "port": self.port,
            "is_running": self._is_running,
        }


__all__ = ['GRPCServer']
