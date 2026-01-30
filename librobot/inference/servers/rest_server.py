"""REST API server for VLA inference."""

from typing import Any, Optional

from ..server.base_server import AbstractServer


class RESTServer(AbstractServer):
    """FastAPI-based REST server for VLA inference."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        model: Optional[Any] = None,
        cors_origins: Optional[list] = None,
    ):
        """
        Args:
            host: Server host
            port: Server port
            model: VLA model or policy
            cors_origins: CORS allowed origins
        """
        super().__init__(host=host, port=port)
        self.model = model
        self.cors_origins = cors_origins or ["*"]
        self._app = None
        self._server = None

    def _setup_app(self):
        """Setup FastAPI application."""
        try:
            import numpy as np
            from fastapi import FastAPI, HTTPException
            from fastapi.middleware.cors import CORSMiddleware
            from pydantic import BaseModel

            app = FastAPI(
                title="VLA Inference Server",
                description="REST API for Vision-Language-Action model inference",
                version="1.0.0",
            )

            app.add_middleware(
                CORSMiddleware,
                allow_origins=self.cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            class PredictRequest(BaseModel):
                images: list  # Base64 encoded or nested list
                proprioception: Optional[list] = None
                instruction: str = ""

            class PredictResponse(BaseModel):
                actions: list
                success: bool = True
                message: str = ""

            @app.get("/health")
            async def health():
                return {"status": "healthy", "model_loaded": self.model is not None}

            @app.get("/info")
            async def info():
                return self.get_server_info()

            @app.post("/predict", response_model=PredictResponse)
            async def predict(request: PredictRequest):
                try:
                    result = await self.predict(request.dict())
                    return PredictResponse(
                        actions=result.get("actions", []),
                        success=True,
                    )
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

            @app.post("/batch_predict")
            async def batch_predict(requests: list):
                results = []
                for req in requests:
                    result = await self.predict(req)
                    results.append(result)
                return {"results": results}

            self._app = app
            return app

        except ImportError:
            return None

    async def start(self) -> None:
        """Start the REST server."""
        if self._app is None:
            self._setup_app()

        if self._app is None:
            raise ImportError("FastAPI required for REST server")

        try:
            import uvicorn

            config = uvicorn.Config(
                self._app,
                host=self.host,
                port=self.port,
                log_level="info",
            )
            self._server = uvicorn.Server(config)
            self._is_running = True
            await self._server.serve()

        except ImportError:
            raise ImportError("uvicorn required for REST server")

    async def stop(self) -> None:
        """Stop the REST server."""
        if self._server:
            self._server.should_exit = True
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

        # Prepare observation
        observation = {}

        if "images" in request:
            images = np.array(request["images"], dtype=np.float32)
            observation["images"] = images

        if "proprioception" in request and request["proprioception"]:
            state = np.array(request["proprioception"], dtype=np.float32)
            observation["proprioception"] = state

        instruction = request.get("instruction", "")

        # Get action
        if hasattr(self.model, 'get_action'):
            action = self.model.get_action(observation, instruction)
        elif callable(self.model):
            action = self.model(observation, instruction)
        else:
            action = np.zeros(7)

        return {"actions": action.tolist()}

    def load_model(self, model_path: str, **kwargs) -> None:
        """Load model for inference."""
        try:
            import torch
            self.model = torch.load(model_path, map_location='cpu')
            if hasattr(self.model, 'eval'):
                self.model.eval()
        except ImportError:
            pass

    def get_server_info(self) -> dict[str, Any]:
        """Get server information."""
        return {
            "type": "REST",
            "host": self.host,
            "port": self.port,
            "is_running": self._is_running,
            "model_loaded": self.model is not None,
            "endpoints": ["/health", "/info", "/predict", "/batch_predict"],
        }


__all__ = ['RESTServer']
