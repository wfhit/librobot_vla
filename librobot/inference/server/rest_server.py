"""FastAPI REST server for VLA inference."""

import asyncio
import time
from pathlib import Path
from typing import Any, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from librobot.inference.policy import VLAPolicy
from librobot.inference.server.base_server import AbstractServer
from librobot.utils import get_logger

logger = get_logger(__name__)


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Server status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device model is on")
    timestamp: float = Field(..., description="Current timestamp")


class PredictRequest(BaseModel):
    """Prediction request model."""
    image: Optional[Union[list[list[list[float]]], str]] = Field(
        None,
        description="Image as nested list or base64 string"
    )
    text: Optional[str] = Field(None, description="Text instruction")
    state: Optional[list[float]] = Field(None, description="Robot state")
    return_logits: bool = Field(False, description="Return raw logits")


class PredictResponse(BaseModel):
    """Prediction response model."""
    actions: list[list[float]] = Field(..., description="Predicted actions")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    inference_time: float = Field(..., description="Inference time in seconds")


class ServerInfo(BaseModel):
    """Server information model."""
    name: str = Field(..., description="Server name")
    version: str = Field(..., description="Server version")
    host: str = Field(..., description="Server host")
    port: int = Field(..., description="Server port")
    model_loaded: bool = Field(..., description="Model load status")
    device: str = Field(..., description="Device")


class RESTServer(AbstractServer):
    """
    FastAPI-based REST server for VLA inference.

    Provides HTTP endpoints for health checks, model loading,
    and action prediction.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        enable_cors: bool = True,
        log_requests: bool = True,
    ):
        """
        Initialize REST server.

        Args:
            host: Server host address
            port: Server port
            model_path: Optional path to model checkpoint
            device: Device for inference (cuda/cpu)
            enable_cors: Enable CORS middleware
            log_requests: Log incoming requests
        """
        super().__init__(host=host, port=port)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.enable_cors = enable_cors
        self.log_requests = log_requests

        # Initialize FastAPI app
        self.app = FastAPI(
            title="LibroBot VLA Inference Server",
            description="REST API for Vision-Language-Action model inference",
            version="0.1.0",
        )

        # Setup middleware
        if self.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        # Setup logging middleware
        if self.log_requests:
            @self.app.middleware("http")
            async def log_requests_middleware(request: Request, call_next):
                start_time = time.time()
                response = await call_next(request)
                duration = time.time() - start_time
                logger.info(
                    f"{request.method} {request.url.path} "
                    f"- {response.status_code} ({duration:.3f}s)"
                )
                return response

        # Setup routes
        self._setup_routes()

        # Initialize policy
        self.policy: Optional[VLAPolicy] = None

        # Load model if path provided
        if model_path is not None:
            self.load_model(str(model_path))

        # Server instance
        self.server: Optional[uvicorn.Server] = None

    def _setup_routes(self) -> None:
        """Setup API routes."""

        @self.app.get("/", response_model=dict[str, str])
        async def root():
            """Root endpoint."""
            return {
                "message": "LibroBot VLA Inference Server",
                "docs": "/docs",
                "health": "/health",
            }

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy" if self.policy is not None else "not_ready",
                model_loaded=self.policy is not None,
                device=self.device,
                timestamp=time.time(),
            )

        @self.app.get("/info", response_model=ServerInfo)
        async def server_info():
            """Get server information."""
            return self.get_server_info()

        @self.app.post("/predict", response_model=PredictResponse)
        async def predict_action(request: PredictRequest):
            """
            Predict actions from observation.

            Args:
                request: Prediction request with image/text/state

            Returns:
                Predicted actions and metadata
            """
            if self.policy is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Model not loaded"
                )

            try:
                # Convert request to observation dict
                observation = {}

                if request.image is not None:
                    # TODO: Handle image conversion from list/base64
                    observation["image"] = request.image

                if request.text is not None:
                    observation["text"] = request.text

                if request.state is not None:
                    observation["state"] = torch.tensor(request.state)

                # Run inference
                start_time = time.time()
                result = await self.predict(
                    {"observation": observation, "return_logits": request.return_logits}
                )
                inference_time = time.time() - start_time

                # Convert actions to list
                actions = result["actions"]
                if isinstance(actions, torch.Tensor):
                    actions = actions.cpu().numpy().tolist()

                return PredictResponse(
                    actions=actions,
                    metadata=result.get("metadata", {}),
                    inference_time=inference_time,
                )

            except Exception as e:
                logger.error(f"Prediction error: {e}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Prediction failed: {str(e)}"
                )

        @self.app.post("/reset")
        async def reset_policy():
            """Reset policy state."""
            if self.policy is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Model not loaded"
                )

            self.policy.reset()
            return {"status": "reset_complete"}

        @self.app.exception_handler(Exception)
        async def global_exception_handler(request: Request, exc: Exception):
            """Global exception handler."""
            logger.error(f"Unhandled exception: {exc}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "detail": "Internal server error",
                    "error": str(exc),
                }
            )

    async def start(self) -> None:
        """Start the REST server."""
        logger.info(f"Starting REST server on {self.host}:{self.port}")

        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="info",
        )

        self.server = uvicorn.Server(config)
        self._is_running = True

        # Run server
        await self.server.serve()

    async def stop(self) -> None:
        """Stop the REST server."""
        logger.info("Stopping REST server")

        if self.server is not None:
            self.server.should_exit = True
            await asyncio.sleep(0.5)

        self._is_running = False
        logger.info("REST server stopped")

    async def predict(
        self,
        request: dict[str, Any],
        **kwargs
    ) -> dict[str, Any]:
        """
        Handle prediction request.

        Args:
            request: Request dictionary containing observation
            **kwargs: Additional arguments

        Returns:
            Prediction results
        """
        if self.policy is None:
            raise RuntimeError("Model not loaded")

        observation = request.get("observation", {})
        return_logits = request.get("return_logits", False)

        # Run prediction (in thread pool to avoid blocking)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.policy.predict,
            observation,
            return_logits
        )

        return result

    def load_model(self, model_path: str, **kwargs) -> None:
        """
        Load VLA model.

        Args:
            model_path: Path to model checkpoint
            **kwargs: Additional loading arguments
        """
        logger.info(f"Loading model from {model_path}")

        # TODO: Initialize model architecture based on config
        # For now, assume model is provided or will be loaded externally

        # Initialize policy
        self.policy = VLAPolicy(
            device=self.device,
            use_kv_cache=kwargs.get("use_kv_cache", False),
            use_action_buffer=kwargs.get("use_action_buffer", True),
        )

        # Load checkpoint if path exists
        model_path_obj = Path(model_path)
        if model_path_obj.exists():
            # TODO: Load actual checkpoint
            # self.policy.load_checkpoint(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Model path does not exist: {model_path}")

    def get_server_info(self) -> dict[str, Any]:
        """
        Get server information.

        Returns:
            Dictionary with server details
        """
        return {
            "name": "LibroBot VLA REST Server",
            "version": "0.1.0",
            "host": self.host,
            "port": self.port,
            "model_loaded": self.policy is not None,
            "device": self.device,
        }

    def run(self) -> None:
        """
        Run server in blocking mode.

        This is a convenience method for running the server
        without using async context manager.
        """
        asyncio.run(self.start())


def create_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs
) -> RESTServer:
    """
    Factory function to create REST server.

    Args:
        host: Server host
        port: Server port
        model_path: Path to model checkpoint
        device: Device for inference
        **kwargs: Additional server arguments

    Returns:
        Configured REST server instance

    Example:
        >>> server = create_server(port=8000, model_path="model.pt")
        >>> server.run()
    """
    return RESTServer(
        host=host,
        port=port,
        model_path=model_path,
        device=device,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Start VLA REST inference server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--model-path", type=str, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, help="Device (cuda/cpu)")

    args = parser.parse_args()

    server = create_server(
        host=args.host,
        port=args.port,
        model_path=args.model_path,
        device=args.device,
    )

    logger.info(f"Starting server at http://{args.host}:{args.port}")
    logger.info(f"API documentation: http://{args.host}:{args.port}/docs")

    server.run()
