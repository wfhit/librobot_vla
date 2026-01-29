"""Serve CLI command for inference server."""

import argparse
from pathlib import Path
from typing import Optional
import sys
import asyncio


def serve_cli(args: Optional[list] = None) -> int:
    """
    Start a VLA inference server.
    
    Usage:
        librobot-serve --checkpoint model.pt --port 8000
        librobot-serve --checkpoint model.pt --type grpc --port 50051
    """
    parser = argparse.ArgumentParser(
        description="Start a VLA inference server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to model configuration",
    )
    
    # Server
    parser.add_argument(
        "--type", "-t",
        type=str,
        default="rest",
        choices=["rest", "grpc", "websocket", "ros2"],
        help="Server type",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Server port",
    )
    
    # Performance
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference device",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Maximum batch size",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of workers",
    )
    
    # Optimization
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Use quantized model",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        help="Path to ONNX model (instead of checkpoint)",
    )
    parser.add_argument(
        "--tensorrt",
        type=str,
        help="Path to TensorRT engine",
    )
    
    # Misc
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    
    parsed_args = parser.parse_args(args)
    return run_server(parsed_args)


def run_server(args) -> int:
    """Start the inference server."""
    try:
        print(f"Starting {args.type.upper()} server...")
        print(f"  Checkpoint: {args.checkpoint}")
        print(f"  Host: {args.host}")
        print(f"  Port: {args.port}")
        print(f"  Device: {args.device}")
        
        # Load model
        model = load_model(args)
        
        # Create server
        if args.type == "rest":
            from librobot.inference.servers import RESTServer
            server = RESTServer(
                host=args.host,
                port=args.port,
                model=model,
            )
        elif args.type == "grpc":
            from librobot.inference.servers import GRPCServer
            server = GRPCServer(
                host=args.host,
                port=args.port,
                model=model,
            )
        elif args.type == "websocket":
            from librobot.inference.servers import WebSocketServer
            server = WebSocketServer(
                host=args.host,
                port=args.port,
                model=model,
            )
        elif args.type == "ros2":
            from librobot.inference.servers import ROS2Server
            server = ROS2Server(model=model)
        
        # Run server
        print(f"\nServer running at {args.host}:{args.port}")
        print("Press Ctrl+C to stop")
        
        asyncio.run(server.start())
        
        return 0
        
    except KeyboardInterrupt:
        print("\nShutting down server...")
        return 0
    except Exception as e:
        print(f"Server error: {e}")
        return 1


def load_model(args):
    """Load model for inference."""
    from librobot.inference.policy import BasePolicy
    
    # Try loading optimized model first
    if args.onnx:
        from librobot.inference.optimization import OptimizedModel
        return OptimizedModel(args.onnx, backend="onnx")
    
    if args.tensorrt:
        from librobot.inference.optimization import OptimizedModel
        return OptimizedModel(args.tensorrt, backend="tensorrt")
    
    # Load PyTorch model
    try:
        import torch
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Need to reconstruct model
            return checkpoint  # Simplified
        
        return checkpoint
        
    except ImportError:
        print("PyTorch not available, returning dummy model")
        return None


def main():
    sys.exit(serve_cli())


if __name__ == "__main__":
    main()
