#!/bin/bash
# Run inference server in Docker container

set -e

PORT=${1:-8000}
GPUS=${2:-"0"}

echo "Starting inference server on port $PORT"
echo "Using GPUs: $GPUS"

docker run --rm -it \
    --runtime=nvidia \
    --gpus "device=$GPUS" \
    -p $PORT:8000 \
    -v $(pwd)/../../checkpoints:/app/checkpoints:ro \
    -v $(pwd)/../../configs:/app/configs:ro \
    -e CUDA_VISIBLE_DEVICES=$GPUS \
    librobot-deploy:latest \
    python scripts/inference.py --server --host 0.0.0.0 --port 8000
