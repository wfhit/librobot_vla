#!/bin/bash
# Run training in Docker container

set -e

CONFIG=${1:-"configs/experiment/default.yaml"}
GPUS=${2:-"0"}

echo "Starting training with config: $CONFIG"
echo "Using GPUs: $GPUS"

docker run --rm -it \
    --runtime=nvidia \
    --gpus "device=$GPUS" \
    --shm-size 16g \
    --ipc=host \
    -v $(pwd)/../../librobot:/workspace/librobot_vla/librobot \
    -v $(pwd)/../../scripts:/workspace/librobot_vla/scripts \
    -v $(pwd)/../../configs:/workspace/librobot_vla/configs \
    -v $(pwd)/../../data:/workspace/data \
    -v $(pwd)/../../outputs:/workspace/outputs \
    -v $(pwd)/../../checkpoints:/workspace/checkpoints \
    -e CUDA_VISIBLE_DEVICES=$GPUS \
    -e WANDB_API_KEY=${WANDB_API_KEY} \
    -e HF_TOKEN=${HF_TOKEN} \
    librobot-train:latest \
    python scripts/train.py config=$CONFIG
