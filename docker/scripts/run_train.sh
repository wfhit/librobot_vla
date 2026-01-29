#!/bin/bash
# Run training in Docker container

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DOCKER_DIR="$( dirname "$SCRIPT_DIR" )"
PROJECT_DIR="$( dirname "$DOCKER_DIR" )"

cd "$DOCKER_DIR"

# Default config
CONFIG="${1:-configs/experiment/wheel_loader_groot.yaml}"

echo "Running training with config: $CONFIG"

docker-compose run --rm train python3 scripts/train.py config="$CONFIG"
