#!/bin/bash
# Run inference server in Docker container

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DOCKER_DIR="$( dirname "$SCRIPT_DIR" )"

cd "$DOCKER_DIR"

echo "Starting inference server..."

docker-compose up inference
