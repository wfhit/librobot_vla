#!/bin/bash
# Build Docker images for LibroBot

set -e

echo "Building LibroBot Docker images..."

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DOCKER_DIR="$( dirname "$SCRIPT_DIR" )"
PROJECT_DIR="$( dirname "$DOCKER_DIR" )"

cd "$PROJECT_DIR"

# Build base image
echo "Building base image..."
docker build -f docker/Dockerfile.base -t librobot:base .

# Build training image
echo "Building training image..."
docker build -f docker/Dockerfile.train -t librobot:train .

# Build deployment image
echo "Building deployment image..."
docker build -f docker/Dockerfile.deploy -t librobot:deploy .

echo "Docker images built successfully!"
echo "Available images:"
docker images | grep librobot
