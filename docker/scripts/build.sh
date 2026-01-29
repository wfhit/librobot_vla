#!/bin/bash
# Build Docker images for LibroBot VLA

set -e

echo "Building LibroBot VLA Docker images..."

# Build base image
echo "Building base image..."
docker build -t librobot-base:latest -f ../Dockerfile.base ../../

# Build training image
echo "Building training image..."
docker build -t librobot-train:latest -f ../Dockerfile.train ../../

# Build deployment image
echo "Building deployment image..."
docker build -t librobot-deploy:latest -f ../Dockerfile.deploy ../../

echo "All images built successfully!"
echo ""
echo "Available images:"
docker images | grep librobot
