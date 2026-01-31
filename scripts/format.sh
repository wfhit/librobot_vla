#!/bin/bash
# Local code formatter using Docker
# Runs black, isort, and ruff to auto-format all code
#
# Usage:
#   ./scripts/format.sh              # Format all code
#   ./scripts/format.sh --check      # Check only, don't modify
#   ./scripts/format.sh librobot/    # Format specific directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default settings
CHECK_ONLY=false
TARGET_DIRS="librobot tests scripts"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --check|-c)
            CHECK_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS] [DIRECTORIES]"
            echo ""
            echo "Options:"
            echo "  --check, -c    Check formatting without modifying files"
            echo "  --help, -h     Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Format all code"
            echo "  $0 --check            # Check only"
            echo "  $0 librobot/          # Format specific directory"
            exit 0
            ;;
        *)
            TARGET_DIRS="$1"
            shift
            ;;
    esac
done

echo -e "${YELLOW}🐳 Running code formatters via Docker...${NC}"

# Build or use existing image
IMAGE_NAME="librobot-formatter"

if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
    echo -e "${YELLOW}Building formatter image...${NC}"
    docker build -t "$IMAGE_NAME" -f - "$PROJECT_ROOT" << 'DOCKERFILE'
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    black==24.1.1 \
    ruff==0.1.15 \
    isort==5.13.2

WORKDIR /workspace
ENTRYPOINT ["/bin/bash", "-c"]
DOCKERFILE
fi

# Run formatters
if [ "$CHECK_ONLY" = true ]; then
    echo -e "${YELLOW}Checking code format...${NC}"
    
    docker run --rm -v "$PROJECT_ROOT:/workspace" "$IMAGE_NAME" "
        echo '=== Checking isort ===' && \
        isort --check-only --profile black --line-length 100 $TARGET_DIRS && \
        echo '=== Checking black ===' && \
        black --check --line-length 100 $TARGET_DIRS && \
        echo '=== Checking ruff ===' && \
        ruff check $TARGET_DIRS
    "
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ All checks passed!${NC}"
    else
        echo -e "${RED}❌ Formatting issues found. Run '$0' to fix.${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Formatting code...${NC}"
    
    docker run --rm -v "$PROJECT_ROOT:/workspace" "$IMAGE_NAME" "
        echo '=== Running isort ===' && \
        isort --profile black --line-length 100 $TARGET_DIRS && \
        echo '=== Running black ===' && \
        black --line-length 100 $TARGET_DIRS && \
        echo '=== Running ruff --fix ===' && \
        ruff check --fix $TARGET_DIRS || true
    "
    
    echo -e "${GREEN}✅ Formatting complete!${NC}"
    
    # Show what changed
    cd "$PROJECT_ROOT"
    if [ -n "$(git status --porcelain)" ]; then
        echo -e "${YELLOW}Modified files:${NC}"
        git status --short
    else
        echo -e "${GREEN}No changes needed.${NC}"
    fi
fi
