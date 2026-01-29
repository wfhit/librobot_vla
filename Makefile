.PHONY: help install install-dev install-all clean lint format test test-unit test-integration test-benchmark docs docker-build docker-run

help:
	@echo "LibroBot VLA Framework - Available targets:"
	@echo "  install          - Install base dependencies"
	@echo "  install-dev      - Install development dependencies"
	@echo "  install-all      - Install all dependencies"
	@echo "  clean            - Remove build artifacts and cache"
	@echo "  lint             - Run code linting (ruff)"
	@echo "  format           - Format code (black)"
	@echo "  type-check       - Run type checking (mypy)"
	@echo "  test             - Run all tests"
	@echo "  test-unit        - Run unit tests"
	@echo "  test-integration - Run integration tests"
	@echo "  test-benchmark   - Run benchmark tests"
	@echo "  docs             - Build documentation"
	@echo "  docker-build     - Build Docker images"
	@echo "  docker-run       - Run Docker container"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all]"

clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

lint:
	ruff check librobot/ tests/ scripts/

format:
	black librobot/ tests/ scripts/
	ruff check --fix librobot/ tests/ scripts/

type-check:
	mypy librobot/

test:
	pytest tests/

test-unit:
	pytest tests/unit/

test-integration:
	pytest tests/integration/

test-benchmark:
	pytest tests/benchmarks/

docs:
	cd docs && make html

docker-build:
	./docker/scripts/build.sh

docker-run:
	./docker/scripts/run_dev.sh
