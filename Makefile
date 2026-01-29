.PHONY: help install install-dev install-train install-inference install-all clean lint format test docker-build docker-run

help:
	@echo "LibroBot VLA Framework"
	@echo ""
	@echo "Available commands:"
	@echo "  install          - Install core dependencies"
	@echo "  install-dev      - Install development dependencies"
	@echo "  install-train    - Install training dependencies"
	@echo "  install-inference- Install inference dependencies"
	@echo "  install-all      - Install all dependencies"
	@echo "  clean            - Remove build artifacts"
	@echo "  lint             - Run linters"
	@echo "  format           - Format code with black and isort"
	@echo "  test             - Run tests"
	@echo "  docker-build     - Build Docker images"
	@echo "  docker-run       - Run Docker container"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-train:
	pip install -e ".[train]"

install-inference:
	pip install -e ".[inference]"

install-all:
	pip install -e ".[all]"

clean:
	rm -rf build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.so" -delete

lint:
	flake8 librobot tests --max-line-length=100 --extend-ignore=E203,W503
	mypy librobot --ignore-missing-imports

format:
	black librobot tests scripts
	isort librobot tests scripts

test:
	pytest tests/ -v --cov=librobot --cov-report=html --cov-report=term

docker-build:
	cd docker && bash scripts/build.sh

docker-run:
	cd docker && bash scripts/run_train.sh
