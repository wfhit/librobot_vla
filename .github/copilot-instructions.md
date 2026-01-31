# LibroBot VLA - AI Coding Agent Instructions

## Project Overview

LibroBot VLA is a modular robotics framework for Vision-Language-Action (VLA) models. The architecture follows a strict registry-based plugin pattern enabling new components without modifying core code.

**Core data flow**: Camera Images + Text → VLM → Features → VLA Framework → Action Head → Robot Actions

## Architecture & Key Components

### Registry Pattern (Central Design)

All components register via decorators and instantiate through factory functions:

```python
# Registration (e.g., librobot/models/vlm/qwen_vl.py)
@register_vlm(name="qwen2-vl-2b", aliases=["qwen2-vl"])
class Qwen2VL2B(AbstractVLM):
    pass

# Instantiation anywhere in codebase
vlm = create_vlm("qwen2-vl-2b", pretrained=True)
```

**Registry locations**: Each component type has its own registry file:
- VLMs: [librobot/models/vlm/registry.py](librobot/models/vlm/registry.py)
- VLA Frameworks: [librobot/models/frameworks/registry.py](librobot/models/frameworks/registry.py)  
- Action Heads: [librobot/models/action_heads/registry.py](librobot/models/action_heads/registry.py)

### Adding New Components

1. **Inherit** from abstract base class (`AbstractVLM`, `AbstractVLA`, `AbstractActionHead`)
2. **Implement** all `@abstractmethod` functions
3. **Decorate** with `@register_*` to auto-register
4. **Import** in module's `__init__.py` to trigger registration

Base classes with required interfaces:
- [librobot/models/vlm/base.py](librobot/models/vlm/base.py) - `forward()`, `encode_image()`, `encode_text()`, `get_embedding_dim()`
- [librobot/models/frameworks/base.py](librobot/models/frameworks/base.py) - `forward()`, `predict_action()`, `compute_loss()`
- [librobot/models/action_heads/base.py](librobot/models/action_heads/base.py) - `forward()`, `compute_loss()`, `sample()`

### Configuration System

Uses Hydra/OmegaConf with hierarchical YAML configs:

```
configs/
├── defaults.yaml           # Global defaults (seed, device, logging)
├── model/                  # Component configs (vlm/, framework/, action_head/)
├── experiment/             # Complete experiment definitions
└── robot/                  # Robot-specific configs
```

**Override pattern** (CLI):
```bash
python scripts/train.py --config configs/experiment/my_exp.yaml \
    --override training.max_epochs=100 model.hidden_size=768
```

## Git Workflow

**The `main` branch is protected.** Always create a feature branch before making changes:

```bash
git checkout -b feature/my-feature-name   # Create and switch to new branch
# ... make changes ...
./scripts/format.sh                        # Format code before committing
git push -u origin feature/my-feature-name  # Push and open PR
```

### Code Formatting (Required Before Commit)

**Always run the Docker formatter before committing** to avoid CI failures:

```bash
./scripts/format.sh           # Auto-format all code (isort, black, ruff)
./scripts/format.sh --check   # Check only, don't modify files
./scripts/format.sh librobot/ # Format specific directory
```

This uses Docker to ensure consistent formatting regardless of local tool versions.

## Developer Commands

```bash
# Install
pip install -e ".[dev]"       # Development (includes pytest, black, ruff, mypy)
pip install -e ".[train]"     # Training (accelerate, deepspeed, wandb)
pip install -e ".[all]"       # Everything

# Test
make test                     # All tests
make test-unit               # Unit tests only
pytest tests/ -m "not slow"  # Skip slow tests
pytest tests/ -m "not gpu"   # Skip GPU tests

# Code quality
make lint                    # ruff check
make format                  # black + ruff fix
make type-check              # mypy

# Train
python scripts/train.py --config configs/experiment/default.yaml
torchrun --nproc_per_node=4 scripts/train.py  # Distributed
```

## Testing Conventions

- Test markers: `@pytest.mark.slow`, `@pytest.mark.gpu`, `@pytest.mark.integration`
- Fixtures in [tests/conftest.py](tests/conftest.py): `device`, `model_config`, `small_model_config`, `random_seed`
- Use `small_model_config` fixture for fast unit tests

## Code Patterns

### Return Dictionary Convention

All model `forward()` methods return dictionaries, not raw tensors:

```python
def forward(self, images, text, ...) -> dict[str, torch.Tensor]:
    return {
        "embeddings": ...,   # VLM
        "actions": ...,      # VLA
        "loss": ...,         # When training
    }
```

### Type Annotations

All functions use type hints. Use `list[T]` not `List[T]` (Python 3.12+):

```python
def encode_text(self, text: Union[str, list[str]], **kwargs) -> torch.Tensor:
```

### Framework naming

VLA frameworks in `librobot/models/frameworks/` use `*_style.py` suffix (e.g., `groot_style.py`, `pi0_style.py`) to indicate they are "inspired by" rather than exact reimplementations.

## Key Files for Context

- Entry point: [librobot/__init__.py](librobot/__init__.py) - All public exports
- Registry core: [librobot/utils/registry.py](librobot/utils/registry.py) - `Registry`, `GlobalRegistry`
- Training script: [scripts/train.py](scripts/train.py) - CLI args, distributed setup
- Example configs: [configs/experiment/default.yaml](configs/experiment/default.yaml)
