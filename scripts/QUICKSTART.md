# Quick Reference Guide for LibroBot Scripts

## Quick Start Commands

### Training
```bash
# Basic training
python scripts/train.py --config configs/experiment/my_experiment.yaml

# Training with overrides
python scripts/train.py --config config.yaml --override training.batch_size=64

# Distributed training
torchrun --nproc_per_node=4 scripts/train.py --config config.yaml

# Resume training
python scripts/train.py --config config.yaml --resume checkpoints/latest.pt
```

### Evaluation
```bash
# Basic evaluation
python scripts/evaluate.py --checkpoint checkpoints/best.pt --config config.yaml

# Save predictions
python scripts/evaluate.py --checkpoint best.pt --config config.yaml --save-predictions

# Evaluate multiple checkpoints
python scripts/evaluate.py --checkpoint "checkpoints/*.pt" --config config.yaml
```

### Inference
```bash
# Single inference
python scripts/inference.py --checkpoint best.pt --image img.jpg --text "Pick up block"

# Batch inference
python scripts/inference.py --checkpoint best.pt --batch-dir images/ --output predictions.json

# Start REST server
python scripts/inference.py --checkpoint best.pt --server rest --port 8000

# Start gRPC server
python scripts/inference.py --checkpoint best.pt --server grpc --port 50051
```

### Export
```bash
# Export to ONNX
python scripts/export.py --checkpoint best.pt --format onnx --output model.onnx

# Export with optimization
python scripts/export.py --checkpoint best.pt --format onnx --optimize --validate

# Export to multiple formats
python scripts/export.py --checkpoint best.pt --format onnx torchscript --output-dir exports/
```

## Common Options

### All Scripts
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `--log-file`: Path to log file
- `--device`: Device to use (cuda, cpu, mps)

### Config-based Scripts (train, evaluate, inference, export)
- `--config`: Path to configuration YAML file
- `--checkpoint`: Path to checkpoint file

## Quick Troubleshooting

### Issue: Module not found
```bash
pip install -e .
# or
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: CUDA out of memory
```bash
# Reduce batch size
python scripts/train.py --config config.yaml --override training.batch_size=16

# Enable gradient accumulation
python scripts/train.py --config config.yaml --override training.gradient_accumulation_steps=4
```

### Issue: Checkpoint not found
```bash
# Use absolute path
python scripts/evaluate.py --checkpoint $(pwd)/checkpoints/best.pt --config config.yaml
```

## Environment Setup

```bash
# Install basic dependencies
pip install -e .

# Install all extras
pip install -e ".[dev,onnx,tensorboard,wandb]"

# For ONNX export
pip install onnx onnxruntime onnxoptimizer

# For TensorRT export
pip install torch-tensorrt
```

## Docker Usage

```bash
# Build image
docker build -t librobot-vla -f docker/Dockerfile .

# Run training
docker run --gpus all -v $(pwd)/data:/data -v $(pwd)/outputs:/outputs \
    librobot-vla python scripts/train.py --config /app/configs/defaults.yaml

# Run inference server
docker run --gpus all -p 8000:8000 \
    librobot-vla python scripts/inference.py \
    --checkpoint /outputs/best_checkpoint.pt \
    --server rest --host 0.0.0.0 --port 8000
```

## File Locations

- **Scripts**: `scripts/`
- **Configs**: `configs/`
- **Checkpoints**: `outputs/checkpoints/` (default)
- **Logs**: `outputs/logs/` (default)
- **Exports**: `models/exports/` (default)
- **Results**: `results/` (default)

## For More Information

See `scripts/README.md` for comprehensive documentation.
