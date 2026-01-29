# LibroBot VLA Scripts

This directory contains entry point scripts for training, evaluation, inference, and exporting LibroBot VLA models.

## Scripts Overview

### 1. train.py - Training Entry Point
Train VLA models with comprehensive configuration support.

**Features:**
- Load configurations from YAML files
- Distributed training (DDP, FSDP, DeepSpeed)
- Mixed precision training
- Checkpoint resumption
- WandB and TensorBoard logging
- Evaluation during training

**Basic Usage:**
```bash
# Train with default config
python scripts/train.py

# Train with custom config
python scripts/train.py --config configs/experiment/my_experiment.yaml

# Resume from checkpoint
python scripts/train.py --config configs/experiment/my_experiment.yaml \
    --resume checkpoints/checkpoint_epoch_10.pt

# Override config values from CLI
python scripts/train.py --config configs/experiment/my_experiment.yaml \
    --override training.max_epochs=100 model.hidden_size=768

# Distributed training with torchrun
torchrun --nproc_per_node=4 scripts/train.py --config configs/experiment/my_experiment.yaml
```

**Key Arguments:**
- `--config`: Path to configuration YAML file
- `--resume`: Path to checkpoint to resume from
- `--output-dir`: Directory for outputs (checkpoints, logs)
- `--override`: Override config values (e.g., `training.max_epochs=100`)
- `--dry-run`: Run a few iterations to verify setup
- `--validate-only`: Only run validation, don't train
- `--no-wandb`: Disable Weights & Biases logging

---

### 2. evaluate.py - Evaluation Script
Evaluate trained models on test datasets and compute comprehensive metrics.

**Features:**
- Load models from checkpoints
- Evaluate on test datasets
- Compute comprehensive metrics (MSE, MAE, RMSE, per-dimension metrics)
- Save predictions and visualizations
- Generate evaluation reports
- Support for multiple checkpoint evaluation

**Basic Usage:**
```bash
# Evaluate on test set
python scripts/evaluate.py --checkpoint checkpoints/best.pt \
    --config configs/experiment/my_experiment.yaml

# Evaluate with custom test data
python scripts/evaluate.py --checkpoint checkpoints/best.pt \
    --config configs/experiment/my_experiment.yaml \
    --test-data path/to/test_data

# Evaluate and save predictions
python scripts/evaluate.py --checkpoint checkpoints/best.pt \
    --config configs/experiment/my_experiment.yaml \
    --save-predictions --output-dir results/evaluation

# Evaluate multiple checkpoints (using glob patterns)
python scripts/evaluate.py --checkpoint "checkpoints/*.pt" \
    --config configs/experiment/my_experiment.yaml \
    --output-dir results/multi_checkpoint_eval
```

**Key Arguments:**
- `--checkpoint`: Path to checkpoint file(s) (supports glob patterns)
- `--config`: Path to configuration YAML file
- `--test-data`: Path to test dataset
- `--output-dir`: Directory for evaluation outputs
- `--save-predictions`: Save model predictions to file
- `--save-visualizations`: Save visualizations of predictions
- `--metrics`: Specific metrics to compute

**Output Files:**
- `metrics.json`: Computed metrics
- `predictions.npz`: Model predictions (if `--save-predictions`)
- `evaluation_report.txt`: Summary report
- `config.yaml`: Configuration used for evaluation

---

### 3. inference.py - Inference Script
Run inference with VLA models in three modes: single inference, batch inference, or server mode.

**Features:**
- Single inference on images/text/state
- Batch inference on directories
- REST API server with FastAPI
- gRPC server for high-performance inference
- Model compilation with torch.compile
- Multiple precision modes (FP32, FP16, BF16)

**Basic Usage:**

#### Single Inference
```bash
# Inference with image and text
python scripts/inference.py --checkpoint checkpoints/best.pt \
    --image path/to/image.jpg \
    --text "Pick up the red block"

# Inference with image, text, and robot state
python scripts/inference.py --checkpoint checkpoints/best.pt \
    --image path/to/image.jpg \
    --text "Pick up the red block" \
    --state "[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]"
```

#### Batch Inference
```bash
# Batch inference on directory
python scripts/inference.py --checkpoint checkpoints/best.pt \
    --config configs/experiment/my_experiment.yaml \
    --batch-dir path/to/images/ \
    --output-file predictions.json
```

#### Server Mode
```bash
# Start REST API server
python scripts/inference.py --checkpoint checkpoints/best.pt \
    --server rest --host 0.0.0.0 --port 8000

# Start gRPC server
python scripts/inference.py --checkpoint checkpoints/best.pt \
    --server grpc --host 0.0.0.0 --port 50051

# REST server with CORS enabled
python scripts/inference.py --checkpoint checkpoints/best.pt \
    --config configs/experiment/my_experiment.yaml \
    --server rest --port 8000 --enable-cors
```

**REST API Endpoints:**
- `GET /health`: Health check
- `GET /info`: Server information
- `POST /predict`: Run prediction
- `POST /reset`: Reset model state

**Key Arguments:**
- `--checkpoint`: Path to model checkpoint
- `--config`: Path to configuration YAML file (optional)
- `--server`: Start server (rest or grpc)
- `--image`: Path to input image for single inference
- `--text`: Text instruction for single inference
- `--state`: Robot state as JSON string or file path
- `--batch-dir`: Directory for batch inference
- `--host`: Server host address
- `--port`: Server port
- `--device`: Device to run on (cuda/cpu/mps)
- `--precision`: Inference precision (fp32/fp16/bf16)
- `--compile`: Compile model with torch.compile

---

### 4. export.py - Model Export Script
Export trained models to various deployment formats.

**Features:**
- Export to ONNX, TorchScript, TensorRT, CoreML, OpenVINO
- Model optimization and quantization
- Validation of exported models
- Support for dynamic batch sizes
- Generate deployment metadata

**Basic Usage:**
```bash
# Export to ONNX
python scripts/export.py --checkpoint checkpoints/best.pt \
    --format onnx --output models/model.onnx

# Export to TorchScript
python scripts/export.py --checkpoint checkpoints/best.pt \
    --format torchscript --output models/model.pt

# Export with optimization
python scripts/export.py --checkpoint checkpoints/best.pt \
    --format onnx --output models/model.onnx \
    --optimize --opset-version 14

# Export to multiple formats
python scripts/export.py --checkpoint checkpoints/best.pt \
    --format onnx torchscript \
    --output-dir models/exports

# Export with custom input shapes
python scripts/export.py --checkpoint checkpoints/best.pt \
    --config configs/experiment/my_experiment.yaml \
    --format onnx --output models/model.onnx \
    --input-shape 1,3,224,224

# Export and validate
python scripts/export.py --checkpoint checkpoints/best.pt \
    --format onnx --output models/model.onnx \
    --validate --validation-samples 10
```

**Supported Formats:**
- **ONNX**: For deployment with ONNX Runtime
- **TorchScript**: For deployment in C++ or mobile
- **TensorRT**: For optimized NVIDIA GPU inference
- **CoreML**: For Apple devices (iOS, macOS)
- **OpenVINO**: For Intel hardware

**Key Arguments:**
- `--checkpoint`: Path to model checkpoint
- `--format`: Export format(s) (onnx/torchscript/tensorrt/coreml/openvino)
- `--output`: Output file path (for single format)
- `--output-dir`: Output directory (for multiple formats)
- `--input-shape`: Input shape specification
- `--batch-size`: Batch size for exported model
- `--opset-version`: ONNX opset version
- `--optimize`: Apply optimization to exported model
- `--validate`: Validate exported model against original
- `--half-precision`: Export model in FP16

**Output Files:**
- `model.[onnx|pt|trt|mlmodel|xml]`: Exported model
- `export_metadata.json`: Export metadata and configuration
- `export.log`: Export log file

---

## Common Patterns

### Training with Evaluation
```bash
# Train with periodic evaluation
python scripts/train.py --config configs/experiment/my_experiment.yaml

# After training, run final evaluation
python scripts/evaluate.py --checkpoint outputs/best_checkpoint.pt \
    --config configs/experiment/my_experiment.yaml \
    --save-predictions --output-dir results/final_eval
```

### Export and Deploy Workflow
```bash
# 1. Export model
python scripts/export.py --checkpoint checkpoints/best.pt \
    --config configs/experiment/my_experiment.yaml \
    --format onnx --output models/model.onnx \
    --optimize --validate

# 2. Start inference server with exported model
python scripts/inference.py --checkpoint models/model.onnx \
    --server rest --port 8000
```

### Docker Deployment
```bash
# Build Docker image (from repository root)
docker build -t librobot-vla -f docker/Dockerfile .

# Run training in Docker
docker run --gpus all -v $(pwd)/data:/data -v $(pwd)/outputs:/outputs \
    librobot-vla python scripts/train.py --config /app/configs/defaults.yaml

# Run inference server in Docker
docker run --gpus all -p 8000:8000 \
    librobot-vla python scripts/inference.py \
    --checkpoint /outputs/best_checkpoint.pt \
    --server rest --host 0.0.0.0 --port 8000
```

---

## Configuration

All scripts support loading configuration from YAML files. The configuration follows a hierarchical structure:

```yaml
# configs/experiment/my_experiment.yaml
seed: 42
device: cuda
mixed_precision: true

model:
  name: "vla_model"
  hidden_size: 768
  num_layers: 12

dataset:
  train:
    name: "lerobot"
    path: "path/to/train_data"
  val:
    name: "lerobot"
    path: "path/to/val_data"

training:
  batch_size: 32
  max_epochs: 100
  gradient_clip_norm: 1.0
  output_dir: "./outputs"

optimizer:
  name: "adamw"
  lr: 3e-4
  weight_decay: 0.01

scheduler:
  name: "cosine"
  warmup_steps: 1000

logging:
  log_interval: 10
  eval_interval: 500
  save_interval: 1000
  wandb:
    enabled: true
    project: "librobot-vla"
```

---

## Environment Setup

### Install Dependencies
```bash
# Basic installation
pip install -e .

# With all extras for full functionality
pip install -e ".[dev,onnx,tensorboard,wandb]"

# For ONNX export
pip install onnx onnxruntime onnxoptimizer

# For TensorRT export (requires CUDA)
pip install torch-tensorrt
```

### Environment Variables
```bash
# Enable CUDA
export CUDA_VISIBLE_DEVICES=0,1,2,3

# WandB authentication
export WANDB_API_KEY=your_api_key

# Distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=29500
```

---

## Troubleshooting

### Common Issues

**Issue: "CUDA out of memory"**
```bash
# Solution 1: Reduce batch size
python scripts/train.py --config config.yaml --override training.batch_size=16

# Solution 2: Enable gradient accumulation
python scripts/train.py --config config.yaml \
    --override training.gradient_accumulation_steps=4
```

**Issue: "Checkpoint not found"**
```bash
# Check checkpoint path
ls -lh checkpoints/

# Use absolute path
python scripts/evaluate.py --checkpoint $(pwd)/checkpoints/best.pt --config config.yaml
```

**Issue: "Module not found"**
```bash
# Ensure package is installed
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## Additional Resources

- **Documentation**: See `docs/` directory for detailed guides
- **Examples**: See `examples/` directory for complete examples
- **Docker**: See `docker/` directory for containerization
- **Configs**: See `configs/` directory for configuration templates

---

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check the documentation in `docs/`
- See examples in `examples/`
