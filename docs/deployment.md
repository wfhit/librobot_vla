# Deployment Guide

This guide covers deploying LibroBot VLA models to production environments. Learn how to optimize, containerize, and serve your models at scale.

## Table of Contents

- [Overview](#overview)
- [Pre-Deployment Checklist](#pre-deployment-checklist)
- [Model Optimization](#model-optimization)
- [Docker Deployment](#docker-deployment)
- [Inference Server](#inference-server)
- [Production Best Practices](#production-best-practices)
- [Monitoring and Logging](#monitoring-and-logging)
- [Scaling](#scaling)
- [Troubleshooting](#troubleshooting)

## Overview

### Deployment Options

LibroBot VLA supports multiple deployment strategies:

1. **Direct Python**: Simple scripts for development
2. **Docker Containers**: Reproducible deployments
3. **FastAPI Server**: REST API for web integration
4. **gRPC Server**: High-performance RPC
5. **Edge Devices**: Optimized for resource-constrained hardware

### Typical Production Stack

```
┌─────────────────────────────────────────┐
│         Load Balancer (nginx)           │
└────────────────┬────────────────────────┘
                 │
     ┌───────────┴───────────┐
     │                       │
┌────▼────┐            ┌────▼────┐
│ Server 1│            │ Server 2│
│         │            │         │
│ FastAPI │            │ FastAPI │
│   +     │            │   +     │
│  Model  │            │  Model  │
└─────────┘            └─────────┘
     │                       │
     └───────────┬───────────┘
                 │
         ┌───────▼────────┐
         │   Monitoring   │
         │ (Prometheus +  │
         │    Grafana)    │
         └────────────────┘
```

## Pre-Deployment Checklist

### ✓ Model Validation

```bash
# Test your model thoroughly
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --config configs/experiment/my_experiment.yaml \
    --split test

# Check inference speed
python scripts/benchmark.py \
    --checkpoint checkpoints/best_model.pt \
    --batch-size 1 \
    --num-iterations 100
```

### ✓ Export Model

```bash
# Export to TorchScript (recommended)
python scripts/export.py \
    --checkpoint checkpoints/best_model.pt \
    --output models/model.pt \
    --format torchscript

# Or export to ONNX
python scripts/export.py \
    --checkpoint checkpoints/best_model.pt \
    --output models/model.onnx \
    --format onnx
```

### ✓ Create Deployment Configuration

Create `configs/deployment/production.yaml`:

```yaml
# Production Deployment Configuration

# Model
model:
  checkpoint_path: "models/model.pt"
  format: "torchscript"  # or "onnx", "tensorrt"
  device: "cuda:0"

# Inference
inference:
  batch_size: 1
  precision: "fp16"  # fp32, fp16, bf16
  compile: true  # torch.compile for speedup
  
  # Optimization
  optimization:
    use_flash_attention: true
    gradient_checkpointing: false  # Disable for inference
    
  # Caching
  cache_embeddings: true
  cache_size: 1000

# Server
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 30  # seconds
  
  # Rate limiting
  rate_limit:
    requests_per_minute: 60
    
  # CORS
  cors:
    enabled: true
    origins: ["*"]

# Monitoring
monitoring:
  prometheus:
    enabled: true
    port: 9090
  logging:
    level: "INFO"
    format: "json"
    file: "logs/inference.log"

# Safety
safety:
  max_request_size: "10MB"
  timeout: 30
  enable_auth: true
  api_key_env: "LIBROBOT_API_KEY"
```

### ✓ Prepare Dependencies

```bash
# Create requirements file for production
pip freeze > requirements.txt

# Or use minimal dependencies
cat > requirements-prod.txt << EOF
torch>=2.9.0
torchvision>=0.24.0
transformers>=4.57.0
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pillow>=10.0.0
numpy>=1.24.0
pydantic>=2.0.0
EOF
```

## Model Optimization

### 1. Quantization

Reduce model size and increase speed:

```python
# quantize.py
import torch
from librobot.models import create_vla, create_vlm

# Load model
vlm = create_vlm("qwen2-vl-2b", pretrained=True)
vla = create_vla("groot", vlm=vlm, action_dim=7)
vla.load_state_dict(torch.load("checkpoints/best_model.pt"))
vla.eval()

# Quantize to INT8
vla_quantized = torch.quantization.quantize_dynamic(
    vla,
    {torch.nn.Linear},  # Quantize linear layers
    dtype=torch.qint8
)

# Test
images = torch.randn(1, 3, 224, 224)
text = ["pick up the cup"]
proprio = torch.randn(1, 14)

with torch.no_grad():
    actions = vla_quantized.predict_action(images, text, proprio)

print(f"Quantized model size: {torch.save(vla_quantized, 'temp.pt')}")
torch.save(vla_quantized, "models/model_quantized.pt")
```

### 2. TorchScript Export

Convert to optimized format:

```python
# export_torchscript.py
import torch
from librobot.models import create_vla, create_vlm

# Load model
vlm = create_vlm("qwen2-vl-2b", pretrained=True)
vla = create_vla("groot", vlm=vlm, action_dim=7)
vla.load_state_dict(torch.load("checkpoints/best_model.pt"))
vla.eval()

# Create example inputs
example_images = torch.randn(1, 3, 224, 224)
example_text = ["pick up the cup"]
example_proprio = torch.randn(1, 14)

# Trace the model
with torch.no_grad():
    traced_model = torch.jit.trace(
        vla,
        (example_images, example_text, example_proprio)
    )

# Save
traced_model.save("models/model_traced.pt")

# Test loaded model
loaded_model = torch.jit.load("models/model_traced.pt")
output = loaded_model(example_images, example_text, example_proprio)
print(f"Output shape: {output.shape}")
```

### 3. ONNX Export

For cross-platform deployment:

```python
# export_onnx.py
import torch
import torch.onnx
from librobot.models import create_vla, create_vlm

# Load model
vlm = create_vlm("qwen2-vl-2b", pretrained=True)
vla = create_vla("groot", vlm=vlm, action_dim=7)
vla.load_state_dict(torch.load("checkpoints/best_model.pt"))
vla.eval()

# Example inputs
example_images = torch.randn(1, 3, 224, 224)
example_proprio = torch.randn(1, 14)

# Export to ONNX
torch.onnx.export(
    vla,
    (example_images, ["pick up"], example_proprio),
    "models/model.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["images", "proprio"],
    output_names=["actions"],
    dynamic_axes={
        "images": {0: "batch_size"},
        "proprio": {0: "batch_size"},
        "actions": {0: "batch_size"}
    }
)

# Verify ONNX model
import onnx
onnx_model = onnx.load("models/model.onnx")
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")
```

### 4. TensorRT Optimization (NVIDIA GPUs)

Maximum performance on NVIDIA hardware:

```python
# optimize_tensorrt.py
import torch
import torch_tensorrt

# Load TorchScript model
model = torch.jit.load("models/model_traced.pt").cuda()

# Compile with TensorRT
trt_model = torch_tensorrt.compile(
    model,
    inputs=[
        torch_tensorrt.Input(
            min_shape=[1, 3, 224, 224],
            opt_shape=[1, 3, 224, 224],
            max_shape=[4, 3, 224, 224],
            dtype=torch.float16
        ),
        torch_tensorrt.Input(
            min_shape=[1, 14],
            opt_shape=[1, 14],
            max_shape=[4, 14],
            dtype=torch.float16
        )
    ],
    enabled_precisions={torch.float16},
    workspace_size=1 << 30  # 1GB
)

# Save
torch.jit.save(trt_model, "models/model_trt.pt")
```

## Docker Deployment

### Option 1: Pre-built Images

Use the provided Dockerfiles:

```bash
# Build base image
docker build -t librobot-base -f docker/Dockerfile.base .

# Build deployment image
docker build -t librobot-deploy -f docker/Dockerfile.deploy .

# Run container
docker run --gpus all -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/configs:/app/configs \
    -e LIBROBOT_CONFIG=configs/deployment/production.yaml \
    librobot-deploy
```

### Option 2: Custom Dockerfile

Create a custom deployment Dockerfile:

```dockerfile
# Dockerfile.custom
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch
RUN pip3 install --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Install LibroBot
COPY requirements-prod.txt /tmp/
RUN pip install -r /tmp/requirements-prod.txt

# Copy application
WORKDIR /app
COPY librobot/ librobot/
COPY configs/ configs/
COPY models/ models/
COPY scripts/inference.py .

# Install LibroBot
RUN pip install -e .

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "inference.py", "--server", "--config", "configs/deployment/production.yaml"]
```

Build and run:

```bash
docker build -t my-vla-server -f Dockerfile.custom .
docker run --gpus all -p 8000:8000 my-vla-server
```

### Option 3: Docker Compose

For multi-service deployments:

```yaml
# docker-compose.yml
version: '3.8'

services:
  librobot-server:
    build:
      context: .
      dockerfile: docker/Dockerfile.deploy
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./configs:/app/configs
      - ./logs:/app/logs
    environment:
      - LIBROBOT_CONFIG=configs/deployment/production.yaml
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - librobot-server
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  prometheus-data:
  grafana-data:
```

Start services:

```bash
docker-compose up -d
```

## Inference Server

### FastAPI Server

Create `inference_server.py`:

```python
"""
FastAPI Inference Server for LibroBot VLA
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
from PIL import Image
import io
from typing import List, Optional

from librobot.models import create_vla, create_vlm
from librobot.utils.config import Config

# Create FastAPI app
app = FastAPI(
    title="LibroBot VLA Inference Server",
    description="REST API for robot action prediction",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
config = Config.load("configs/deployment/production.yaml")

# Load model (do this once at startup)
device = torch.device(config.model.device)
print(f"Loading model from {config.model.checkpoint_path}...")
vlm = create_vlm("qwen2-vl-2b", pretrained=True)
vla = create_vla("groot", vlm=vlm, action_dim=7)
vla.load_state_dict(torch.load(config.model.checkpoint_path, map_location=device))
vla.to(device)
vla.eval()
print("Model loaded successfully!")

# Request/Response models
class PredictRequest(BaseModel):
    text: str
    proprioception: List[float]

class PredictResponse(BaseModel):
    actions: List[float]
    confidence: Optional[float] = None
    latency_ms: float

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "LibroBot VLA",
        "version": "0.1.0"
    }

@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "gpu_available": torch.cuda.is_available(),
        "device": str(device)
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    text: str = Form(...),
    proprioception: str = Form(...)  # JSON string of floats
):
    """
    Predict robot actions from image, text, and proprioception.
    
    Args:
        image: Robot camera image (JPEG/PNG)
        text: Natural language instruction
        proprioception: Current robot state (JSON array of floats)
    
    Returns:
        Predicted actions and metadata
    """
    import time
    start_time = time.time()
    
    try:
        # Parse proprioception
        import json
        proprio = json.loads(proprioception)
        proprio = np.array(proprio, dtype=np.float32)
        
        # Load and preprocess image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        pil_image = pil_image.resize((224, 224))
        image_tensor = torch.from_numpy(np.array(pil_image)).float()
        image_tensor = image_tensor.permute(2, 0, 1) / 255.0  # [C, H, W]
        image_tensor = image_tensor.unsqueeze(0).to(device)  # [1, C, H, W]
        
        # Convert proprioception to tensor
        proprio_tensor = torch.from_numpy(proprio).float().unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            actions = vla.predict_action(
                images=image_tensor,
                text=[text],
                proprioception=proprio_tensor
            )
        
        # Convert to list
        actions_list = actions[0].cpu().numpy().tolist()
        
        # Compute latency
        latency_ms = (time.time() - start_time) * 1000
        
        return PredictResponse(
            actions=actions_list,
            latency_ms=latency_ms
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(requests: List[PredictRequest]):
    """Batch prediction endpoint."""
    # Implementation for batch processing
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        workers=config.server.workers
    )
```

Run the server:

```bash
# Development
python inference_server.py

# Production with Gunicorn
gunicorn inference_server:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120
```

### Client Example

```python
# client.py
import requests
from PIL import Image
import json

# Prepare request
url = "http://localhost:8000/predict"

with open("robot_view.jpg", "rb") as f:
    image_data = f.read()

data = {
    "text": "pick up the red cup",
    "proprioception": json.dumps([0.0] * 14)
}

files = {
    "image": ("image.jpg", image_data, "image/jpeg")
}

# Send request
response = requests.post(url, data=data, files=files)

if response.status_code == 200:
    result = response.json()
    print(f"Actions: {result['actions']}")
    print(f"Latency: {result['latency_ms']:.2f}ms")
else:
    print(f"Error: {response.text}")
```

## Production Best Practices

### 1. Use Environment Variables

```bash
# .env
LIBROBOT_MODEL_PATH=/models/model.pt
LIBROBOT_CONFIG=/configs/production.yaml
LIBROBOT_API_KEY=your-secret-key
CUDA_VISIBLE_DEVICES=0
LOG_LEVEL=INFO
```

```python
import os
from dotenv import load_dotenv

load_dotenv()

model_path = os.getenv("LIBROBOT_MODEL_PATH")
api_key = os.getenv("LIBROBOT_API_KEY")
```

### 2. Add Authentication

```python
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
import os

api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("LIBROBOT_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key

@app.post("/predict")
async def predict(
    api_key: str = Security(verify_api_key),
    # ... other parameters
):
    # Your prediction logic
    pass
```

### 3. Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("60/minute")
async def predict(request: Request):
    # Your prediction logic
    pass
```

### 4. Input Validation

```python
from pydantic import BaseModel, validator, Field

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=200)
    proprioception: List[float] = Field(..., min_items=14, max_items=14)
    
    @validator('proprioception')
    def validate_proprio(cls, v):
        if not all(-10.0 <= x <= 10.0 for x in v):
            raise ValueError('Proprioception values out of range')
        return v
```

### 5. Error Handling

```python
import logging

logger = logging.getLogger(__name__)

@app.post("/predict")
async def predict(...):
    try:
        # Prediction logic
        return result
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA OOM")
        raise HTTPException(status_code=503, detail="Server overloaded")
    except Exception as e:
        logger.exception("Unexpected error")
        raise HTTPException(status_code=500, detail="Internal server error")
```

## Monitoring and Logging

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

# Metrics
request_count = Counter(
    'librobot_requests_total',
    'Total number of requests'
)

request_latency = Histogram(
    'librobot_request_latency_seconds',
    'Request latency in seconds'
)

gpu_memory = Gauge(
    'librobot_gpu_memory_bytes',
    'GPU memory usage in bytes'
)

@app.post("/predict")
async def predict(...):
    request_count.inc()
    
    with request_latency.time():
        # Prediction logic
        result = model.predict(...)
    
    # Update GPU memory
    if torch.cuda.is_available():
        gpu_memory.set(torch.cuda.memory_allocated())
    
    return result

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

### Structured Logging

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)

# Configure logging
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)
```

### Grafana Dashboard

Create `grafana-dashboard.json`:

```json
{
  "dashboard": {
    "title": "LibroBot VLA Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(librobot_requests_total[5m])"
          }
        ]
      },
      {
        "title": "P95 Latency",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, librobot_request_latency_seconds)"
          }
        ]
      },
      {
        "title": "GPU Memory Usage",
        "targets": [
          {
            "expr": "librobot_gpu_memory_bytes"
          }
        ]
      }
    ]
  }
}
```

## Scaling

### Horizontal Scaling

Use multiple server instances behind a load balancer:

```nginx
# nginx.conf
upstream librobot_servers {
    least_conn;
    server librobot-server-1:8000;
    server librobot-server-2:8000;
    server librobot-server-3:8000;
    server librobot-server-4:8000;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://librobot_servers;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
        proxy_read_timeout 300;
    }
}
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: librobot-vla
spec:
  replicas: 4
  selector:
    matchLabels:
      app: librobot-vla
  template:
    metadata:
      labels:
        app: librobot-vla
    spec:
      containers:
      - name: librobot
        image: librobot-deploy:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "16Gi"
          requests:
            nvidia.com/gpu: 1
            memory: "8Gi"
        env:
        - name: LIBROBOT_CONFIG
          value: "/app/configs/production.yaml"
---
apiVersion: v1
kind: Service
metadata:
  name: librobot-service
spec:
  selector:
    app: librobot-vla
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Troubleshooting

### High Latency

**Solutions**:
1. Enable model compilation: `model = torch.compile(model)`
2. Use TensorRT optimization
3. Reduce batch size
4. Check GPU utilization: `nvidia-smi`

### Memory Issues

**Solutions**:
1. Use quantization (INT8/FP16)
2. Enable gradient checkpointing (if applicable)
3. Clear cache: `torch.cuda.empty_cache()`
4. Monitor memory: `torch.cuda.memory_summary()`

### Connection Timeouts

**Solutions**:
1. Increase timeout in nginx/load balancer
2. Add request queuing
3. Implement async processing
4. Scale horizontally

---

**Next Steps:**

- [Monitoring Best Practices](monitoring.md)
- [Security Guide](security.md)
- [Scaling Guide](scaling.md)

For production support, see [Production Runbook](production_runbook.md).
