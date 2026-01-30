# LibroBot VLA Quick Reference

## Table of Contents
- [Visual Directory Tree](#visual-directory-tree)
- [Quick Lookup Tables](#quick-lookup-tables)
- [Common Patterns](#common-patterns)
- [Command Cheatsheet](#command-cheatsheet)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)

## Visual Directory Tree

```
librobot_vla/
│
├── 📦 librobot/                    # Main package
│   ├── 🧠 models/                  # Neural network models
│   │   ├── 👁️ vlm/                 # Vision-Language Models (5 families, 11 variants)
│   │   │   ├── qwen_vl.py         # Qwen2/3-VL (795 LOC)
│   │   │   ├── florence.py        # Florence-2 (730 LOC)
│   │   │   ├── paligemma.py       # PaliGemma (653 LOC)
│   │   │   ├── internvl.py        # InternVL2 (701 LOC)
│   │   │   ├── llava.py           # LLaVA (741 LOC)
│   │   │   ├── adapters/          # LoRA, QLoRA
│   │   │   └── utils/             # KV cache, attention
│   │   │
│   │   ├── 🤖 frameworks/          # VLA Frameworks (8 implementations)
│   │   │   ├── groot_style.py     # NVIDIA GR00T (290 LOC)
│   │   │   ├── pi0_style.py       # Physical Intelligence π0 (295 LOC)
│   │   │   ├── octo_style.py      # Berkeley Octo (355 LOC)
│   │   │   ├── openvla_style.py   # Berkeley OpenVLA (315 LOC)
│   │   │   ├── rt2_style.py       # Google RT-2 (365 LOC)
│   │   │   ├── act_style.py       # ALOHA ACT (420 LOC)
│   │   │   ├── helix_style.py     # Figure AI Helix (440 LOC)
│   │   │   └── custom.py          # Custom template (430 LOC)
│   │   │
│   │   ├── 🎯 action_heads/        # Action prediction mechanisms
│   │   │   ├── mlp_oft.py         # MLP output-from-tokens
│   │   │   ├── transformer_act.py # Transformer-based
│   │   │   ├── diffusion/         # DDPM, DDIM, EDM
│   │   │   └── flow_matching/     # Rectified Flow, OT-CFM
│   │   │
│   │   ├── 🔧 encoders/            # State, history, fusion encoders
│   │   └── 🧩 components/          # Reusable building blocks
│   │       ├── attention/         # Flash, sliding window, block-wise
│   │       ├── normalization/     # LayerNorm, RMSNorm, GroupNorm
│   │       └── positional/        # Sinusoidal, RoPE, ALiBi
│   │
│   ├── 📊 data/                    # Data handling
│   │   ├── datasets/              # RLDS, HDF5, custom
│   │   ├── transforms/            # Image & action transforms
│   │   └── tokenizers/            # Text & action tokenizers
│   │
│   ├── 🏋️ training/                # Training infrastructure
│   │   ├── trainer.py             # Main training loop
│   │   ├── losses/                # Loss functions
│   │   └── callbacks/             # Checkpointing, logging
│   │
│   ├── 🚀 inference/               # Model serving
│   │   ├── predictor.py           # Base predictor
│   │   └── server/                # FastAPI, gRPC servers
│   │
│   ├── 🦾 robots/                  # Robot interfaces
│   │   ├── so100_arm.py          # SO-100 robot arm
│   │   ├── humanoid.py           # Humanoid robots
│   │   └── wheel_loader.py       # Wheel loader
│   │
│   ├── 📈 evaluation/              # Evaluation tools
│   └── 🛠️ utils/                   # Shared utilities
│       ├── registry.py            # Component registry
│       ├── config.py              # Config management
│       └── checkpoint.py          # Checkpointing
│
├── ⚙️ configs/                     # Configuration files
│   ├── models/                    # Model configs
│   ├── data/                      # Data configs
│   └── training/                  # Training configs
│
├── 📜 scripts/                     # Executable scripts
│   ├── train.py                   # Training
│   ├── inference.py               # Inference
│   ├── evaluate.py                # Evaluation
│   └── export.py                  # Model export
│
├── 📚 examples/                    # Example code
├── 🧪 tests/                       # Test suite
└── 📖 docs/                        # Documentation
    └── design/                    # Design docs
        ├── ARCHITECTURE.md        # System architecture
        ├── PROJECT_STRUCTURE.md   # File organization
        ├── DESIGN_PRINCIPLES.md   # Design patterns
        ├── COMPONENT_GUIDE.md     # Adding components
        ├── API_CONTRACTS.md       # Interface definitions
        ├── ROADMAP.md             # Development roadmap
        └── QUICK_REFERENCE.md     # This file
```

## Quick Lookup Tables

### VLM Models

| Model | Size | Parameters | Hidden Dim | Use Case |
|-------|------|------------|------------|----------|
| Qwen2-VL | 2B, 7B | 2B, 7B | 1536, 3072 | General purpose, high quality |
| Qwen3-VL | 4B, 7B | 4B, 7B | 2048, 3072 | Latest version, best performance |
| Florence-2 | base, large | 230M, 770M | 768, 1024 | Multi-task, lightweight |
| PaliGemma | 3B | 3B | 2048 | Efficient, SigLIP vision |
| InternVL2 | 2B, 8B | 2B, 8B | 2048, 4096 | High-res images, multilingual |
| LLaVA | 7B, 13B | 7B, 13B | 4096, 5120 | CLIP vision, established baseline |

### VLA Frameworks

| Framework | Action Type | Best For | Params (Train) | Speed |
|-----------|-------------|----------|----------------|-------|
| GR00T | Diffusion | Multi-camera, stable | 39M (535K) | Fast |
| π0 | Flow Matching | Complex states | 42M (3.7M) | Medium |
| Octo | Direct MLP | Multi-task, cross-dataset | 6.7M (all) | Fast |
| OpenVLA | Direct MLP | Language-guided | 38.7M (133K) | Slow |
| RT-2 | Token-based | Discrete actions | 40.9M (2.4M) | Medium |
| ACT | Action Chunks | Bi-manual manipulation | 30.7M (all) | Medium |
| Helix | Hierarchical | Long-horizon tasks | 40.8M (2.3M) | Medium |
| Custom | Flexible | Experimentation | Variable | Variable |

### Action Heads

| Type | Description | Use Case | Complexity |
|------|-------------|----------|------------|
| MLP OFT | Simple MLP projection | Fast inference | Low |
| Transformer | Self-attention based | Sequence modeling | Medium |
| Autoregressive | Sequential prediction | Language-like actions | Medium |
| DDPM | Denoising diffusion | High-quality actions | High |
| DDIM | Deterministic diffusion | Fast sampling | High |
| Rectified Flow | Optimal transport flow | Smooth trajectories | High |
| OT-CFM | Conditional flow matching | Flexible conditioning | High |

### Neural Components

| Component | Variants | Location |
|-----------|----------|----------|
| Attention | Standard, Flash, Sliding Window, Block-wise | `models/components/attention/` |
| Normalization | LayerNorm, RMSNorm, GroupNorm | `models/components/normalization/` |
| Position Encoding | Sinusoidal, RoPE, ALiBi | `models/components/positional/` |
| Activations | SwiGLU, GeGLU, GELU, ReLU | `models/components/activations.py` |
| State Encoders | MLP, Transformer | `models/encoders/state/` |
| History Encoders | LSTM, Transformer | `models/encoders/history/` |
| Fusion | Attention, FiLM | `models/encoders/fusion/` |

### Dataset Formats

| Format | File Extension | Loader | Use Case |
|--------|---------------|--------|----------|
| RLDS | `.tfrecord` | `RLDSDataset` | Robot learning datasets |
| HDF5 | `.h5`, `.hdf5` | `HDF5Dataset` | Custom robot data |
| Dummy | N/A | `DummyDataset` | Testing, debugging |

### Robot Interfaces

| Robot | Type | DOF | Interface Class |
|-------|------|-----|-----------------|
| SO-100 | Arm | 7 | `SO100Arm` |
| Humanoid | Full body | 20+ | `Humanoid` |
| Wheel Loader | Mobile | Variable | `WheelLoaderRobot` |

## Common Patterns

### 1. Create a VLA Model

```python
from librobot.models.vlm import create_vlm
from librobot.models.frameworks import create_vla

# Create VLM
vlm = create_vlm("qwen2-vl-2b", pretrained=True)

# Create VLA
vla = create_vla(
    "groot",
    vlm=vlm,
    action_dim=7,
    state_dim=14,
    hidden_dim=512
)
```

### 2. Training Loop

```python
# Setup
optimizer = torch.optim.AdamW(vla.parameters(), lr=1e-4)
dataloader = DataLoader(dataset, batch_size=32)

# Training
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward
        outputs = vla(
            batch["images"],
            batch["text"],
            batch["proprioception"],
            batch["actions"]
        )
        loss = outputs["loss"]
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 3. Inference

```python
# Load model
vla = create_vla("groot", vlm=vlm, action_dim=7)
vla.load_state_dict(torch.load("checkpoint.pth"))
vla.eval()

# Predict
with torch.no_grad():
    actions = vla.predict_action(
        images=images,
        text="pick up the cup",
        proprioception=state
    )

# Execute on robot
robot.execute_action(actions.cpu().numpy())
```

### 4. Configuration-Driven

```python
from librobot.utils.config import load_config

# Load config
config = load_config("configs/models/groot.yaml")

# Create from config
vlm = create_vlm(**config["model"]["vlm"])
vla = create_vla(
    config["model"]["framework"],
    vlm=vlm,
    **config["model"]["action_head"]
)
```

### 5. Custom Component

```python
from librobot.models.vlm import AbstractVLM, register_vlm

@register_vlm(name="my-vlm", aliases=["mvlm"])
class MyVLM(AbstractVLM):
    def forward(self, images, text, **kwargs):
        # Implementation
        pass
    
    def encode_image(self, images):
        # Implementation
        pass
    
    def encode_text(self, text):
        # Implementation
        pass
    
    def get_embedding_dim(self):
        return 768
    
    @property
    def config(self):
        return {"name": "my-vlm"}

# Use immediately
vlm = create_vlm("my-vlm")
```

### 6. Multi-GPU Training

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize
dist.init_process_group("nccl")
vla = vla.to(local_rank)
vla = DDP(vla, device_ids=[local_rank])

# Train as usual
outputs = vla(images, text, proprioception, actions)
loss = outputs["loss"]
loss.backward()
```

### 7. Mixed Precision

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    # Forward with autocast
    with autocast():
        outputs = vla(
            batch["images"],
            batch["text"],
            batch["proprioception"],
            batch["actions"]
        )
        loss = outputs["loss"]
    
    # Backward with scaler
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 8. LoRA Fine-Tuning

```python
from librobot.models.vlm.adapters import LoRAAdapter

# Add LoRA to VLM
vlm = create_vlm("qwen2-vl-2b", pretrained=True)
vlm = LoRAAdapter(vlm, rank=8, alpha=16)

# Only LoRA parameters are trainable
trainable_params = sum(
    p.numel() for p in vlm.parameters() if p.requires_grad
)
print(f"Trainable: {trainable_params:,}")  # Much smaller!
```

## Command Cheatsheet

### Training

```bash
# Basic training
python scripts/train.py --config configs/models/groot.yaml

# Distributed training (single node, 4 GPUs)
torchrun --nproc_per_node=4 scripts/train.py --config configs/models/groot.yaml

# Resume from checkpoint
python scripts/train.py --config configs/models/groot.yaml --resume checkpoints/latest.pth

# Override config
python scripts/train.py --config configs/models/groot.yaml --batch_size 64 --lr 1e-3
```

### Inference

```bash
# Single inference
python scripts/inference.py --checkpoint checkpoints/best.pth --image test.jpg --text "pick up cup"

# Batch inference
python scripts/inference.py --checkpoint checkpoints/best.pth --data_dir test_data/ --output results.json

# Real-time inference (robot)
python scripts/inference.py --checkpoint checkpoints/best.pth --robot so100 --control_freq 20
```

### Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py --checkpoint checkpoints/best.pth --data configs/data/test.yaml

# Benchmark speed
python scripts/evaluate.py --checkpoint checkpoints/best.pth --benchmark --num_runs 100

# Evaluate with metrics
python scripts/evaluate.py --checkpoint checkpoints/best.pth --metrics success_rate,fps,latency
```

### Model Export

```bash
# Export to ONNX
python scripts/export.py --checkpoint checkpoints/best.pth --format onnx --output model.onnx

# Export to TorchScript
python scripts/export.py --checkpoint checkpoints/best.pth --format torchscript --output model.pt

# Export with optimization
python scripts/export.py --checkpoint checkpoints/best.pth --format onnx --optimize --fp16
```

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_vlm_implementations.py

# Run with coverage
pytest --cov=librobot tests/

# Run specific test
pytest tests/test_vlm_implementations.py::TestQwenVL::test_forward_pass

# Run with verbose output
pytest -v tests/
```

### Development

```bash
# Install in editable mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Run linter
flake8 librobot/
black librobot/
isort librobot/

# Type checking
mypy librobot/

# Generate documentation
cd docs && make html
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Symptoms:** `RuntimeError: CUDA out of memory`

**Solutions:**
```python
# Reduce batch size
batch_size = 16  # Instead of 32

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    outputs = model(inputs)

# Freeze VLM backbone
vla.freeze_backbone()

# Use LoRA/QLoRA
from librobot.models.vlm.adapters import LoRAAdapter
vlm = LoRAAdapter(vlm, rank=8)
```

#### 2. Slow Training

**Symptoms:** Training is taking too long

**Solutions:**
```python
# Use Flash Attention 2
# Automatically enabled if available

# Freeze VLM backbone
vla.freeze_backbone()

# Use smaller VLM
vlm = create_vlm("florence-2-base")  # 230M params

# Reduce precision
with torch.cuda.amp.autocast():
    outputs = model(inputs)

# Distributed training
torchrun --nproc_per_node=4 train.py
```

#### 3. Model Not Found

**Symptoms:** `KeyError: Component 'xxx' not found in registry`

**Solutions:**
```python
# List available models
from librobot.models.vlm import list_vlms
print(list_vlms())

# Check spelling and aliases
vlm = create_vlm("qwen2-vl-2b")  # Correct
vlm = create_vlm("qwen-vl")      # Alias works
vlm = create_vlm("qwen2vl")      # ❌ Wrong

# Import the module if it's a custom plugin
import my_plugin  # Registers components
vlm = create_vlm("my-vlm")
```

#### 4. Shape Mismatch

**Symptoms:** `RuntimeError: shape mismatch`

**Solutions:**
```python
# Check input shapes
print(images.shape)      # Should be [B, C, H, W]
print(actions.shape)     # Should be [B, action_dim]

# Ensure correct dimensions
if images.ndim == 3:
    images = images.unsqueeze(0)  # Add batch dim

# Check device
if images.device != model.device:
    images = images.to(model.device)
```

#### 5. Gradient Not Flowing

**Symptoms:** Loss not decreasing, gradients are None

**Solutions:**
```python
# Check requires_grad
for name, param in model.named_parameters():
    if not param.requires_grad:
        print(f"{name}: requires_grad=False")

# Unfreeze if needed
vla.unfreeze_backbone()

# Check for detach() calls
# Remove any .detach() in forward pass

# Verify loss has grad
assert outputs["loss"].requires_grad
```

#### 6. Import Error

**Symptoms:** `ModuleNotFoundError: No module named 'librobot'`

**Solutions:**
```bash
# Install package
pip install -e .

# Check PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/librobot_vla

# Check installation
python -c "import librobot; print(librobot.__version__)"
```

### Performance Tuning

#### Memory Optimization

```python
# 1. Gradient checkpointing
model.gradient_checkpointing_enable()

# 2. Mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# 3. Clear cache
torch.cuda.empty_cache()

# 4. Use smaller batch size
batch_size = 8

# 5. Gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch)["loss"] / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### Speed Optimization

```python
# 1. Use Flash Attention (automatic if available)

# 2. Compile model (PyTorch 2.0+)
model = torch.compile(model)

# 3. Use DataLoader efficiently
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,      # Parallel loading
    pin_memory=True,    # Faster GPU transfer
    prefetch_factor=2,  # Prefetch batches
)

# 4. Profile and optimize
with torch.profiler.profile() as prof:
    outputs = model(inputs)
print(prof.key_averages().table())
```

## FAQ

### General Questions

**Q: What is LibroBot VLA?**  
A: A comprehensive framework for building Vision-Language-Action models for robot learning, featuring 5 VLM families (11 variants) and 8 VLA framework architectures.

**Q: What's the difference between VLM and VLA?**  
A: VLMs are vision-language models that understand images and text. VLAs add action prediction for robot control.

**Q: Which framework should I use?**  
A: 
- **GR00T**: Multi-camera, stable training
- **π0**: Complex state spaces
- **Octo**: Multi-task learning
- **OpenVLA**: Language-guided tasks
- **RT-2**: Discrete action spaces
- **ACT**: Bi-manual manipulation
- **Helix**: Long-horizon planning

### Technical Questions

**Q: How much GPU memory do I need?**  
A: 
- Florence-2 base: 8GB+
- Qwen2-VL 2B: 16GB+
- Qwen2-VL 7B: 24GB+
- With LoRA: -50% memory

**Q: Can I use CPU only?**  
A: Yes, but training will be very slow. Inference is feasible for small models.

**Q: How do I add a custom VLM?**  
A: See [COMPONENT_GUIDE.md](./COMPONENT_GUIDE.md#adding-a-new-vlm) for step-by-step instructions.

**Q: How do I save/load checkpoints?**  
A:
```python
# Save
torch.save({
    'model': vla.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch,
}, 'checkpoint.pth')

# Load
checkpoint = torch.load('checkpoint.pth')
vla.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
```

**Q: How do I use multiple GPUs?**  
A:
```bash
# DataParallel (simple but slower)
vla = torch.nn.DataParallel(vla)

# DistributedDataParallel (recommended)
torchrun --nproc_per_node=4 train.py
```

**Q: Can I use pre-trained weights from HuggingFace?**  
A: Some models support this. Check model documentation.

### Workflow Questions

**Q: What's the typical workflow?**  
A:
1. Choose VLM and VLA framework
2. Prepare dataset
3. Train model
4. Evaluate performance
5. Deploy to robot

**Q: How long does training take?**  
A: Depends on dataset size and hardware:
- Small dataset (1K episodes): Hours
- Medium dataset (10K episodes): Days
- Large dataset (100K+ episodes): Weeks

**Q: How do I debug my model?**  
A:
1. Use dummy dataset first
2. Overfit on single batch
3. Check gradient flow
4. Visualize attention weights
5. Profile performance

**Q: How do I contribute?**  
A: See [COMPONENT_GUIDE.md](./COMPONENT_GUIDE.md#publishing-your-plugin) for contribution guidelines.

## Conclusion

This quick reference provides the essential information needed to work with LibroBot VLA. For more detailed information, see:

- [Architecture](./ARCHITECTURE.md) - System architecture
- [Project Structure](./PROJECT_STRUCTURE.md) - File organization
- [Design Principles](./DESIGN_PRINCIPLES.md) - Design patterns
- [Component Guide](./COMPONENT_GUIDE.md) - Adding components
- [API Contracts](./API_CONTRACTS.md) - Interface definitions
- [Roadmap](./ROADMAP.md) - Development plans
