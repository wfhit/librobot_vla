# VLA Framework Implementations

Complete, production-ready implementations of 8 major Vision-Language-Action (VLA) framework architectures for robot learning.

## 📚 Overview

This module provides fully-implemented VLA frameworks that combine vision, language, and action prediction for end-to-end robot learning. Each framework follows best practices from recent research and is ready for training and deployment.

## 🏗️ Available Frameworks

### 1. GR00TVLA - NVIDIA GR00T Style
**Architecture**: Frozen VLM + State Encoder + FiLM + Diffusion Policy

```python
from librobot.models.frameworks import GR00TVLA
from librobot.models.vlm import get_vlm

vlm = get_vlm('prismatic', pretrained=True)
model = GR00TVLA(
    vlm=vlm,
    action_dim=7,
    state_dim=14,
    hidden_dim=512,
    num_cameras=2,
    diffusion_steps=100,
    freeze_vlm=True,
)

# Training
outputs = model(images, text, proprioception, actions)
loss = outputs['loss']

# Inference
actions = model.predict_action(images, text, proprioception)
```

**Key Features**:
- Frozen VLM backbone for stability
- FiLM conditioning from VLM features
- DDPM diffusion for continuous actions
- Multi-camera support
- State encoding via MLP

---

### 2. Pi0VLA - Physical Intelligence π0 Style
**Architecture**: VLM + State Tokenization + Flow Matching

```python
from librobot.models.frameworks import Pi0VLA

model = Pi0VLA(
    vlm=vlm,
    action_dim=7,
    state_dim=14,
    hidden_dim=512,
    num_state_tokens=1024,
    flow_steps=50,
    fine_tune_vlm=True,
)

# State is tokenized via VQ-VAE
outputs = model(images, text, proprioception, actions)
loss = outputs['loss']  # Includes VQ loss
```

**Key Features**:
- State tokenization (VQ-VAE)
- States as first-class tokens
- Flow matching for actions
- Block-wise attention
- End-to-end trainable

---

### 3. OctoVLA - Berkeley Octo Style
**Architecture**: Unified Transformer + Task Conditioning + Multi-Task

```python
from librobot.models.frameworks import OctoVLA

model = OctoVLA(
    vision_encoder=vision_encoder,
    action_dim=7,
    state_dim=14,
    hidden_dim=512,
    num_tasks=100,
    history_length=5,
)

# Task-conditioned prediction
outputs = model(
    images=images,
    proprioception=proprioception,
    task_id=task_ids,
    history_images=history_imgs,
    actions=actions,
)
```

**Key Features**:
- Unified transformer for all modalities
- Task embeddings for multi-task learning
- History integration
- Flexible observation/action spaces
- Cross-dataset training

---

### 4. OpenVLA - Berkeley OpenVLA Style
**Architecture**: Fine-tuned VLM + Action Tokens + MLP OFT

```python
from librobot.models.frameworks import OpenVLA

model = OpenVLA(
    vlm=vlm,
    action_dim=7,
    hidden_dim=512,
    num_action_tokens=1,
    fine_tune_vlm=True,
    action_token_pattern='last',  # 'last', 'first', 'mean', 'special'
)

# Natural language instructions
outputs = model(
    images=images,
    text="Pick up the red block and place it in the box",
    proprioception=proprioception,
    actions=actions,
)
```

**Key Features**:
- End-to-end VLM fine-tuning
- Output-from-Tokens (OFT) MLP head
- Action token extraction
- Natural language interface
- Open-source VLM support

---

### 5. RT2VLA - Google RT-2 Style
**Architecture**: VLM + Discretized Actions + Autoregressive Prediction

```python
from librobot.models.frameworks import RT2VLA

model = RT2VLA(
    vlm=vlm,
    action_dim=7,
    num_bins=256,
    action_bounds=[(-1, 1)] * 7,  # Per-dimension bounds
    temperature=1.0,
)

# Actions predicted as discrete tokens
outputs = model(images, text, actions=actions)
loss = outputs['loss']

# Sample from distribution
actions = model.predict_action(images, text, temperature=0.8)
```

**Key Features**:
- 256-bin action discretization
- Token-based prediction
- Autoregressive decoding
- Classification loss
- Temperature sampling

---

### 6. ACTVLA - ALOHA ACT Style
**Architecture**: Transformer Encoder-Decoder + CVAE + Action Chunking

```python
from librobot.models.frameworks import ACTVLA

model = ACTVLA(
    vision_encoder=vision_encoder,
    action_dim=7,
    state_dim=14,
    chunk_size=10,
    latent_dim=32,
    kl_weight=0.0001,
)

# Predict action sequences
action_sequences = torch.randn(batch_size, chunk_size, action_dim)
outputs = model(images, proprioception=proprioception, actions=action_sequences)
loss = outputs['loss']  # CVAE loss = recon + KL

# Inference: Sample from prior
action_chunk = model.predict_action(images, proprioception)
# Returns [batch, chunk_size, action_dim]
```

**Key Features**:
- CVAE for multi-modal distributions
- Action chunking (predict sequences)
- Transformer encoder-decoder
- Temporal consistency
- Bi-manual robot support

---

### 7. HelixVLA - Figure AI Helix Style
**Architecture**: 3-Tier Hierarchical (High/Mid/Low)

```python
from librobot.models.frameworks import HelixVLA

model = HelixVLA(
    vlm=vlm,
    action_dim=7,
    state_dim=14,
    plan_dim=256,
    plan_horizon=10,
    freeze_vlm=True,
)

# Hierarchical prediction
outputs = model(images, text, proprioception, actions)
plan = outputs['plan']  # High-level plan
action_seq = outputs['action_sequence']  # Mid-level
action = outputs['actions']  # Low-level

# Can return full sequence
action, action_seq = model.predict_action(
    images, text, proprioception, return_sequence=True
)
```

**Key Features**:
- 3-tier hierarchy (VLM → Policy → Motor Control)
- Different time scales per tier
- Plan representation learning
- Temporal smoothing
- Hierarchical training

---

### 8. CustomVLA - Custom Template
**Architecture**: Flexible Component Composition

```python
from librobot.models.frameworks import CustomVLA
from librobot.models.action_heads import DDPMActionHead
from librobot.models.encoders import MLPStateEncoder

# Option 1: Direct composition
model = CustomVLA(
    action_dim=7,
    vlm=my_vlm,
    action_head=my_action_head,
    state_encoder=my_state_encoder,
    fusion_module=my_fusion,
    hidden_dim=512,
)

# Option 2: Subclass for complex architectures
class MyCustomVLA(CustomVLA):
    def __init__(self, ...):
        super().__init__(...)
        self.my_layer = nn.Linear(512, 512)
    
    def _process_features(self, vision, text, state):
        features = super()._process_features(vision, text, state)
        return self.my_layer(features)

model = MyCustomVLA(...)
```

**Key Features**:
- Mix-and-match components
- Easy experimentation
- Inheritance-based extension
- Flexible architecture
- Custom processing

---

## 🎯 Common Interface

All frameworks implement the `AbstractVLA` interface:

```python
# Training
outputs = model(
    images=images,          # [batch, C, H, W]
    text=text,             # Optional: list of strings
    proprioception=state,   # Optional: [batch, state_dim]
    actions=actions,        # [batch, action_dim] or [batch, chunk, action_dim]
)
loss = outputs['loss']
loss.backward()

# Inference
actions = model.predict_action(images, text, proprioception)

# Utilities
config = model.get_config()
model.freeze_backbone()
model.unfreeze_backbone()
num_params = model.get_num_parameters(trainable_only=True)
```

---

## 📊 Comparison Table

| Framework | Backbone | Action Space | Key Feature | Best For |
|-----------|----------|--------------|-------------|----------|
| GR00T | Frozen VLM | Continuous (Diffusion) | FiLM conditioning | Multi-camera, stable training |
| π0 | Fine-tuned VLM | Continuous (Flow) | State tokenization | Complex state spaces |
| Octo | Vision + Text | Continuous (MLP) | Multi-task learning | Cross-dataset training |
| OpenVLA | Fine-tuned VLM | Continuous (MLP) | Instruction following | Natural language tasks |
| RT-2 | Fine-tuned VLM | Discrete (256 bins) | Token prediction | Language conditioning |
| ACT | Vision only | Continuous (CVAE) | Action chunking | Bi-manual manipulation |
| Helix | Frozen VLM | Continuous (Hierarchical) | 3-tier hierarchy | Complex long-horizon tasks |
| Custom | Flexible | Flexible | Composable | Experimentation |

---

## 🚀 Quick Start

### Installation
```bash
pip install librobot-vla
```

### Basic Training Loop
```python
import torch
from librobot.models.frameworks import GR00TVLA
from librobot.models.vlm import get_vlm
from torch.utils.data import DataLoader

# Initialize model
vlm = get_vlm('prismatic', pretrained=True)
model = GR00TVLA(vlm=vlm, action_dim=7, state_dim=14)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        images = batch['images']
        text = batch['instructions']
        state = batch['proprioception']
        actions = batch['actions']
        
        outputs = model(images, text, state, actions)
        loss = outputs['loss']
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save model
model.save_pretrained('checkpoints/model.pt')
```

### Inference
```python
model.eval()
with torch.no_grad():
    predicted_actions = model.predict_action(
        images=camera_images,
        text="Pick up the cup",
        proprioception=robot_state,
    )
    
# Execute on robot
robot.execute(predicted_actions)
```

---

## 🔧 Advanced Usage

### Multi-Camera Support (GR00T)
```python
# Images: [batch, num_cameras, C, H, W]
multi_cam_images = torch.stack([cam1, cam2, cam3], dim=1)
model = GR00TVLA(vlm=vlm, action_dim=7, num_cameras=3)
outputs = model(multi_cam_images, text, state, actions)
```

### Task-Conditioned Training (Octo)
```python
# Train on multiple tasks
for batch in multi_task_dataloader:
    outputs = model(
        images=batch['images'],
        task_id=batch['task_ids'],  # [batch]
        proprioception=batch['state'],
        actions=batch['actions'],
    )
    loss = outputs['loss']
```

### Action Chunking (ACT)
```python
# Predict sequences of actions
model = ACTVLA(vision_encoder=encoder, action_dim=7, chunk_size=10)
action_chunk = model.predict_action(images, state)  # [batch, 10, 7]

# Execute actions sequentially
for t in range(chunk_size):
    robot.execute(action_chunk[0, t])
    time.sleep(dt)
```

### Hierarchical Control (Helix)
```python
# Get plan and actions at different levels
outputs = model(images, text, state)
high_level_plan = outputs['plan']          # Strategic goal
mid_level_seq = outputs['action_sequence'] # Action plan
low_level_action = outputs['actions']      # Immediate action

# Monitor execution
if not execution_successful:
    # Replan at high level
    new_plan = model.high_level_planning(images, text)
```

---

## 📖 References

1. **GR00T**: NVIDIA's GR00T Foundation Model for Humanoid Robots
2. **π0**: Physical Intelligence π0 - A Physical World Foundation Model
3. **Octo**: Berkeley's Open X-Embodiment Robotic Learning at Scale
4. **OpenVLA**: An Open-Source Vision-Language-Action Model
5. **RT-2**: Robotics Transformer 2 (Google DeepMind)
6. **ACT**: Action Chunking with Transformers (ALOHA)
7. **Helix**: Figure AI's Hierarchical Robot Learning
8. **Custom**: Research-friendly template

---

## 📝 Citation

If you use these implementations in your research, please cite:

```bibtex
@software{librobot_vla,
  title = {LibRobot VLA: Production-Ready Vision-Language-Action Frameworks},
  author = {LibRobot Contributors},
  year = {2024},
  url = {https://github.com/yourrepo/librobot_vla}
}
```

---

## 🤝 Contributing

We welcome contributions! Each framework is self-contained and follows the `AbstractVLA` interface. To add a new framework:

1. Create `your_framework.py` in `librobot/models/frameworks/`
2. Inherit from `AbstractVLA`
3. Implement required methods: `forward()`, `predict_action()`, `compute_loss()`, `get_config()`
4. Add tests
5. Update documentation

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 🐛 Known Issues & Troubleshooting

### Issue: Out of memory during training
**Solution**: Reduce batch size or use gradient accumulation
```python
# Enable gradient checkpointing for VLM
model.vlm.gradient_checkpointing_enable()
```

### Issue: Slow diffusion sampling (GR00T)
**Solution**: Reduce diffusion steps or use DDIM sampling
```python
model = GR00TVLA(..., diffusion_steps=10)  # Instead of 100
```

### Issue: NaN loss with RT-2
**Solution**: Check action bounds and clip gradients
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 💡 Tips & Best Practices

1. **Start with frozen VLM**: Freeze the VLM backbone initially, then fine-tune later
2. **Use mixed precision**: Enable AMP for faster training
3. **Monitor losses**: Track component losses separately (VQ loss, KL loss, etc.)
4. **Curriculum learning**: Start with simple tasks, gradually increase difficulty
5. **Ensemble predictions**: Average predictions from multiple inference runs for stability

---

For more examples and tutorials, see `examples/frameworks/`.
