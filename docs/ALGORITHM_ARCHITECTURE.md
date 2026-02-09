# Algorithm Architecture: Wheel Loader VLA

> **Vision-Language-Action Model for Autonomous Wheel Loader Operation**

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Perception System](#perception-system)
  - [Surround-View Cameras (OAK FPV)](#surround-view-cameras-oak-fpv)
  - [Depth Camera (ZED 2)](#depth-camera-zed-2)
- [Vision-Language Model (Qwen3-VL)](#vision-language-model-qwen3-vl)
- [Input Representation](#input-representation)
  - [Language Commands](#language-commands)
  - [Robot State](#robot-state)
- [Flow Matching Action Head (LingVLA-Inspired)](#flow-matching-action-head-lingvla-inspired)
- [Trajectory Generation](#trajectory-generation)
- [KV-Cache Optimization](#kv-cache-optimization)
- [End-to-End Pipeline](#end-to-end-pipeline)
- [References](#references)

---

## Overview

This document describes the algorithm architecture for applying Vision-Language-Action (VLA) models to autonomous wheel loader operation. The system combines:

- **Qwen3-VL** as the vision-language backbone for multimodal understanding
- **Multi-camera surround view** (4× OAK FPV) plus a **ZED 2 stereo camera** (depth + RGB) for comprehensive scene perception
- **Language commands** and **robot state** as additional inputs
- A **LingVLA-inspired flow matching action head** that takes token features from Qwen3 and robot state to generate smooth trajectories
- **Trajectory output**: 7 degrees of freedom for the wheel loader, predicted at 10 Hz with a 16-step horizon
- **KV-cache** to accelerate autoregressive inference

```
                          ┌──────────────────────────────────────────┐
                          │         Wheel Loader VLA Pipeline        │
                          └──────────────────────────────────────────┘

  ┌───────────────┐   ┌───────────┐   ┌──────────────┐
  │ 4× OAK FPV   │   │  ZED 2    │   │  Language     │
  │ (surround     │   │ (depth +  │   │  Command      │
  │  RGB views)   │   │  RGB)     │   │  (text)       │
  └──────┬────────┘   └─────┬─────┘   └──────┬────────┘
         │                  │                 │
         ▼                  ▼                 ▼
  ┌─────────────────────────────────────────────────────┐
  │              Qwen3-VL  (Vision-Language Model)      │
  │    ┌────────────────┐    ┌──────────────────┐       │
  │    │ Vision Encoder  │    │ Language Encoder │       │
  │    │ (ViT + 3D RoPE)│    │ (Causal LM)     │       │
  │    └───────┬────────┘    └────────┬─────────┘       │
  │            └──────────┬───────────┘                  │
  │                       ▼                              │
  │           [VLM Token Embeddings]                     │
  │               (with KV-Cache)                        │
  └───────────────────────┬─────────────────────────────┘
                          │
            ┌─────────────┼──────────────┐
            │             │              │
            ▼             ▼              ▼
   ┌──────────────┐ ┌───────────┐ ┌────────────────┐
   │ VLM Tokens   │ │ Robot     │ │ Noise Tokens   │
   │ (visual +    │ │ State     │ │ (action query) │
   │  language)   │ │ Tokens    │ │                │
   └──────┬───────┘ └─────┬─────┘ └──────┬─────────┘
          └───────────┬────┘──────────────┘
                      ▼
  ┌─────────────────────────────────────────────────────┐
  │   Flow Matching Action Head (LingVLA-Inspired)      │
  │                                                     │
  │   x(t) = (1 - t)·x₀ + t·x₁                        │
  │   v_θ(x(t), t, c) → velocity field                 │
  │                                                     │
  │   Condition c = [VLM tokens; Robot state tokens]    │
  └───────────────────────┬─────────────────────────────┘
                          ▼
  ┌─────────────────────────────────────────────────────┐
  │          Trajectory Output                          │
  │   7 DOF × 16 steps @ 10 Hz  (1.6 s horizon)        │
  │                                                     │
  │   [steering, throttle, brake, bucket_tilt,          │
  │    boom_lift, arm_curl, transmission]               │
  └─────────────────────────────────────────────────────┘
```

---

## System Architecture

The algorithm follows a three-stage pipeline that maps raw sensor data and language instructions to continuous wheel loader trajectories:

| Stage | Component | Role |
|-------|-----------|------|
| **1. Perception** | 4× OAK FPV + ZED 2 | Multi-view RGB images and depth map |
| **2. Understanding** | Qwen3-VL | Fuse vision and language into token embeddings |
| **3. Action** | Flow Matching Head | Generate 7-DOF trajectory from tokens + robot state |

### Data Flow

```
Cameras (5 views)  ──┐
                     ├──▶  Qwen3-VL  ──▶  Token Features  ──┐
Language Command   ──┘                                       ├──▶  Flow Matching  ──▶  Trajectory
                                                             │       Action Head        (7 DOF × 16)
Robot State  ──────────────▶  State Tokenizer  ──────────────┘
```

---

## Perception System

### Surround-View Cameras (OAK FPV)

Four **Luxonis OAK FPV** (Fixed-Focus PoE) cameras are mounted around the wheel loader to provide a 360° surround view:

| Camera | Mounting Position | Field of View | Purpose |
|--------|------------------|---------------|---------|
| Front  | Cab front center  | ~90° HFOV    | Forward obstacle detection, loading target |
| Rear   | Rear body         | ~90° HFOV    | Reversing safety, dump truck alignment |
| Left   | Left side mirror  | ~90° HFOV    | Side clearance, pedestrian detection |
| Right  | Right side mirror | ~90° HFOV    | Side clearance, traffic awareness |

- **Resolution**: 1280 × 800 (downsampled to 448 × 448 for VLM input)
- **Frame rate**: 30 FPS (captured), 10 FPS (processed by VLA)
- **Output**: 4 × RGB images per timestep

### Depth Camera (ZED 2)

A **Stereolabs ZED 2** stereo camera is mounted at the front of the cab to provide high-quality depth perception for close-range operations:

- **RGB output**: 1920 × 1080 (downsampled to 448 × 448 for VLM input)
- **Depth output**: Dense depth map up to 20 m range
- **Use case**: Precise bucket positioning, pile distance estimation, obstacle height measurement
- **Depth encoding**: Depth map is normalized and concatenated as an additional channel to the front RGB view, forming a 4-channel (RGBD) input to the vision encoder

> **Total visual input**: 5 camera views (4× OAK FPV RGB + 1× ZED 2 RGB-D) are processed by the Qwen3-VL vision encoder. The depth channel from ZED 2 provides geometric grounding that pure RGB lacks.

---

## Vision-Language Model (Qwen3-VL)

The VLM backbone is **Qwen3-VL** ([`librobot/models/vlm/qwen_vl.py`](../librobot/models/vlm/qwen_vl.py)), which processes multi-view images and language instructions into a unified token sequence.

### Why Qwen3-VL

- **Native multi-image support**: Handles multiple camera views in a single forward pass
- **3D Rotary Position Embedding (RoPE)**: Provides spatial-aware positional encoding for vision tokens across different camera views
- **High-resolution vision encoder**: ViT-based encoder with patch merging supports varied input resolutions
- **Strong instruction following**: Causal language model backbone understands complex operational commands

### Configuration

The system uses the `qwen3-vl-4b` variant (registered in the VLM registry):

```python
from librobot.models.vlm import create_vlm

vlm = create_vlm("qwen3-vl-4b", pretrained=True)
# Hidden dim: 2048
# Vision encoder: ViT with 3D RoPE
# Language model: 4B parameter causal LM
```

### Image Tokenization

Each camera view is processed by the Qwen3 vision encoder:

```
Per camera view (448 × 448 RGB):
  → Patch embedding (14 × 14 patches) → 1024 vision tokens
  → 3D RoPE applied (spatial + temporal dimensions)

5 views × 1024 tokens = 5120 vision tokens total
```

The vision tokens from all camera views are concatenated into a single sequence alongside the language tokens and passed through the Qwen3 language model layers, producing a fused multimodal token representation.

---

## Input Representation

### Language Commands

Natural language instructions describe the desired wheel loader operation:

| Example Command | Operation |
|----------------|-----------|
| `"Load dirt from the pile ahead"` | Approach pile, scoop material |
| `"Dump the bucket into the truck"` | Navigate to truck, raise boom, tilt bucket |
| `"Drive forward 10 meters"` | Straight-line traversal |
| `"Back up and turn left"` | Reverse with steering |

Language commands are tokenized by the Qwen3 tokenizer and processed jointly with vision tokens through the language model layers.

### Robot State

The robot proprioceptive state provides the current configuration of the wheel loader. This state vector is tokenized and injected as additional tokens into the sequence:

| State Dimension | Description | Unit |
|----------------|-------------|------|
| `steering_angle` | Current steering angle | radians |
| `vehicle_speed` | Forward/backward speed | m/s |
| `bucket_angle` | Bucket tilt angle | radians |
| `boom_height` | Boom vertical position | meters |
| `hydraulic_pressure` | Hydraulic system pressure | PSI |
| `engine_rpm` | Engine revolutions per minute | RPM |
| `heading` | Vehicle heading from IMU | radians |

The robot state is encoded into tokens through a state tokenizer (see [`librobot/models/encoders/state/`](../librobot/models/encoders/state/)):

```python
# Robot state → discrete tokens → embeddings
state_vector = [steering, speed, bucket, boom, pressure, rpm, heading]
state_tokens = state_tokenizer(state_vector)  # shape: [num_state_tokens, hidden_dim]
```

---

## Flow Matching Action Head (LingVLA-Inspired)

The action prediction head is inspired by **LingVLA**, which uses a flow matching formulation for trajectory generation. The implementation builds on the existing flow matching infrastructure in [`librobot/models/action_heads/flow_matching/`](../librobot/models/action_heads/flow_matching/).

### Flow Matching Formulation

Flow matching learns a velocity field `v_θ` that transports samples from a noise distribution to the target action distribution:

```
Interpolation:   x(t) = (1 - t) · x₀ + t · x₁     where t ∈ [0, 1]
                 x₀ ~ N(0, I)                        (noise)
                 x₁ ~ p_data(actions)                 (ground truth trajectory)

Velocity field:  v_θ(x(t), t, c)                     (predicted by network)
Target:          v* = x₁ - x₀                         (ground truth velocity)

Loss:            L = ||v_θ(x(t), t, c) - v*||²
```

### Conditioning

The flow matching head is conditioned on both VLM token features and robot state tokens, following the LingVLA approach:

```
Condition tokens  c = [VLM_tokens ; Robot_state_tokens]

VLM_tokens:         Pooled or selected output tokens from Qwen3-VL
                    (captures visual scene + language intent)

Robot_state_tokens: Encoded proprioceptive state
                    (captures current machine configuration)
```

The condition is injected into the velocity network via cross-attention or adaptive layer normalization, enabling the flow to be guided by both the visual-language understanding and the current robot state.

### Inference (Trajectory Sampling)

At inference time, the trajectory is generated by integrating the learned velocity field from noise:

```python
# Euler integration over K denoising steps
x = torch.randn(batch, 16, 7)       # noise initialization (16 steps × 7 DOF)
dt = 1.0 / num_flow_steps

for i in range(num_flow_steps):
    t = i * dt
    v = velocity_net(x, t, condition)  # predict velocity
    x = x + v * dt                     # Euler step

trajectory = x  # shape: [batch, 16, 7]
```

A small number of flow steps (e.g., 4–8) can be used at inference time for fast generation, since rectified flow paths are approximately straight.

---

## Trajectory Generation

The model outputs a **multi-step trajectory** that controls the wheel loader over a short planning horizon.

### Output Specification

| Parameter | Value | Description |
|-----------|-------|-------------|
| **DOF** | 7 | Degrees of freedom per timestep |
| **Frequency** | 10 Hz | Control frequency (one action every 100 ms) |
| **Horizon** | 16 steps | Planning horizon (1.6 seconds into the future) |
| **Output shape** | `[batch, 16, 7]` | Batch of trajectory chunks |

### 7 Degrees of Freedom

The 7-DOF action vector for the wheel loader at each timestep:

| Index | DOF | Range | Description |
|-------|-----|-------|-------------|
| 0 | `steering` | [-1.0, 1.0] | Steering angle (normalized) |
| 1 | `throttle` | [0.0, 1.0] | Forward drive power |
| 2 | `brake` | [0.0, 1.0] | Braking pressure |
| 3 | `bucket_tilt` | [-1.0, 1.0] | Bucket angle (dump/scoop) |
| 4 | `boom_lift` | [-1.0, 1.0] | Boom vertical position (lower/raise) |
| 5 | `arm_curl` | [-1.0, 1.0] | Arm curl for precise bucket control |
| 6 | `transmission` | [-1.0, 1.0] | Gear selection (reverse/neutral/forward) |

### Action Chunking

Following the action chunking approach, the model predicts 16 future actions at once rather than a single next action. At each control step (10 Hz):

1. The model predicts a 16-step trajectory `[a_0, a_1, ..., a_15]`
2. Only the first `k` actions (e.g., `k=4`) are executed on the robot
3. After executing `k` steps, the model re-plans with updated observations

This provides temporal consistency while allowing the model to adapt to changing conditions.

```
Time ─────────────────────────────────────────────▶

t=0:  Predict [a₀  a₁  a₂  a₃  a₄  ... a₁₅]
      Execute  ▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░

t=4:  Predict [a₀' a₁' a₂' a₃' a₄' ... a₁₅']
      Execute       ▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░

t=8:  Predict [a₀''a₁''a₂''a₃''a₄''... a₁₅'']
      Execute            ▓▓▓▓░░░░░░░░░░░░░░░░

▓ = executed actions    ░ = predicted but not executed
```

---

## KV-Cache Optimization

To achieve real-time 10 Hz control, the system uses **KV-cache** to avoid redundant computation in the Qwen3-VL backbone during inference. The implementation leverages the existing KV-cache infrastructure in [`librobot/inference/kv_cache.py`](../librobot/inference/kv_cache.py) and [`librobot/models/vlm/utils/kv_cache.py`](../librobot/models/vlm/utils/kv_cache.py).

### Why KV-Cache is Critical

Without KV-cache, each inference step requires a full forward pass through the Qwen3-VL model over all vision and language tokens (~5000+ tokens). With KV-cache:

| Metric | Without KV-Cache | With KV-Cache |
|--------|-----------------|---------------|
| Tokens processed per step | ~5120 (all) | ~10–50 (new only) |
| VLM forward pass time | ~200 ms | ~15 ms |
| Achievable control rate | ~5 Hz | **>10 Hz** ✅ |

### How It Works

```
Step 1 (Initial / Scene Change):
  Full forward pass through Qwen3-VL
  ├─ Process all 5120 vision tokens + language tokens
  ├─ Store Key-Value pairs for each transformer layer
  └─ Cache: { layer_i: (K_i, V_i) for i in range(num_layers) }

Step 2..N (Incremental):
  Only process new/changed tokens
  ├─ Updated robot state tokens (7 state tokens)
  ├─ Reuse cached K,V for vision + language tokens
  └─ Attend over full cached sequence efficiently
```

### Cache Invalidation Strategy

| Event | Action |
|-------|--------|
| New control step (same scene) | Reuse cache, only update robot state tokens |
| Camera image update (periodic) | Partially invalidate vision token cache, recompute affected views |
| New language command | Invalidate language token cache, recompute |
| Scene change (e.g., new task) | Full cache reset and recomputation |

### Integration with Action Chunking

KV-cache works synergistically with action chunking. Since the model predicts 16 steps at once and only re-plans every `k` steps, the VLM forward pass is needed only every `k/10` seconds (e.g., every 0.4 s if `k=4`), further reducing computational load.

```
10 Hz control loop:
  t=0.0s  [VLM forward + cache]  [Flow matching] → execute a₀
  t=0.1s  [cache hit]            [---]            → execute a₁
  t=0.2s  [cache hit]            [---]            → execute a₂
  t=0.3s  [cache hit]            [---]            → execute a₃
  t=0.4s  [VLM forward + cache]  [Flow matching] → execute a₀'
  ...
```

---

## End-to-End Pipeline

### Training

```python
from librobot.models.vlm import create_vlm
from librobot.models.frameworks import create_vla

# 1. Create Qwen3-VL backbone
vlm = create_vlm("qwen3-vl-4b", pretrained=True)

# 2. Create VLA with flow matching action head
vla = create_vla(
    "pi0",                      # Flow matching framework (π0-style)
    vlm=vlm,
    action_dim=7 * 16,          # 7 DOF × 16 steps (flattened trajectory)
    state_dim=7,                # Robot state dimension
    hidden_dim=512,
    flow_steps=50,              # Training flow steps
    freeze_vlm=True,            # Freeze VLM backbone
)

# 3. Training loop
for batch in dataloader:
    outputs = vla(
        images=batch["images"],         # [B, 5, 3, 448, 448]  (5 camera views)
        text=batch["commands"],         # List of language commands
        proprio=batch["robot_state"],   # [B, 7]  robot state
        actions=batch["trajectories"],  # [B, 16, 7]  ground truth trajectories
    )
    loss = outputs["loss"]
    loss.backward()
    optimizer.step()
```

### Inference

```python
from librobot.inference.kv_cache import KVCache

# Setup KV-cache for fast inference
kv_cache = KVCache(
    max_length=8192,
    num_layers=vlm.num_layers,
    device="cuda",
)

# Real-time control loop at 10 Hz
while operating:
    # Gather observations
    images = camera_system.get_images()         # 5 camera views
    robot_state = robot.get_state()             # 7-dim state vector
    command = current_language_command           # text instruction

    # Predict trajectory (uses KV-cache internally)
    trajectory = vla.predict_action(
        images=images,
        text=command,
        proprio=robot_state,
    )
    # trajectory shape: [16, 7]

    # Execute first k actions
    for step in range(k):
        robot.execute_action(trajectory[step])  # Send 7-DOF command
        wait_for_next_timestep()                # 100 ms (10 Hz)
```

---

## References

- **Qwen3-VL**: [Qwen2.5-VL Technical Report (Alibaba)](https://arxiv.org/abs/2502.13923) — Multi-image vision-language model with 3D RoPE
- **LingVLA**: [A VLA Model with Linguistic Flow Matching Action Head](https://arxiv.org/abs/2506.18711) — Flow matching for robot action generation with language conditioning
- **Flow Matching**: [Flow Matching for Generative Modeling (Lipman et al.)](https://arxiv.org/abs/2210.02747) — Foundation of the flow matching formulation
- **π0**: [π0: A Vision-Language-Action Flow Model for General Robot Control](https://www.physicalintelligence.company/blog/pi0) — Flow matching VLA architecture
- **Action Chunking**: [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (Zhao et al.)](https://arxiv.org/abs/2304.13705) — Action chunking with transformers
- **OAK FPV Cameras**: [Luxonis OAK Series](https://docs.luxonis.com/) — Edge AI cameras for computer vision
- **ZED 2**: [Stereolabs ZED 2](https://www.stereolabs.com/products/zed-2) — Stereo depth camera

### Related LibroBot Components

| Component | Path | Relevance |
|-----------|------|-----------|
| Qwen VLM | [`librobot/models/vlm/qwen_vl.py`](../librobot/models/vlm/qwen_vl.py) | Qwen3-VL backbone implementation |
| Flow Matching Head | [`librobot/models/action_heads/flow_matching/`](../librobot/models/action_heads/flow_matching/) | Flow matching action head |
| π0 Framework | [`librobot/models/frameworks/pi0_style.py`](../librobot/models/frameworks/pi0_style.py) | Flow matching VLA framework |
| KV-Cache | [`librobot/inference/kv_cache.py`](../librobot/inference/kv_cache.py) | Inference KV-cache implementation |
| VLM KV-Cache | [`librobot/models/vlm/utils/kv_cache.py`](../librobot/models/vlm/utils/kv_cache.py) | VLM-level KV-cache utilities |
| Wheel Loader | [`librobot/robots/wheel_loaders/`](../librobot/robots/wheel_loaders/) | Wheel loader robot interface |
| State Encoder | [`librobot/models/encoders/state/`](../librobot/models/encoders/state/) | Robot state tokenization |

---

*For more information on the overall LibroBot VLA architecture, see [Architecture Overview](architecture.md) and [Design Documentation](design/ARCHITECTURE.md).*
