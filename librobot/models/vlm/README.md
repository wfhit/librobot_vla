# VLM (Vision-Language Model) Implementations

This directory contains complete, production-ready implementations of state-of-the-art Vision-Language Models for the LibroBot VLA framework.

## Available Models

### Priority 0 (P0) - CRITICAL

#### Qwen-VL (Qwen2-VL & Qwen3-VL)
**File:** `qwen_vl.py`

Complete implementation of Qwen Vision-Language models with support for both Qwen2-VL and Qwen3-VL variants.

**Variants:**
- `qwen2-vl-2b` (2B parameters, default)
- `qwen2-vl-7b` (7B parameters)
- `qwen3-vl-4b` (4B parameters)
- `qwen3-vl-7b` (7B parameters)

**Features:**
- ✅ Dynamic resolution support (up to 4K)
- ✅ Patch-based vision encoding with 3D spatial-temporal patches
- ✅ 3D Rotary Position Embeddings (RoPE) for vision tokens
- ✅ Grouped-query attention in language model
- ✅ Support for both pretraining and instruction-tuned versions
- ✅ Gradient checkpointing for memory efficiency
- ✅ LoRA/QLoRA adapters for efficient fine-tuning
- ✅ KV cache for efficient autoregressive generation
- ✅ Flash Attention 2 support (optional)
- ✅ HuggingFace model hub integration

**Architecture:**
- Vision: Custom vision encoder with 3D patch embedding and window attention
- Language: Qwen2/Qwen3 decoder with GQA (Grouped Query Attention)
- Fusion: Linear projection from vision to language space

**Usage:**
```python
from librobot.models.vlm import create_vlm

# Create model
model = create_vlm('qwen2-vl-2b', pretrained='Qwen/Qwen2-VL-2B-Instruct')

# Encode images
image_features = model.encode_image(images)  # [B, N, D]

# Forward pass
outputs = model(images=images, input_ids=input_ids, labels=labels)

# Generate text
generated = model.generate(images=images, input_ids=prompts, max_new_tokens=100)
```

---

### Priority 1 (P1) - IMPORTANT

#### Florence-2
**File:** `florence.py`

Microsoft's Florence-2 unified vision-language model for multi-task learning.

**Variants:**
- `florence-2-base` (230M parameters, default)
- `florence-2-large` (770M parameters)

**Features:**
- ✅ Unified architecture for multiple vision tasks
- ✅ DaViT (Dual Attention Vision Transformer) backbone
- ✅ Window-based attention for efficiency
- ✅ Task prompt support for multi-task learning
- ✅ OCR and spatial understanding capabilities
- ✅ Dynamic resolution support
- ✅ Relative position bias in attention
- ✅ HuggingFace integration

**Architecture:**
- Vision: DaViT with hierarchical window attention
- Language: Transformer decoder with cross-attention
- Tasks: Caption, detailed caption, OCR, grounding, detection, segmentation

**Usage:**
```python
from librobot.models.vlm import create_vlm

# Create model
model = create_vlm('florence-2-base', pretrained='microsoft/Florence-2-base')

# Task-specific generation
outputs = model.generate(
    images=images,
    input_ids=prompts,
    task='caption',  # or 'ocr', 'detection', etc.
    max_new_tokens=100
)
```

---

#### PaliGemma
**File:** `paligemma.py`

Google's PaliGemma model combining SigLIP vision and Gemma language model.

**Variants:**
- `paligemma-3b` (3B parameters)

**Features:**
- ✅ SigLIP vision encoder (high-quality image understanding)
- ✅ Gemma language model with grouped-query attention
- ✅ Full image-text interleaving support
- ✅ Transfer learning from PaLI architecture
- ✅ Simple but effective projection layer
- ✅ Efficient inference with GQA
- ✅ Flash Attention 2 support
- ✅ HuggingFace integration

**Architecture:**
- Vision: SigLIP (Sigmoid Loss for Language-Image Pre-training)
- Language: Gemma decoder with GeGLU activation
- Fusion: Linear projection

**Usage:**
```python
from librobot.models.vlm import create_vlm

# Create model
model = create_vlm('paligemma-3b', pretrained='google/paligemma-3b-pt-224')

# Process image-text pairs
outputs = model(images=images, input_ids=input_ids, labels=labels)
```

---

### Priority 2 (P2) - NICE TO HAVE

#### InternVL2
**File:** `internvl.py`

InternVL2 with InternViT vision encoder and InternLM2 language model.

**Variants:**
- `internvl2-2b` (2B parameters, default)
- `internvl2-8b` (8B parameters)

**Features:**
- ✅ InternViT vision encoder with high-resolution support
- ✅ Pixel shuffle for efficient high-res processing (up to 4K)
- ✅ Dynamic resolution with interpolated position embeddings
- ✅ InternLM2 language model with SwiGLU
- ✅ Strong multilingual capabilities
- ✅ Vision projector with intermediate activation
- ✅ Flash Attention 2 support
- ✅ HuggingFace integration

**Architecture:**
- Vision: InternViT with learnable positional embeddings
- Language: InternLM2 decoder with grouped-query attention
- Fusion: Two-layer MLP projection

**Usage:**
```python
from librobot.models.vlm import create_vlm

# Create model
model = create_vlm('internvl2-2b', pretrained='OpenGVLab/InternVL2-2B')

# High-resolution image processing
outputs = model(
    images=high_res_images,  # Up to 4K resolution
    input_ids=input_ids,
    labels=labels
)
```

---

#### LLaVA (v1.5)
**File:** `llava.py`

LLaVA (Large Language and Vision Assistant) - simple and effective VLM.

**Variants:**
- `llava-v1.5-7b` (7B parameters, default)
- `llava-v1.5-13b` (13B parameters)

**Features:**
- ✅ CLIP vision encoder for robust image understanding
- ✅ LLaMA/Vicuna language model
- ✅ Simple MLP projection layer (2-layer with GELU)
- ✅ Instruction tuning for chat and reasoning
- ✅ Image token integration into text sequence
- ✅ Efficient architecture
- ✅ Flash Attention 2 support
- ✅ HuggingFace integration

**Architecture:**
- Vision: CLIP ViT-L/14
- Language: LLaMA-2 or Vicuna decoder
- Fusion: Two-layer MLP with GELU activation

**Usage:**
```python
from librobot.models.vlm import create_vlm

# Create model
model = create_vlm('llava-v1.5-7b', pretrained='liuhaotian/llava-v1.5-7b')

# Chat-style interaction
outputs = model.generate(
    images=images,
    input_ids=chat_prompt,
    max_new_tokens=512,
    temperature=0.7
)
```

---

## Common Features

All VLM implementations include:

### Core Functionality
- ✅ Image encoding (single or multi-view)
- ✅ Text encoding with tokenizer
- ✅ Image-text fusion
- ✅ Autoregressive generation
- ✅ Feature extraction for downstream VLA tasks

### Optimization
- ✅ Gradient checkpointing support
- ✅ Parameter freezing (vision/language)
- ✅ Mixed precision (FP16/BF16)
- ✅ Flash Attention 2 (optional, when available)
- ✅ Memory-efficient attention mechanisms

### Training
- ✅ LoRA adapter support (Qwen-VL)
- ✅ QLoRA with quantization (Qwen-VL)
- ✅ Flexible loss computation
- ✅ Label masking for vision tokens

### Integration
- ✅ AbstractVLM interface compliance
- ✅ Registry registration
- ✅ Config-driven initialization
- ✅ HuggingFace model hub loading
- ✅ Checkpoint saving/loading

### Code Quality
- ✅ Full type hints
- ✅ Comprehensive docstrings
- ✅ Error handling
- ✅ Device management
- ✅ Batch processing support

---

## Usage Examples

### Basic Usage

```python
from librobot.models.vlm import create_vlm, list_vlms
import torch

# List available models
print("Available VLMs:", list_vlms())

# Create a VLM
config = {
    'variant': 'qwen2-vl-2b',
    'use_flash_attn': True,
    'use_lora': True,
    'lora_rank': 16,
}
model = create_vlm('qwen2-vl-2b', config=config, pretrained='Qwen/Qwen2-VL-2B')

# Freeze vision encoder
for param in model.vision_encoder.parameters():
    param.requires_grad = False

# Prepare inputs
images = torch.randn(2, 3, 224, 224)  # Batch of 2 images
input_ids = torch.randint(0, 1000, (2, 50))  # Text tokens
labels = torch.randint(0, 1000, (2, 50))  # Target tokens

# Forward pass
outputs = model(images=images, input_ids=input_ids, labels=labels)
print(f"Loss: {outputs['loss'].item()}")

# Generation
generated_ids = model.generate(
    images=images,
    input_ids=input_ids[:, :10],  # Prompt
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.95,
)
```

### Integration with VLA Framework

```python
from librobot.models.vlm import create_vlm
from librobot.models.frameworks.rt2_style import RT2StyleVLA

# Create VLM backend
vlm = create_vlm('florence-2-base', pretrained='microsoft/Florence-2-base')

# Create VLA with VLM
vla = RT2StyleVLA(
    vlm_backbone=vlm,
    action_dim=7,
    freeze_vlm=True,  # Freeze VLM during VLA training
)

# Train VLA
for batch in dataloader:
    images, instructions, actions = batch
    outputs = vla(images=images, text=instructions, actions=actions)
    loss = outputs['loss']
    loss.backward()
```

### Custom Configuration

```python
from librobot.models.vlm import get_vlm
from librobot.models.vlm.qwen_vl import QwenVLConfig

# Create custom config
config = QwenVLConfig(
    variant='qwen2-vl-7b',
    vision_num_layers=32,
    num_hidden_layers=32,
    use_flash_attn=True,
    use_lora=True,
    lora_rank=32,
    lora_alpha=64,
)

# Create model with custom config
VLMClass = get_vlm('qwen2-vl-7b')
model = VLMClass(config=config, freeze_vision=True)
```

---

## Model Comparison

| Model | Size | Vision Encoder | Language Model | Special Features |
|-------|------|----------------|----------------|------------------|
| **Qwen2-VL** | 2B-7B | Custom 3D patches | Qwen2 | 3D RoPE, dynamic resolution |
| **Qwen3-VL** | 4B-7B | Custom 3D patches | Qwen3 | Latest architecture |
| **Florence-2** | 230M-770M | DaViT | Transformer | Multi-task, OCR |
| **PaliGemma** | 3B | SigLIP | Gemma | Simple, effective |
| **InternVL2** | 2B-8B | InternViT | InternLM2 | High-res (4K), multilingual |
| **LLaVA-1.5** | 7B-13B | CLIP | LLaMA/Vicuna | Instruction tuning |

---

## Performance Tips

### Memory Optimization
- Enable gradient checkpointing for large models
- Use Flash Attention 2 when available
- Freeze vision/language components if fine-tuning
- Use LoRA/QLoRA for parameter-efficient fine-tuning

### Inference Optimization
- Use KV cache for generation (Qwen-VL)
- Batch images when possible
- Use lower precision (FP16/BF16)
- Consider quantization for deployment

### Training Tips
- Start with frozen VLM, fine-tune action head first
- Gradually unfreeze layers (layer-wise fine-tuning)
- Use task-specific prompts (Florence-2)
- Monitor attention patterns for debugging

---

## Architecture Details

Each VLM follows this general structure:

```
Image Input → Vision Encoder → Vision Projection → 
                                ↓
Text Input → Embedding → Fusion → Language Model → LM Head → Output
```

Specific differences:
- **Qwen-VL**: 3D vision patches, vision tokens interleaved with text
- **Florence-2**: Cross-attention decoder, task embeddings
- **PaliGemma**: Prepend vision tokens to text sequence
- **InternVL2**: Pixel shuffle for efficiency, two-layer projection
- **LLaVA**: Insert vision tokens at special positions in text

---

## Testing

All models have been tested for:
- ✅ Syntax correctness
- ✅ Import resolution
- ✅ Interface compliance
- ✅ Forward pass shape consistency
- ✅ Gradient flow (when training enabled)

---

## Citation

If you use these implementations, please cite the original papers:

**Qwen-VL:**
```bibtex
@article{qwen2vl,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Qwen Team},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024}
}
```

**Florence-2:**
```bibtex
@article{florence2,
  title={Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks},
  author={Bin Xiao and Haiping Wu and Weijian Xu and Xiyang Dai and Houdong Hu and Yumao Lu and Michael Zeng and Ce Liu and Lu Yuan},
  journal={arXiv preprint arXiv:2311.06242},
  year={2023}
}
```

**PaliGemma:**
```bibtex
@article{paligemma,
  title={PaliGemma: A versatile 3B VLM for transfer},
  author={Google DeepMind},
  year={2024}
}
```

**InternVL2:**
```bibtex
@article{internvl2,
  title={InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks},
  author={Chen, Zhe and Wu, Jiannan and Wang, Wenhai and Su, Weijie and Chen, Guo and Xing, Sen and Zhong, Muyan and Zhang, Qinglong and Zhu, Xizhou and Lu, Lewei and Li, Bin and Luo, Ping and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  journal={arXiv preprint arXiv:2404.16821},
  year={2024}
}
```

**LLaVA:**
```bibtex
@article{llava1.5,
  title={Improved Baselines with Visual Instruction Tuning},
  author={Liu, Haotian and Li, Chunyuan and Li, Yuheng and Lee, Yong Jae},
  journal={arXiv preprint arXiv:2310.03744},
  year={2023}
}
```

---

## Contributing

To add a new VLM:
1. Implement the model in a new file following the existing patterns
2. Inherit from `AbstractVLM` and implement all required methods
3. Register the model using `@register_vlm` decorator
4. Add comprehensive docstrings and type hints
5. Import in `__init__.py`
6. Update this README

---

## License

All implementations are provided under the MIT license, consistent with the LibroBot framework.
