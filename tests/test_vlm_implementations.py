"""Unit tests for VLM implementations."""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any

from librobot.models.vlm import (
    AbstractVLM,
    list_vlms,
    get_vlm,
    create_vlm,
)


class TestVLMRegistry:
    """Test VLM registry functionality."""

    def test_list_vlms(self):
        """Test listing registered VLMs."""
        vlms = list_vlms()
        assert isinstance(vlms, list)
        assert len(vlms) > 0

        # Check expected models are registered
        expected_models = [
            "qwen2-vl-2b",
            "qwen2-vl-7b",
            "qwen3-vl-4b",
            "qwen3-vl-7b",
            "florence-2-base",
            "florence-2-large",
            "paligemma-3b",
            "internvl2-2b",
            "internvl2-8b",
            "llava-v1.5-7b",
            "llava-v1.5-13b",
        ]

        for model in expected_models:
            assert model in vlms, f"Model {model} not registered"

    def test_get_vlm_class(self):
        """Test getting VLM class from registry."""
        vlm_class = get_vlm("qwen2-vl-2b")
        assert vlm_class is not None
        assert issubclass(vlm_class, AbstractVLM)

    def test_get_vlm_with_alias(self):
        """Test getting VLM with alias."""
        vlm_class_1 = get_vlm("qwen2-vl-2b")
        vlm_class_2 = get_vlm("qwen2-vl")  # alias
        assert vlm_class_1 == vlm_class_2


class TestVLMInterface:
    """Test VLM interface compliance."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "qwen2-vl-2b",
            "florence-2-base",
            "paligemma-3b",
            "internvl2-2b",
            "llava-v1.5-7b",
        ],
    )
    def test_vlm_creation(self, model_name):
        """Test VLM creation without pretrained weights."""
        config = {
            "variant": model_name,
            "use_flash_attn": False,  # Disable flash attention for testing
        }

        vlm = create_vlm(model_name, config=config)
        assert isinstance(vlm, AbstractVLM)
        assert isinstance(vlm, nn.Module)

    @pytest.mark.parametrize(
        "model_name",
        [
            "qwen2-vl-2b",
            "florence-2-base",
            "paligemma-3b",
            "internvl2-2b",
            "llava-v1.5-7b",
        ],
    )
    def test_vlm_has_required_methods(self, model_name):
        """Test VLM has all required methods."""
        config = {"use_flash_attn": False}
        vlm = create_vlm(model_name, config=config)

        # Check required methods exist
        assert hasattr(vlm, "forward")
        assert hasattr(vlm, "encode_image")
        assert hasattr(vlm, "encode_text")
        assert hasattr(vlm, "get_embedding_dim")
        assert hasattr(vlm, "config")
        assert callable(vlm.forward)
        assert callable(vlm.encode_image)
        assert callable(vlm.encode_text)
        assert callable(vlm.get_embedding_dim)

    @pytest.mark.parametrize(
        "model_name",
        [
            "qwen2-vl-2b",
            "florence-2-base",
            "paligemma-3b",
            "internvl2-2b",
            "llava-v1.5-7b",
        ],
    )
    def test_vlm_get_embedding_dim(self, model_name):
        """Test get_embedding_dim returns valid integer."""
        config = {"use_flash_attn": False}
        vlm = create_vlm(model_name, config=config)

        dim = vlm.get_embedding_dim()
        assert isinstance(dim, int)
        assert dim > 0

    @pytest.mark.parametrize(
        "model_name",
        [
            "qwen2-vl-2b",
            "florence-2-base",
            "paligemma-3b",
            "internvl2-2b",
            "llava-v1.5-7b",
        ],
    )
    def test_vlm_config_property(self, model_name):
        """Test config property returns dictionary."""
        config = {"use_flash_attn": False}
        vlm = create_vlm(model_name, config=config)

        cfg = vlm.config
        assert isinstance(cfg, dict)
        assert "model_type" in cfg
        assert "variant" in cfg


class TestVLMForward:
    """Test VLM forward pass."""

    @pytest.mark.parametrize(
        "model_name,img_size",
        [
            ("qwen2-vl-2b", 224),
            ("florence-2-base", 224),
            ("paligemma-3b", 224),
            ("internvl2-2b", 224),
            ("llava-v1.5-7b", 336),
        ],
    )
    def test_encode_image(self, model_name, img_size):
        """Test image encoding."""
        config = {"use_flash_attn": False}
        vlm = create_vlm(model_name, config=config)
        vlm.eval()

        # Create dummy image
        batch_size = 2
        images = torch.randn(batch_size, 3, img_size, img_size)

        with torch.no_grad():
            features = vlm.encode_image(images)

        # Check output shape
        assert features.ndim == 3  # [B, N, D]
        assert features.shape[0] == batch_size
        assert features.shape[2] == vlm.get_embedding_dim()

    @pytest.mark.parametrize(
        "model_name",
        [
            "qwen2-vl-2b",
            "florence-2-base",
            "paligemma-3b",
            "internvl2-2b",
            "llava-v1.5-7b",
        ],
    )
    def test_forward_with_images(self, model_name):
        """Test forward pass with images."""
        config = {"use_flash_attn": False}
        vlm = create_vlm(model_name, config=config)
        vlm.eval()

        # Create dummy inputs
        batch_size = 2
        seq_len = 10
        img_size = 224 if model_name != "llava-v1.5-7b" else 336

        images = torch.randn(batch_size, 3, img_size, img_size)
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))

        with torch.no_grad():
            outputs = vlm(images=images, input_ids=input_ids)

        # Check outputs
        assert isinstance(outputs, dict)
        assert "embeddings" in outputs
        assert "logits" in outputs

        # Check shapes
        embeddings = outputs["embeddings"]
        logits = outputs["logits"]

        assert embeddings.ndim == 3  # [B, L, D]
        assert logits.ndim == 3  # [B, L, V]
        assert embeddings.shape[0] == batch_size
        assert logits.shape[0] == batch_size

    @pytest.mark.parametrize(
        "model_name",
        [
            "qwen2-vl-2b",
            "florence-2-base",
            "paligemma-3b",
            "internvl2-2b",
            "llava-v1.5-7b",
        ],
    )
    def test_forward_with_labels(self, model_name):
        """Test forward pass with labels for loss computation."""
        config = {"use_flash_attn": False}
        vlm = create_vlm(model_name, config=config)
        vlm.eval()

        # Create dummy inputs
        batch_size = 2
        seq_len = 10
        img_size = 224 if model_name != "llava-v1.5-7b" else 336

        images = torch.randn(batch_size, 3, img_size, img_size)
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        labels = torch.randint(0, 1000, (batch_size, seq_len))

        with torch.no_grad():
            outputs = vlm(images=images, input_ids=input_ids, labels=labels)

        # Check loss is computed
        assert "loss" in outputs
        assert isinstance(outputs["loss"], torch.Tensor)
        assert outputs["loss"].ndim == 0  # scalar


class TestVLMGeneration:
    """Test VLM generation capabilities."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "qwen2-vl-2b",
            "florence-2-base",
            "paligemma-3b",
            "internvl2-2b",
            "llava-v1.5-7b",
        ],
    )
    def test_generate(self, model_name):
        """Test text generation."""
        config = {"use_flash_attn": False}
        vlm = create_vlm(model_name, config=config)
        vlm.eval()

        # Create dummy inputs
        batch_size = 1
        img_size = 224 if model_name != "llava-v1.5-7b" else 336

        images = torch.randn(batch_size, 3, img_size, img_size)
        input_ids = torch.randint(0, 1000, (batch_size, 5))

        with torch.no_grad():
            generated = vlm.generate(
                images=images,
                input_ids=input_ids,
                max_new_tokens=10,
                temperature=1.0,
            )

        # Check output
        assert isinstance(generated, torch.Tensor)
        assert generated.shape[0] == batch_size
        assert generated.shape[1] > input_ids.shape[1]  # Generated more tokens


class TestVLMUtilities:
    """Test VLM utility functions."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "qwen2-vl-2b",
            "florence-2-base",
            "paligemma-3b",
            "internvl2-2b",
            "llava-v1.5-7b",
        ],
    )
    def test_freeze_unfreeze(self, model_name):
        """Test freeze and unfreeze methods."""
        config = {"use_flash_attn": False}
        vlm = create_vlm(model_name, config=config)

        # Initially parameters should be trainable
        trainable_before = sum(p.requires_grad for p in vlm.parameters())
        assert trainable_before > 0

        # Freeze
        vlm.freeze()
        trainable_frozen = sum(p.requires_grad for p in vlm.parameters())
        assert trainable_frozen == 0

        # Unfreeze
        vlm.unfreeze()
        trainable_after = sum(p.requires_grad for p in vlm.parameters())
        assert trainable_after == trainable_before

    @pytest.mark.parametrize(
        "model_name",
        [
            "qwen2-vl-2b",
            "florence-2-base",
            "paligemma-3b",
            "internvl2-2b",
            "llava-v1.5-7b",
        ],
    )
    def test_get_num_parameters(self, model_name):
        """Test get_num_parameters method."""
        config = {"use_flash_attn": False}
        vlm = create_vlm(model_name, config=config)

        # Get total parameters
        total_params = vlm.get_num_parameters(trainable_only=False)
        assert total_params > 0

        # Get trainable parameters
        trainable_params = vlm.get_num_parameters(trainable_only=True)
        assert trainable_params > 0
        assert trainable_params <= total_params

        # Freeze and check
        vlm.freeze()
        trainable_after_freeze = vlm.get_num_parameters(trainable_only=True)
        assert trainable_after_freeze == 0


class TestQwenVLSpecific:
    """Test Qwen-VL specific features."""

    def test_qwen_vl_lora(self):
        """Test Qwen-VL with LoRA."""
        config = {
            "use_flash_attn": False,
            "use_lora": True,
            "lora_rank": 8,
            "lora_alpha": 16,
        }
        vlm = create_vlm("qwen2-vl-2b", config=config)

        # Check LoRA adapters are created
        assert hasattr(vlm, "_config")
        assert vlm._config.use_lora

    def test_qwen_vl_3d_input(self):
        """Test Qwen-VL with temporal dimension."""
        config = {"use_flash_attn": False}
        vlm = create_vlm("qwen2-vl-2b", config=config)
        vlm.eval()

        # Create 5D input [B, T, C, H, W]
        batch_size = 1
        temporal = 4
        images = torch.randn(batch_size, temporal, 3, 224, 224)

        with torch.no_grad():
            features = vlm.encode_image(images)

        assert features.ndim == 3
        assert features.shape[0] == batch_size


class TestFlorenceSpecific:
    """Test Florence-2 specific features."""

    def test_florence_task_support(self):
        """Test Florence-2 with different tasks."""
        config = {"use_flash_attn": False}
        vlm = create_vlm("florence-2-base", config=config)
        vlm.eval()

        batch_size = 1
        images = torch.randn(batch_size, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (batch_size, 10))

        # Test with task
        with torch.no_grad():
            outputs = vlm(images=images, input_ids=input_ids, task="caption")

        assert "embeddings" in outputs
        assert "logits" in outputs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
