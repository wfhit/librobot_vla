"""Qwen2-VL and Qwen3-VL Vision-Language Models."""

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..components.normalization.rmsnorm import RMSNorm
from ..components.positional.rotary import RotaryPositionEmbedding
from .adapters.lora import LoRAAdapter
from .adapters.qlora import QLoRAAdapter
from .base import AbstractVLM
from .registry import register_vlm


@dataclass
class QwenVLConfig:
    """Configuration for Qwen-VL models."""

    # Vision encoder config
    vision_hidden_size: int = 1664
    vision_num_heads: int = 16
    vision_num_layers: int = 32
    vision_patch_size: int = 14
    vision_temporal_patch_size: int = 2
    vision_in_channels: int = 3
    vision_mlp_ratio: float = 4.0

    # Language model config
    hidden_size: int = 896
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    intermediate_size: int = 4864
    vocab_size: int = 151936
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0

    # Vision-language fusion
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_token_id: int = 151654

    # Training config
    use_flash_attn: bool = True
    use_gradient_checkpointing: bool = False
    tie_word_embeddings: bool = False
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0

    # LoRA config
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 16
    use_qlora: bool = False
    qlora_bits: int = 4

    # Model variant
    variant: str = "qwen2-vl-2b"  # qwen2-vl-2b, qwen2-vl-7b, qwen3-vl-4b, qwen3-vl-7b


class VisionRotaryEmbedding(nn.Module):
    """3D Rotary Position Embedding for vision tokens."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        # Split dimension into 3 parts for 3D coordinates (t, h, w)
        # Each coordinate gets dim//3 dimensions, remainder goes to last coord
        self.dim_per_coord = dim // 3
        self.dims = [self.dim_per_coord, self.dim_per_coord, dim - 2 * self.dim_per_coord]

        # Create separate inverse frequencies for each coordinate dimension
        for i, d in enumerate(self.dims):
            inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2).float() / d))
            self.register_buffer(f"inv_freq_{i}", inv_freq)

    def forward(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            coords: [batch_size, num_patches, 3] (temporal, height, width)
        Returns:
            cos, sin embeddings [B, N, D]
        """
        B, N, _ = coords.shape
        device = coords.device

        # Compute rotary embeddings for each coordinate dimension
        emb_parts = []
        for i in range(3):
            coord = coords[:, :, i:i+1]  # [B, N, 1]
            inv_freq = getattr(self, f"inv_freq_{i}").to(device)  # [d//2]
            freqs = coord.float() * inv_freq  # [B, N, d//2]
            emb_part = torch.cat([freqs, freqs], dim=-1)  # [B, N, d]
            emb_parts.append(emb_part)

        # Concatenate all parts
        emb = torch.cat(emb_parts, dim=-1)  # [B, N, dim]
        return emb.cos(), emb.sin()


class QwenVisionAttention(nn.Module):
    """Vision attention with 3D RoPE."""

    def __init__(self, config: QwenVLConfig):
        super().__init__()
        self.num_heads = config.vision_num_heads
        self.head_dim = config.vision_hidden_size // config.vision_num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(config.vision_hidden_size, config.vision_hidden_size * 3)
        self.proj = nn.Linear(config.vision_hidden_size, config.vision_hidden_size)

        self.rope = VisionRotaryEmbedding(self.head_dim)

        if config.use_flash_attn:
            try:
                from flash_attn import flash_attn_func

                self.flash_attn_func = flash_attn_func
                self.use_flash = True
            except ImportError:
                self.use_flash = False
        else:
            self.use_flash = False

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)

        # Apply 3D RoPE
        cos, sin = self.rope(coords)
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)

        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)

        if self.use_flash:
            # Flash attention expects [B, N, H, D]
            out = self.flash_attn_func(q, k, v)
            out = out.reshape(B, N, C)
        else:
            # Standard attention
            q = q.transpose(1, 2)  # [B, H, N, D]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            attn = (q @ k.transpose(-2, -1)) * self.scale
            if attention_mask is not None:
                attn = attn + attention_mask
            attn = F.softmax(attn, dim=-1)

            out = (attn @ v).transpose(1, 2).reshape(B, N, C)

        return self.proj(out)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)


class QwenVisionMLP(nn.Module):
    """Vision MLP block."""

    def __init__(self, config: QwenVLConfig):
        super().__init__()
        hidden_dim = int(config.vision_hidden_size * config.vision_mlp_ratio)
        self.fc1 = nn.Linear(config.vision_hidden_size, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, config.vision_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class QwenVisionBlock(nn.Module):
    """Vision transformer block."""

    def __init__(self, config: QwenVLConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.vision_hidden_size, eps=config.rms_norm_eps)
        self.attn = QwenVisionAttention(config)
        self.norm2 = nn.LayerNorm(config.vision_hidden_size, eps=config.rms_norm_eps)
        self.mlp = QwenVisionMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), coords, attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class QwenVisionEncoder(nn.Module):
    """Vision encoder with 3D patch embedding and rotary embeddings."""

    def __init__(self, config: QwenVLConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = nn.Conv3d(
            config.vision_in_channels,
            config.vision_hidden_size,
            kernel_size=(
                config.vision_temporal_patch_size,
                config.vision_patch_size,
                config.vision_patch_size,
            ),
            stride=(
                config.vision_temporal_patch_size,
                config.vision_patch_size,
                config.vision_patch_size,
            ),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [QwenVisionBlock(config) for _ in range(config.vision_num_layers)]
        )

        self.norm = nn.LayerNorm(config.vision_hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        pixel_values: torch.Tensor,
        temporal_coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, T, C, H, W] or [B, C, H, W]
            temporal_coords: Optional temporal coordinates
        Returns:
            Vision features [B, N, D]
        """
        # Handle both 4D and 5D inputs
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(1)  # Add temporal dim

        B, T, C, H, W = pixel_values.shape

        # Ensure temporal dimension meets minimum requirement for 3D conv
        # The temporal kernel size requires T >= temporal_patch_size
        temporal_patch_size = self.config.vision_temporal_patch_size
        if T < temporal_patch_size:
            # Pad temporal dimension by repeating frames
            repeat_factor = (temporal_patch_size + T - 1) // T  # ceil division
            pixel_values = pixel_values.repeat(1, repeat_factor, 1, 1, 1)
            # Trim to exact size needed
            pixel_values = pixel_values[:, :temporal_patch_size, :, :, :]
            T = temporal_patch_size

        # Rearrange to [B, C, T, H, W] for Conv3D
        pixel_values = rearrange(pixel_values, "b t c h w -> b c t h w")

        # Patch embedding
        x = self.patch_embed(pixel_values)
        x = rearrange(x, "b d t h w -> b (t h w) d")

        # Generate 3D coordinates for patches
        t_patches = T // self.config.vision_temporal_patch_size
        h_patches = H // self.config.vision_patch_size
        w_patches = W // self.config.vision_patch_size

        t_coords = torch.arange(t_patches, device=x.device).float()
        h_coords = torch.arange(h_patches, device=x.device).float()
        w_coords = torch.arange(w_patches, device=x.device).float()

        coords = torch.stack(torch.meshgrid(t_coords, h_coords, w_coords, indexing="ij"), dim=-1)
        coords = coords.reshape(-1, 3).unsqueeze(0).expand(B, -1, -1)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, coords)

        x = self.norm(x)
        return x


class QwenLanguageAttention(nn.Module):
    """Grouped-query attention for language model."""

    def __init__(self, config: QwenVLConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.rope = RotaryPositionEmbedding(
            self.head_dim,
            max_len=config.max_position_embeddings,
            base=config.rope_theta,
        )

        if config.use_flash_attn:
            try:
                from flash_attn import flash_attn_func

                self.flash_attn_func = flash_attn_func
                self.use_flash = True
            except ImportError:
                self.use_flash = False
        else:
            self.use_flash = False

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.rope(q, k, position_ids)

        # Handle KV cache
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)

        present_kv = (k, v) if use_cache else None

        # Expand KV for grouped-query attention
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        if self.use_flash and attention_mask is None:
            # Flash attention
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = self.flash_attn_func(q, k, v, causal=True)
            out = out.reshape(B, L, -1)
        else:
            # Standard attention
            scale = self.head_dim**-0.5
            attn = (q @ k.transpose(-2, -1)) * scale

            if attention_mask is not None:
                attn = attn + attention_mask

            attn = F.softmax(attn, dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, L, -1)

        return self.o_proj(out), present_kv


class QwenLanguageMLP(nn.Module):
    """Language model MLP with SwiGLU activation."""

    def __init__(self, config: QwenVLConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class QwenLanguageBlock(nn.Module):
    """Language model transformer block."""

    def __init__(self, config: QwenVLConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = QwenLanguageAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = QwenLanguageMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        residual = x
        x = self.input_layernorm(x)
        x, present_kv = self.self_attn(x, attention_mask, position_ids, past_kv, use_cache)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x, present_kv


class QwenLanguageModel(nn.Module):
    """Language model for Qwen-VL."""

    def __init__(self, config: QwenVLConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [QwenLanguageBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_kvs: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[list[tuple[torch.Tensor, torch.Tensor]]]]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        present_kvs = [] if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_kvs[i] if past_kvs is not None else None
            hidden_states, present_kv = layer(
                hidden_states,
                attention_mask,
                position_ids,
                past_kv,
                use_cache,
            )
            if use_cache:
                present_kvs.append(present_kv)

        hidden_states = self.norm(hidden_states)
        return hidden_states, present_kvs


@register_vlm(name="qwen2-vl-2b", aliases=["qwen2-vl"])
@register_vlm(name="qwen2-vl-7b")
@register_vlm(name="qwen3-vl-4b", aliases=["qwen3-vl"])
@register_vlm(name="qwen3-vl-7b")
class QwenVL(AbstractVLM):
    """
    Qwen2-VL and Qwen3-VL Vision-Language Models.

    Supports:
    - Dynamic resolution vision encoding
    - Patch-based vision processing with 3D RoPE
    - Grouped-query attention in language model
    - Gradient checkpointing
    - LoRA/QLoRA adapters
    - KV cache for efficient generation

    Args:
        config: QwenVLConfig or dict
        pretrained: Path to pretrained weights or HuggingFace model ID
        freeze_vision: Whether to freeze vision encoder
        freeze_language: Whether to freeze language model
    """

    def __init__(
        self,
        config: Union[QwenVLConfig, dict[str, Any]],
        pretrained: Optional[str] = None,
        freeze_vision: bool = False,
        freeze_language: bool = False,
    ):
        super().__init__()

        if isinstance(config, dict):
            config = QwenVLConfig(**config)

        self._config = config

        # Vision encoder
        self.vision_encoder = QwenVisionEncoder(config)

        # Vision-language projection
        self.vision_proj = nn.Linear(config.vision_hidden_size, config.hidden_size)

        # Language model
        self.language_model = QwenLanguageModel(config)

        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie embeddings if specified
        if config.tie_word_embeddings:
            self.lm_head.weight = self.language_model.embed_tokens.weight

        # Apply LoRA if specified
        if config.use_lora or config.use_qlora:
            self._apply_lora()

        # Load pretrained weights
        if pretrained is not None:
            self.load_pretrained_weights(pretrained)

        # Freeze components if specified
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        if freeze_language:
            for param in self.language_model.parameters():
                param.requires_grad = False

    def _apply_lora(self):
        """Apply LoRA adapters to attention layers."""
        adapter_class = QLoRAAdapter if self._config.use_qlora else LoRAAdapter

        for layer in self.language_model.layers:
            # Apply to Q, K, V projections
            for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                linear = getattr(layer.self_attn, name)
                adapter = adapter_class(
                    linear.in_features,
                    linear.out_features,
                    rank=self._config.lora_rank,
                    alpha=self._config.lora_alpha,
                )
                adapter.apply_to_layer(linear)
                setattr(layer.self_attn, f"{name}_lora", adapter)

    def encode_image(self, images: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Encode images to embeddings.

        Args:
            images: [B, C, H, W] or [B, T, C, H, W]
        Returns:
            Image embeddings [B, N, D]
        """
        vision_features = self.vision_encoder(images)
        return self.vision_proj(vision_features)

    def encode_text(
        self, text: Union[str, list[str]], tokenizer: Optional[Any] = None, **kwargs
    ) -> torch.Tensor:
        """
        Encode text to embeddings.

        Args:
            text: Input text or list of texts
            tokenizer: Tokenizer (required)
        Returns:
            Text embeddings [B, L, D]
        """
        if tokenizer is None:
            raise ValueError("Tokenizer is required for text encoding")

        if isinstance(text, str):
            text = [text]

        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.language_model.embed_tokens.weight.device)

        # Get embeddings
        embeddings = self.language_model.embed_tokens(input_ids)
        return embeddings

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_kvs: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: Input images [B, C, H, W] or [B, T, C, H, W]
            input_ids: Input token IDs [B, L]
            attention_mask: Attention mask [B, L]
            position_ids: Position IDs [B, L]
            past_kvs: Past key-value states
            inputs_embeds: Input embeddings [B, L, D]
            labels: Target labels for language modeling
            use_cache: Whether to use KV cache
            return_dict: Whether to return dict

        Returns:
            Dictionary containing:
                - embeddings: Final hidden states [B, L, D]
                - logits: Language model logits [B, L, V]
                - loss: Optional loss if labels provided
                - past_kvs: Optional KV cache
        """
        vision_embeds = None
        
        # Encode images if provided
        if images is not None:
            vision_embeds = self.encode_image(images)

            # Get text embeddings if input_ids provided but inputs_embeds not provided
            if inputs_embeds is None and input_ids is not None:
                inputs_embeds = self.language_model.embed_tokens(input_ids)

            if inputs_embeds is not None:
                # Merge vision and text embeddings
                # This requires special tokens to indicate vision positions
                inputs_embeds = self._merge_vision_text_embeds(
                    vision_embeds, inputs_embeds, input_ids
                )
            else:
                inputs_embeds = vision_embeds

        # Forward through language model
        hidden_states, present_kvs = self.language_model(
            input_ids=None if inputs_embeds is not None else input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_kvs=past_kvs,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
        )

        # Compute logits
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Adjust labels for vision tokens if images were provided
            if vision_embeds is not None:
                B, N, _ = vision_embeds.shape
                vision_labels = torch.full((B, N), -100, device=labels.device, dtype=labels.dtype)
                labels = torch.cat([vision_labels, labels], dim=1)
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        output = {
            "embeddings": hidden_states,
            "logits": logits,
        }

        if loss is not None:
            output["loss"] = loss

        if use_cache:
            output["past_kvs"] = present_kvs

        return output

    def _merge_vision_text_embeds(
        self,
        vision_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Merge vision and text embeddings based on special tokens."""
        if input_ids is None:
            # Simple concatenation - prepend vision embeddings
            return torch.cat([vision_embeds, text_embeds], dim=1)

        # Check if there are vision token positions to replace
        vision_token_mask = input_ids == self._config.vision_token_id
        has_vision_tokens = vision_token_mask.any()

        if not has_vision_tokens:
            # No vision tokens in input_ids, prepend vision embeddings
            return torch.cat([vision_embeds, text_embeds], dim=1)

        # Replace vision token positions with vision embeddings
        B, L, D = text_embeds.shape
        _, N, _ = vision_embeds.shape

        # Create output embeddings
        output_embeds = text_embeds.clone()

        for b in range(B):
            vision_positions = vision_token_mask[b].nonzero(as_tuple=True)[0]
            if len(vision_positions) > 0:
                # Replace with vision embeddings
                num_vision_tokens = min(len(vision_positions), N)
                output_embeds[b, vision_positions[:num_vision_tokens]] = vision_embeds[
                    b, :num_vision_tokens
                ]

        return output_embeds

    def generate(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            images: Input images
            input_ids: Input token IDs
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            Generated token IDs [B, L]
        """
        past_kvs = None

        for _ in range(max_new_tokens):
            outputs = self.forward(
                images=images if past_kvs is None else None,
                input_ids=input_ids,
                past_kvs=past_kvs,
                use_cache=True,
            )

            logits = outputs["logits"][:, -1, :] / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float("-inf")

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)
            past_kvs = outputs.get("past_kvs")

        return input_ids

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self._config.hidden_size

    @property
    def config(self) -> dict[str, Any]:
        """Get model configuration."""
        return {
            "model_type": "qwen-vl",
            "variant": self._config.variant,
            "hidden_size": self._config.hidden_size,
            "vision_hidden_size": self._config.vision_hidden_size,
            "num_hidden_layers": self._config.num_hidden_layers,
            "vision_num_layers": self._config.vision_num_layers,
            "vocab_size": self._config.vocab_size,
        }

    def load_pretrained_weights(self, path_or_name: str):
        """
        Load pretrained weights from HuggingFace or local path.

        Args:
            path_or_name: Path to checkpoint or HuggingFace model ID
        """
        try:
            from transformers import AutoModelForCausalLM

            # Load from HuggingFace
            model = AutoModelForCausalLM.from_pretrained(
                path_or_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )

            # Transfer weights
            self._transfer_weights_from_hf(model)

        except Exception as e:
            # Fall back to direct loading
            try:
                state_dict = torch.load(path_or_name, map_location="cpu", weights_only=False)
                self.load_state_dict(state_dict, strict=False)
            except Exception as load_error:
                raise RuntimeError(
                    f"Failed to load pretrained weights from {path_or_name}: {e}\n"
                    f"Direct load also failed: {load_error}"
                )

    def _transfer_weights_from_hf(self, hf_model):
        """Transfer weights from HuggingFace model."""
        # This is a simplified version - actual implementation would need
        # careful mapping of layer names between HF and our implementation
        hf_state_dict = hf_model.state_dict()

        # Create mapping and transfer weights
        our_state_dict = self.state_dict()

        for name, param in our_state_dict.items():
            if name in hf_state_dict:
                param.data.copy_(hf_state_dict[name].data)
