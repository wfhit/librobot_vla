"""InternVL2 Vision-Language Model."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .base import AbstractVLM
from .registry import register_vlm
from ..components.attention.flash_attention import FlashAttention
from ..components.positional.rotary import RotaryPositionEmbedding
from ..components.normalization.rmsnorm import RMSNorm
from ..components.normalization.layernorm import LayerNorm


@dataclass
class InternVLConfig:
    """Configuration for InternVL2 model."""

    # InternViT vision encoder config
    vision_hidden_size: int = 1024
    vision_num_hidden_layers: int = 24
    vision_num_attention_heads: int = 16
    vision_intermediate_size: int = 4096
    vision_patch_size: int = 14
    vision_image_size: int = 448
    vision_in_channels: int = 3

    # Language model config (InternLM2)
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    intermediate_size: int = 8192
    vocab_size: int = 92544
    max_position_embeddings: int = 32768
    rope_theta: float = 1000000.0

    # Vision-language fusion
    projection_dim: int = 2048
    pixel_shuffle_factor: int = 2  # For efficient high-res processing

    # Training config
    use_flash_attn: bool = True
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0

    # High-resolution support
    max_image_size: int = 4096
    dynamic_resolution: bool = True

    # Model variant
    variant: str = "internvl2-2b"  # internvl2-2b, internvl2-8b


class InternViTAttention(nn.Module):
    """Multi-head attention for InternViT."""

    def __init__(self, config: InternVLConfig):
        super().__init__()
        self.num_heads = config.vision_num_attention_heads
        self.head_dim = config.vision_hidden_size // config.vision_num_attention_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(config.vision_hidden_size, config.vision_hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.vision_hidden_size, config.vision_hidden_size)

        if config.use_flash_attn:
            try:
                from flash_attn import flash_attn_func
                self.flash_attn_func = flash_attn_func
                self.use_flash = True
            except ImportError:
                self.use_flash = False
        else:
            self.use_flash = False

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)

        if self.use_flash:
            q, k, v = qkv.unbind(2)
            out = self.flash_attn_func(q, k, v)
            out = out.reshape(B, N, C)
        else:
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * self.scale

            if attention_mask is not None:
                attn = attn + attention_mask

            attn = F.softmax(attn, dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, N, C)

        return self.proj(out)


class InternViTMLP(nn.Module):
    """MLP for InternViT."""

    def __init__(self, config: InternVLConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.vision_hidden_size, config.vision_intermediate_size)
        self.fc2 = nn.Linear(config.vision_intermediate_size, config.vision_hidden_size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class InternViTBlock(nn.Module):
    """Transformer block for InternViT."""

    def __init__(self, config: InternVLConfig):
        super().__init__()
        self.norm1 = LayerNorm(config.vision_hidden_size, eps=1e-6)
        self.attn = InternViTAttention(config)
        self.norm2 = LayerNorm(config.vision_hidden_size, eps=1e-6)
        self.mlp = InternViTMLP(config)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class PixelShuffle(nn.Module):
    """Pixel shuffle for efficient high-resolution processing."""

    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C/(scale^2), H*scale, W*scale]
        """
        return F.pixel_shuffle(x, self.scale_factor)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C*(scale^2), H/scale, W/scale]
        """
        B, C, H, W = x.shape
        s = self.scale_factor
        return rearrange(x, 'b c (h s1) (w s2) -> b (c s1 s2) h w', s1=s, s2=s)


class InternViTEncoder(nn.Module):
    """InternViT vision encoder with pixel shuffle."""

    def __init__(self, config: InternVLConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            config.vision_in_channels,
            config.vision_hidden_size,
            kernel_size=config.vision_patch_size,
            stride=config.vision_patch_size,
        )

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.vision_hidden_size))

        # Positional embedding (learnable)
        max_patches = (config.max_image_size // config.vision_patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches + 1, config.vision_hidden_size))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            InternViTBlock(config) for _ in range(config.vision_num_hidden_layers)
        ])

        self.norm = LayerNorm(config.vision_hidden_size, eps=1e-6)

        # Pixel shuffle for efficiency (optional)
        if config.pixel_shuffle_factor > 1:
            self.pixel_shuffle = PixelShuffle(config.pixel_shuffle_factor)
        else:
            self.pixel_shuffle = None

    def interpolate_pos_encoding(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Interpolate positional encoding for different image sizes."""
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        if npatch == N:
            return self.pos_embed

        # Separate class token and patch embeddings
        class_pos_embed = self.pos_embed[:, 0:1]
        patch_pos_embed = self.pos_embed[:, 1:]

        # Get original grid size
        dim = x.shape[-1]
        w0 = w // self.config.vision_patch_size
        h0 = h // self.config.vision_patch_size

        # Interpolate
        patch_pos_embed = F.interpolate(
            patch_pos_embed.reshape(1, int(N**0.5), int(N**0.5), dim).permute(0, 3, 1, 2),
            size=(h0, w0),
            mode='bicubic',
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, C, H, W]
        Returns:
            Vision features [B, N, D]
        """
        B, C, H, W = pixel_values.shape

        # Apply pixel shuffle for efficiency if configured
        if self.pixel_shuffle is not None:
            pixel_values = self.pixel_shuffle.inverse(pixel_values)

        # Patch embedding
        x = self.patch_embed(pixel_values)
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        pos_embed = self.interpolate_pos_encoding(x, H, W)
        x = x + pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Remove class token for downstream processing
        x = x[:, 1:]

        return x


class InternLM2Attention(nn.Module):
    """Grouped-query attention for InternLM2."""

    def __init__(self, config: InternVLConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
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
    ) -> torch.Tensor:
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.rope(q, k, position_ids)

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
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale

            # Causal mask
            causal_mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
            attn = attn.masked_fill(~causal_mask, float('-inf'))

            if attention_mask is not None:
                attn = attn + attention_mask

            attn = F.softmax(attn, dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, L, -1)

        return self.o_proj(out)


class InternLM2MLP(nn.Module):
    """MLP with SwiGLU activation for InternLM2."""

    def __init__(self, config: InternVLConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class InternLM2DecoderBlock(nn.Module):
    """Decoder block for InternLM2."""

    def __init__(self, config: InternVLConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = InternLM2Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = InternLM2MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, attention_mask, position_ids)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class InternLM2Model(nn.Module):
    """InternLM2 language model."""

    def __init__(self, config: InternVLConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            InternLM2DecoderBlock(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)

        hidden_states = self.norm(hidden_states)
        return hidden_states


@register_vlm(name="internvl2-2b", aliases=["internvl2", "internvl"])
@register_vlm(name="internvl2-8b")
class InternVL2(AbstractVLM):
    """
    InternVL2 Vision-Language Model.

    Features:
    - InternViT vision encoder with high-resolution support (up to 4K)
    - Pixel shuffle for efficient processing
    - InternLM2 language model
    - Dynamic resolution support
    - Strong multilingual capabilities

    Args:
        config: InternVLConfig or dict
        pretrained: Path to pretrained weights or HuggingFace model ID
        freeze_vision: Whether to freeze vision encoder
        freeze_language: Whether to freeze language model
    """

    def __init__(
        self,
        config: Union[InternVLConfig, Dict[str, Any]],
        pretrained: Optional[str] = None,
        freeze_vision: bool = False,
        freeze_language: bool = False,
    ):
        super().__init__()

        if isinstance(config, dict):
            config = InternVLConfig(**config)

        self._config = config

        # Vision encoder (InternViT)
        self.vision_encoder = InternViTEncoder(config)

        # Vision projection
        self.vision_proj = nn.Sequential(
            nn.Linear(config.vision_hidden_size, config.projection_dim),
            nn.GELU(),
            nn.Linear(config.projection_dim, config.projection_dim),
        )

        # Language model (InternLM2)
        self.language_model = InternLM2Model(config)

        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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

    def encode_image(self, images: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Encode images to embeddings.

        Args:
            images: [B, C, H, W]
        Returns:
            Image embeddings [B, N, D]
        """
        vision_features = self.vision_encoder(images)
        return self.vision_proj(vision_features)

    def encode_text(
        self,
        text: Union[str, List[str]],
        tokenizer: Optional[Any] = None,
        **kwargs
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
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: Input images [B, C, H, W]
            input_ids: Input token IDs [B, L]
            attention_mask: Attention mask [B, L]
            position_ids: Position IDs [B, L]
            inputs_embeds: Input embeddings [B, L, D]
            labels: Target labels for language modeling
            return_dict: Whether to return dict

        Returns:
            Dictionary containing:
                - embeddings: Final hidden states [B, L, D]
                - logits: Language model logits [B, L, V]
                - loss: Optional loss if labels provided
        """
        # Encode images if provided
        if images is not None:
            vision_embeds = self.encode_image(images)

            if inputs_embeds is not None:
                # Prepend vision embeddings
                inputs_embeds = torch.cat([vision_embeds, inputs_embeds], dim=1)
            else:
                inputs_embeds = vision_embeds

            # Adjust position IDs and attention mask
            B, N, _ = vision_embeds.shape
            if position_ids is not None:
                vision_position_ids = torch.arange(N, device=position_ids.device).unsqueeze(0).expand(B, -1)
                position_ids = torch.cat([vision_position_ids, position_ids + N], dim=1)

            if attention_mask is not None:
                vision_mask = torch.ones(B, N, device=attention_mask.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([vision_mask, attention_mask], dim=1)

        # Forward through language model
        hidden_states = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )

        # Compute logits
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Adjust labels for vision tokens
            if images is not None:
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

        return output

    def generate(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate text autoregressively."""
        # Encode images once
        vision_embeds = None
        if images is not None:
            vision_embeds = self.encode_image(images)

        # Initialize input if not provided
        if input_ids is None:
            B = images.size(0) if images is not None else 1
            input_ids = torch.zeros((B, 1), dtype=torch.long, device=self.lm_head.weight.device)

        for _ in range(max_new_tokens):
            # Get embeddings
            inputs_embeds = self.language_model.embed_tokens(input_ids)

            # Prepend vision embeddings for first iteration
            if vision_embeds is not None:
                inputs_embeds = torch.cat([vision_embeds, inputs_embeds], dim=1)

            # Forward pass
            hidden_states = self.language_model(inputs_embeds=inputs_embeds)
            logits = self.lm_head(hidden_states[:, -1, :]) / temperature

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

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
                logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Clear vision embeddings after first iteration
            vision_embeds = None

        return input_ids

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self._config.hidden_size

    @property
    def config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "model_type": "internvl2",
            "variant": self._config.variant,
            "hidden_size": self._config.hidden_size,
            "vision_hidden_size": self._config.vision_hidden_size,
            "num_hidden_layers": self._config.num_hidden_layers,
            "vision_num_hidden_layers": self._config.vision_num_hidden_layers,
            "vocab_size": self._config.vocab_size,
        }

    def load_pretrained_weights(self, path_or_name: str):
        """Load pretrained weights from HuggingFace or local path."""
        try:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(
                path_or_name,
                trust_remote_code=True,
            )

            self._transfer_weights_from_hf(model)

        except Exception as e:
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
        hf_state_dict = hf_model.state_dict()
        our_state_dict = self.state_dict()

        for name, param in our_state_dict.items():
            if name in hf_state_dict:
                param.data.copy_(hf_state_dict[name].data)
