"""LLaVA Vision-Language Model."""

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components.normalization.rmsnorm import RMSNorm
from ..components.positional.rotary import RotaryPositionEmbedding
from .base import AbstractVLM
from .registry import register_vlm


@dataclass
class LLaVAConfig:
    """Configuration for LLaVA model."""

    # CLIP vision encoder config
    vision_hidden_size: int = 1024
    vision_num_hidden_layers: int = 24
    vision_num_attention_heads: int = 16
    vision_intermediate_size: int = 4096
    vision_patch_size: int = 14
    vision_image_size: int = 336
    vision_in_channels: int = 3

    # Language model config (LLaMA/Vicuna)
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 32
    intermediate_size: int = 11008
    vocab_size: int = 32000
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0

    # Vision-language projection
    mm_projector_type: str = "mlp2x_gelu"  # linear, mlp2x_gelu
    mm_hidden_size: int = 4096

    # Training config
    use_flash_attn: bool = True
    hidden_act: str = "silu"
    rms_norm_eps: float = 1e-5
    attention_dropout: float = 0.0

    # Special tokens
    image_token_index: int = -200
    ignore_index: int = -100

    # Model variant
    variant: str = "llava-v1.5-7b"  # llava-v1.5-7b, llava-v1.5-13b


class CLIPVisionAttention(nn.Module):
    """Multi-head attention for CLIP vision encoder."""

    def __init__(self, config: LLaVAConfig):
        super().__init__()
        self.num_heads = config.vision_num_attention_heads
        self.head_dim = config.vision_hidden_size // config.vision_num_attention_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(config.vision_hidden_size, config.vision_hidden_size * 3)
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

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
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


class CLIPVisionMLP(nn.Module):
    """MLP for CLIP vision encoder."""

    def __init__(self, config: LLaVAConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.vision_hidden_size, config.vision_intermediate_size)
        self.fc2 = nn.Linear(config.vision_intermediate_size, config.vision_hidden_size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class CLIPVisionBlock(nn.Module):
    """Transformer block for CLIP vision encoder."""

    def __init__(self, config: LLaVAConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.vision_hidden_size, eps=1e-5)
        self.attn = CLIPVisionAttention(config)
        self.layer_norm2 = nn.LayerNorm(config.vision_hidden_size, eps=1e-5)
        self.mlp = CLIPVisionMLP(config)

    def forward(
        self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.layer_norm1(x), attention_mask)
        x = x + self.mlp(self.layer_norm2(x))
        return x


class CLIPVisionEncoder(nn.Module):
    """CLIP vision encoder."""

    def __init__(self, config: LLaVAConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            config.vision_in_channels,
            config.vision_hidden_size,
            kernel_size=config.vision_patch_size,
            stride=config.vision_patch_size,
            bias=False,
        )

        # Class token
        self.class_embedding = nn.Parameter(torch.randn(config.vision_hidden_size))

        # Positional embedding
        num_patches = (config.vision_image_size // config.vision_patch_size) ** 2
        self.positional_embedding = nn.Parameter(
            torch.randn(num_patches + 1, config.vision_hidden_size)
        )

        self.pre_layernorm = nn.LayerNorm(config.vision_hidden_size, eps=1e-5)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [CLIPVisionBlock(config) for _ in range(config.vision_num_hidden_layers)]
        )

        self.post_layernorm = nn.LayerNorm(config.vision_hidden_size, eps=1e-5)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, C, H, W]
        Returns:
            Vision features [B, N, D]
        """
        B = pixel_values.shape[0]

        # Patch embedding
        x = self.patch_embed(pixel_values)
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]

        # Add class token
        class_embeds = self.class_embedding.expand(B, 1, -1)
        x = torch.cat([class_embeds, x], dim=1)

        # Add positional embedding
        x = x + self.positional_embedding.unsqueeze(0)

        x = self.pre_layernorm(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        # Remove class token and apply final layernorm
        x = x[:, 1:]
        x = self.post_layernorm(x)

        return x


class MultiModalProjector(nn.Module):
    """Projector to map vision features to language model space."""

    def __init__(self, config: LLaVAConfig):
        super().__init__()
        self.projector_type = config.mm_projector_type

        if config.mm_projector_type == "linear":
            self.layers = nn.Linear(config.vision_hidden_size, config.hidden_size)
        elif config.mm_projector_type == "mlp2x_gelu":
            self.layers = nn.Sequential(
                nn.Linear(config.vision_hidden_size, config.mm_hidden_size),
                nn.GELU(),
                nn.Linear(config.mm_hidden_size, config.hidden_size),
            )
        else:
            raise ValueError(f"Unknown projector type: {config.mm_projector_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LLaMAAttention(nn.Module):
    """Multi-head attention for LLaMA."""

    def __init__(self, config: LLaVAConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads if self.num_kv_heads != 0 else 1

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
            scale = self.head_dim**-0.5
            attn = (q @ k.transpose(-2, -1)) * scale

            # Causal mask
            causal_mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
            attn = attn.masked_fill(~causal_mask, float("-inf"))

            if attention_mask is not None:
                attn = attn + attention_mask

            attn = F.softmax(attn, dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(B, L, -1)

        return self.o_proj(out)


class LLaMAMLP(nn.Module):
    """MLP with SwiGLU activation for LLaMA."""

    def __init__(self, config: LLaVAConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class LLaMADecoderBlock(nn.Module):
    """Decoder block for LLaMA."""

    def __init__(self, config: LLaVAConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LLaMAAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LLaMAMLP(config)

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


class LLaMAModel(nn.Module):
    """LLaMA/Vicuna language model."""

    def __init__(self, config: LLaVAConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LLaMADecoderBlock(config) for _ in range(config.num_hidden_layers)]
        )
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


@register_vlm(name="llava-v1.5-7b", aliases=["llava", "llava-1.5"])
@register_vlm(name="llava-v1.5-13b")
class LLaVA(AbstractVLM):
    """
    LLaVA (Large Language and Vision Assistant) Model.

    Features:
    - CLIP vision encoder for image understanding
    - LLaMA/Vicuna language model
    - Simple projection layer for vision-language fusion
    - Instruction tuning for chat and visual reasoning
    - Efficient architecture

    Args:
        config: LLaVAConfig or dict
        pretrained: Path to pretrained weights or HuggingFace model ID
        freeze_vision: Whether to freeze vision encoder
        freeze_language: Whether to freeze language model
    """

    def __init__(
        self,
        config: Union[LLaVAConfig, dict[str, Any]],
        pretrained: Optional[str] = None,
        freeze_vision: bool = False,
        freeze_language: bool = False,
    ):
        super().__init__()

        if isinstance(config, dict):
            config = LLaVAConfig(**config)

        self._config = config

        # Vision encoder (CLIP)
        self.vision_encoder = CLIPVisionEncoder(config)

        # Multi-modal projector
        self.mm_projector = MultiModalProjector(config)

        # Language model (LLaMA/Vicuna)
        self.language_model = LLaMAModel(config)

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
        return self.mm_projector(vision_features)

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

    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Merge image features with text embeddings based on image token positions.

        Args:
            image_features: [B, N, D]
            inputs_embeds: [B, L, D]
            input_ids: [B, L] with image token indices

        Returns:
            Merged embeddings [B, L', D]
        """
        if input_ids is None:
            # Simple concatenation
            return torch.cat([image_features, inputs_embeds], dim=1)

        B, L, D = inputs_embeds.shape
        _, N, _ = image_features.shape

        # Find image token positions
        image_token_mask = input_ids == self._config.image_token_index

        # Create new embeddings with image features inserted
        new_embeds_list = []

        for b in range(B):
            cur_input_embeds = inputs_embeds[b]
            cur_image_features = image_features[b]

            if image_token_mask[b].any():
                # Split at image token positions
                image_positions = image_token_mask[b].nonzero(as_tuple=True)[0]

                # Build new embedding sequence
                cur_new_embeds = []
                prev_pos = 0

                for img_pos in image_positions:
                    # Add text before image
                    if img_pos > prev_pos:
                        cur_new_embeds.append(cur_input_embeds[prev_pos:img_pos])

                    # Add image features
                    cur_new_embeds.append(cur_image_features)
                    prev_pos = img_pos + 1

                # Add remaining text
                if prev_pos < L:
                    cur_new_embeds.append(cur_input_embeds[prev_pos:])

                cur_new_embeds = torch.cat(cur_new_embeds, dim=0)
            else:
                # No image token, use original embeddings
                cur_new_embeds = cur_input_embeds

            new_embeds_list.append(cur_new_embeds)

        # Pad to same length
        max_len = max(x.shape[0] for x in new_embeds_list)
        new_embeds = []

        for embeds in new_embeds_list:
            if embeds.shape[0] < max_len:
                padding = torch.zeros(
                    max_len - embeds.shape[0], D, device=embeds.device, dtype=embeds.dtype
                )
                embeds = torch.cat([embeds, padding], dim=0)
            new_embeds.append(embeds)

        return torch.stack(new_embeds, dim=0)

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
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
        # Prepare inputs
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.language_model.embed_tokens(input_ids)

        # Encode images if provided
        if images is not None:
            image_features = self.encode_image(images)

            # Merge image features with text embeddings
            inputs_embeds = self._merge_input_ids_with_image_features(
                image_features, inputs_embeds, input_ids
            )

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
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self._config.ignore_index,
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
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            images: Input images [B, C, H, W]
            input_ids: Input token IDs [B, L]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            Generated token IDs [B, L]
        """
        # Encode images once
        image_features = None
        if images is not None:
            image_features = self.encode_image(images)

        # Initialize input if not provided
        if input_ids is None:
            B = images.size(0) if images is not None else 1
            input_ids = torch.zeros((B, 1), dtype=torch.long, device=self.lm_head.weight.device)

        for _ in range(max_new_tokens):
            # Get embeddings
            inputs_embeds = self.language_model.embed_tokens(input_ids)

            # Merge with image features for first iteration
            if image_features is not None:
                inputs_embeds = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids
                )

            # Forward pass
            hidden_states = self.language_model(inputs_embeds=inputs_embeds)
            logits = self.lm_head(hidden_states[:, -1, :]) / temperature

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

            # Clear image features after first iteration
            image_features = None

        return input_ids

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self._config.hidden_size

    @property
    def config(self) -> dict[str, Any]:
        """Get model configuration."""
        return {
            "model_type": "llava",
            "variant": self._config.variant,
            "hidden_size": self._config.hidden_size,
            "vision_hidden_size": self._config.vision_hidden_size,
            "num_hidden_layers": self._config.num_hidden_layers,
            "vision_num_hidden_layers": self._config.vision_num_hidden_layers,
            "vocab_size": self._config.vocab_size,
        }

    def load_pretrained_weights(self, path_or_name: str):
        """
        Load pretrained weights from HuggingFace or local path.

        Args:
            path_or_name: Path to checkpoint or HuggingFace model ID
        """
        try:
            from transformers import LlavaForConditionalGeneration

            # Load from HuggingFace
            model = LlavaForConditionalGeneration.from_pretrained(
                path_or_name,
                trust_remote_code=True,
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
        hf_state_dict = hf_model.state_dict()
        our_state_dict = self.state_dict()

        for name, param in our_state_dict.items():
            if name in hf_state_dict:
                param.data.copy_(hf_state_dict[name].data)
