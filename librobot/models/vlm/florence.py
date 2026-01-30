"""Florence-2 Vision-Language Model."""

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AbstractVLM
from .registry import register_vlm


@dataclass
class FlorenceConfig:
    """Configuration for Florence-2 model."""

    # Vision encoder config (DaViT backbone)
    vision_hidden_sizes: list[int] = None  # [96, 192, 384, 768] for base
    vision_num_heads: list[int] = None  # [3, 6, 12, 24] for base
    vision_depths: list[int] = None  # [1, 1, 9, 1] for base
    vision_patch_size: int = 4
    vision_window_size: int = 12
    vision_in_channels: int = 3

    # Language decoder config
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    vocab_size: int = 51289
    max_position_embeddings: int = 1024

    # Training config
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-6
    use_cache: bool = True

    # Attention config
    use_flash_attn: bool = False

    # Task-specific config
    task_tokens: dict[str, int] = None
    prompt_templates: dict[str, str] = None

    # Model variant
    variant: str = "florence-2-base"  # florence-2-base, florence-2-large

    def __post_init__(self):
        if self.vision_hidden_sizes is None:
            if "large" in self.variant:
                self.vision_hidden_sizes = [192, 384, 768, 1536]
                self.vision_num_heads = [6, 12, 24, 48]
                self.vision_depths = [1, 1, 18, 1]
                self.hidden_size = 1024
                self.num_hidden_layers = 24
                self.num_attention_heads = 16
                self.intermediate_size = 4096
            else:
                self.vision_hidden_sizes = [96, 192, 384, 768]
                self.vision_num_heads = [3, 6, 12, 24]
                self.vision_depths = [1, 1, 9, 1]

        if self.task_tokens is None:
            self.task_tokens = {
                "caption": 51200,
                "detailed_caption": 51201,
                "more_detailed_caption": 51202,
                "ocr": 51203,
                "ocr_with_region": 51204,
                "phrase_grounding": 51205,
                "region_to_segmentation": 51206,
                "open_vocabulary_detection": 51207,
                "region_to_category": 51208,
                "region_to_description": 51209,
            }


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, D, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        x = self.norm(x)
        return x


class WindowAttention(nn.Module):
    """Window-based multi-head self attention."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )

        coords = torch.stack(
            torch.meshgrid([torch.arange(window_size), torch.arange(window_size)], indexing='ij')
        )
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DaViTBlock(nn.Module):
    """DaViT transformer block with window attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size, num_heads, qkv_bias, attn_drop, drop
        )
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class DaViTStage(nn.Module):
    """DaViT stage with multiple blocks."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        downsample: bool = False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            DaViTBlock(dim, num_heads, window_size, mlp_ratio, qkv_bias, drop, attn_drop)
            for _ in range(depth)
        ])

        if downsample:
            self.downsample = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * 2, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)

        if self.downsample is not None:
            # Reshape for spatial downsampling
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            x = x.view(B, H, W, C)

            # 2x2 pooling
            x = x.reshape(B, H // 2, 2, W // 2, 2, C)
            x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H // 2 * W // 2, 4 * C)
            x = self.downsample(x)

        return x


class FlorenceVisionEncoder(nn.Module):
    """DaViT-based vision encoder for Florence-2."""

    def __init__(self, config: FlorenceConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbed(
            patch_size=config.vision_patch_size,
            in_chans=config.vision_in_channels,
            embed_dim=config.vision_hidden_sizes[0],
        )

        # DaViT stages
        self.stages = nn.ModuleList()
        for i in range(len(config.vision_depths)):
            stage = DaViTStage(
                dim=config.vision_hidden_sizes[i],
                depth=config.vision_depths[i],
                num_heads=config.vision_num_heads[i],
                window_size=config.vision_window_size,
                downsample=(i < len(config.vision_depths) - 1),
            )
            self.stages.append(stage)

        self.norm = nn.LayerNorm(config.vision_hidden_sizes[-1])

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, C, H, W]
        Returns:
            Vision features [B, N, D]
        """
        x = self.patch_embed(pixel_values)

        for stage in self.stages:
            x = stage(x)

        x = self.norm(x)
        return x


class FlorenceDecoderAttention(nn.Module):
    """Multi-head attention for decoder."""

    def __init__(self, config: FlorenceConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        B, L, D = hidden_states.shape

        # Self-attention or cross-attention
        q = self.q_proj(hidden_states)

        if encoder_hidden_states is not None:
            # Cross-attention
            k = self.k_proj(encoder_hidden_states)
            v = self.v_proj(encoder_hidden_states)
        else:
            # Self-attention
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if is_causal:
            causal_mask = torch.tril(torch.ones(L, L, device=attn.device, dtype=torch.bool))
            attn = attn.masked_fill(~causal_mask, float('-inf'))

        if attention_mask is not None:
            attn = attn + attention_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, -1, D)
        out = self.out_proj(out)

        return out


class FlorenceDecoderLayer(nn.Module):
    """Transformer decoder layer."""

    def __init__(self, config: FlorenceConfig):
        super().__init__()
        self.self_attn = FlorenceDecoderAttention(config)
        self.cross_attn = FlorenceDecoderAttention(config)

        self.self_attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.cross_attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        residual = hidden_states
        hidden_states = self.self_attn_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask, is_causal=True)
        hidden_states = residual + hidden_states

        # Cross-attention
        residual = hidden_states
        hidden_states = self.cross_attn_norm(hidden_states)
        hidden_states = self.cross_attn(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )
        hidden_states = residual + hidden_states

        # FFN
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class FlorenceDecoder(nn.Module):
    """Transformer decoder for Florence-2."""

    def __init__(self, config: FlorenceConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        self.layers = nn.ModuleList([
            FlorenceDecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])

        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)

        # Embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        position_embeds = self.embed_positions(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)

        # Decoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, encoder_hidden_states, attention_mask)

        hidden_states = self.norm(hidden_states)
        return hidden_states


@register_vlm(name="florence-2-base", aliases=["florence-2", "florence"])
@register_vlm(name="florence-2-large")
class Florence2(AbstractVLM):
    """
    Florence-2 Vision-Language Model.

    Unified vision-language architecture with:
    - DaViT vision encoder with window attention
    - Transformer decoder for text generation
    - Multi-task learning capabilities
    - OCR and spatial understanding
    - Dynamic resolution support

    Args:
        config: FlorenceConfig or dict
        pretrained: Path to pretrained weights or HuggingFace model ID
        freeze_vision: Whether to freeze vision encoder
    """

    def __init__(
        self,
        config: Union[FlorenceConfig, dict[str, Any]],
        pretrained: Optional[str] = None,
        freeze_vision: bool = False,
    ):
        super().__init__()

        if isinstance(config, dict):
            config = FlorenceConfig(**config)

        self._config = config

        # Vision encoder
        self.vision_encoder = FlorenceVisionEncoder(config)

        # Vision projection
        self.vision_proj = nn.Linear(
            config.vision_hidden_sizes[-1],
            config.hidden_size,
        )

        # Text decoder
        self.decoder = FlorenceDecoder(config)

        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Task embeddings
        if config.task_tokens:
            self.task_embeddings = nn.Embedding(
                len(config.task_tokens),
                config.hidden_size,
            )
        else:
            self.task_embeddings = None

        # Load pretrained weights
        if pretrained is not None:
            self.load_pretrained_weights(pretrained)

        # Freeze vision encoder if specified
        if freeze_vision:
            for param in self.vision_encoder.parameters():
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
        text: Union[str, list[str]],
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
        input_ids = inputs["input_ids"].to(self.decoder.embed_tokens.weight.device)

        # Get embeddings
        embeddings = self.decoder.embed_tokens(input_ids)
        return embeddings

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        task: Optional[str] = None,
        return_dict: bool = True,
        **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: Input images [B, C, H, W]
            input_ids: Input token IDs [B, L]
            attention_mask: Attention mask [B, L]
            position_ids: Position IDs [B, L]
            labels: Target labels for language modeling
            task: Task name (e.g., "caption", "ocr")
            return_dict: Whether to return dict

        Returns:
            Dictionary containing:
                - embeddings: Final hidden states [B, L, D]
                - logits: Language model logits [B, L, V]
                - loss: Optional loss if labels provided
        """
        # Encode images
        if images is None:
            raise ValueError("Images are required for Florence-2")

        encoder_hidden_states = self.encode_image(images)

        # Add task embedding if specified
        if task is not None and self.task_embeddings is not None:
            task_id = self._config.task_tokens.get(task)
            if task_id is not None:
                task_embed = self.task_embeddings(
                    torch.tensor([task_id], device=encoder_hidden_states.device)
                )
                encoder_hidden_states = torch.cat([task_embed.unsqueeze(0), encoder_hidden_states], dim=1)

        # Decode text
        hidden_states = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
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
        images: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        task: Optional[str] = None,
        **kwargs
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
            task: Task name

        Returns:
            Generated token IDs [B, L]
        """
        # Encode images once
        encoder_hidden_states = self.encode_image(images)

        # Add task embedding if specified
        if task is not None and self.task_embeddings is not None:
            task_id = self._config.task_tokens.get(task)
            if task_id is not None:
                task_embed = self.task_embeddings(
                    torch.tensor([task_id], device=encoder_hidden_states.device)
                )
                encoder_hidden_states = torch.cat([task_embed.unsqueeze(0), encoder_hidden_states], dim=1)

        # Initialize with start token if no input
        if input_ids is None:
            input_ids = torch.zeros((images.size(0), 1), dtype=torch.long, device=images.device)

        for _ in range(max_new_tokens):
            # Forward pass
            hidden_states = self.decoder(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
            )

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

        return input_ids

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self._config.hidden_size

    @property
    def config(self) -> dict[str, Any]:
        """Get model configuration."""
        return {
            "model_type": "florence-2",
            "variant": self._config.variant,
            "hidden_size": self._config.hidden_size,
            "vision_hidden_sizes": self._config.vision_hidden_sizes,
            "num_hidden_layers": self._config.num_hidden_layers,
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
