"""Image tokenizer for visual observations."""

from typing import Any, Optional, Union

import numpy as np


class ImageTokenizer:
    """
    Tokenizer for image observations.

    Converts images into patch tokens for vision transformers
    or discrete visual tokens for autoregressive models.
    """

    def __init__(
        self,
        image_size: Union[int, tuple[int, int]] = 224,
        patch_size: int = 14,
        num_channels: int = 3,
        normalize: bool = True,
        mean: Optional[list[float]] = None,
        std: Optional[list[float]] = None,
        tokenize_method: str = "patch",
        num_visual_tokens: int = 8192,
    ):
        """
        Initialize image tokenizer.

        Args:
            image_size: Target image size (H, W) or single int for square
            patch_size: Size of image patches
            num_channels: Number of image channels
            normalize: Whether to normalize images
            mean: Normalization mean per channel
            std: Normalization std per channel
            tokenize_method: Method ("patch", "vqvae", "dino")
            num_visual_tokens: Number of discrete visual tokens (for VQ methods)
        """
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = tuple(image_size)

        self.patch_size = patch_size
        self.num_channels = num_channels
        self.normalize = normalize
        self.tokenize_method = tokenize_method
        self.num_visual_tokens = num_visual_tokens

        # Default ImageNet normalization
        self.mean = np.array(mean or [0.485, 0.456, 0.406])
        self.std = np.array(std or [0.229, 0.224, 0.225])

        # Calculate number of patches
        self.num_patches_h = self.image_size[0] // patch_size
        self.num_patches_w = self.image_size[1] // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Patch embedding dimension
        self.patch_dim = patch_size * patch_size * num_channels

        # Special tokens
        self.cls_token_id = 0
        self.sep_token_id = 1
        self.pad_token_id = 2
        self.visual_token_offset = 3

        # Vocab size for discrete tokenization
        self.vocab_size = self.visual_token_offset + num_visual_tokens

        # VQ encoder (lazy loaded)
        self._vq_encoder = None

    def preprocess(
        self,
        image: np.ndarray,
        return_tensors: Optional[str] = None,
    ) -> Union[np.ndarray, dict[str, Any]]:
        """
        Preprocess image for model input.

        Args:
            image: Input image [H, W, C] or [C, H, W] or batch [B, ...]
            return_tensors: Return format ("pt", "np", None)

        Returns:
            Preprocessed image or dict with pixel_values
        """
        image = np.asarray(image, dtype=np.float32)

        # Handle batch dimension
        if image.ndim == 3:
            image = image[np.newaxis, ...]
            squeeze = True
        else:
            squeeze = False

        # Ensure CHW format
        if image.shape[-1] in (1, 3, 4):  # HWC format
            image = np.transpose(image, (0, 3, 1, 2))

        # Resize if needed
        if image.shape[2:] != self.image_size:
            image = self._resize_batch(image)

        # Normalize to [0, 1] if in [0, 255]
        if image.max() > 1.0:
            image = image / 255.0

        # Apply normalization
        if self.normalize:
            image = (image - self.mean[np.newaxis, :, np.newaxis, np.newaxis]) / self.std[
                np.newaxis, :, np.newaxis, np.newaxis
            ]

        if squeeze:
            image = image[0]

        if return_tensors == "pt":
            try:
                import torch

                image = torch.from_numpy(image)
            except ImportError:
                pass

        if return_tensors:
            return {"pixel_values": image}
        return image

    def _resize_batch(self, images: np.ndarray) -> np.ndarray:
        """Resize batch of images."""
        try:
            from PIL import Image

            resized = []
            for img in images:
                # CHW -> HWC for PIL
                img_hwc = np.transpose(img, (1, 2, 0))
                pil_img = Image.fromarray((img_hwc * 255).astype(np.uint8))
                pil_img = pil_img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
                resized_hwc = np.array(pil_img).astype(np.float32) / 255.0
                resized.append(np.transpose(resized_hwc, (2, 0, 1)))
            return np.stack(resized)
        except ImportError:
            # Simple nearest neighbor resize
            return images

    def patchify(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to sequence of patch embeddings.

        Args:
            image: Preprocessed image [C, H, W] or [B, C, H, W]

        Returns:
            Patch embeddings [num_patches, patch_dim] or [B, num_patches, patch_dim]
        """
        image = np.asarray(image)

        if image.ndim == 3:
            image = image[np.newaxis, ...]
            squeeze = True
        else:
            squeeze = False

        B, C, H, W = image.shape

        # Reshape to patches
        patches = image.reshape(
            B, C, self.num_patches_h, self.patch_size, self.num_patches_w, self.patch_size
        )
        patches = patches.transpose(0, 2, 4, 3, 5, 1)  # B, nH, nW, pH, pW, C
        patches = patches.reshape(B, self.num_patches, self.patch_dim)

        if squeeze:
            patches = patches[0]

        return patches

    def encode(
        self,
        image: np.ndarray,
        add_cls_token: bool = True,
    ) -> np.ndarray:
        """
        Encode image to patch or discrete tokens.

        Args:
            image: Input image
            add_cls_token: Whether to add CLS token

        Returns:
            Token IDs or patch embeddings
        """
        # Preprocess
        image = self.preprocess(image)

        if self.tokenize_method == "patch":
            return self._encode_patches(image, add_cls_token)
        elif self.tokenize_method == "vqvae":
            return self._encode_vqvae(image, add_cls_token)
        elif self.tokenize_method == "dino":
            return self._encode_dino(image, add_cls_token)
        else:
            return self._encode_patches(image, add_cls_token)

    def _encode_patches(self, image: np.ndarray, add_cls_token: bool) -> np.ndarray:
        """Encode as patch embeddings."""
        patches = self.patchify(image)

        if add_cls_token:
            if patches.ndim == 2:
                # Add CLS token placeholder
                cls_token = np.zeros((1, self.patch_dim))
                patches = np.concatenate([cls_token, patches], axis=0)
            else:
                B = patches.shape[0]
                cls_token = np.zeros((B, 1, self.patch_dim))
                patches = np.concatenate([cls_token, patches], axis=1)

        return patches

    def _encode_vqvae(self, image: np.ndarray, add_cls_token: bool) -> np.ndarray:
        """Encode using VQ-VAE discrete tokens."""
        # Simplified: hash-based discretization
        patches = self.patchify(image)

        if patches.ndim == 2:
            patches = patches[np.newaxis, ...]
            squeeze = True
        else:
            squeeze = False

        # Simple hash-based quantization
        tokens = (np.sum(patches, axis=-1) * 1000).astype(np.int64)
        tokens = np.abs(tokens) % self.num_visual_tokens
        tokens = tokens + self.visual_token_offset

        if add_cls_token:
            B = tokens.shape[0]
            cls = np.full((B, 1), self.cls_token_id)
            tokens = np.concatenate([cls, tokens], axis=1)

        if squeeze:
            tokens = tokens[0]

        return tokens

    def _encode_dino(self, image: np.ndarray, add_cls_token: bool) -> np.ndarray:
        """Encode using DINO features (requires model)."""
        # Fallback to patch encoding if DINO not available
        return self._encode_patches(image, add_cls_token)

    def decode(
        self,
        tokens: np.ndarray,
        remove_cls_token: bool = True,
    ) -> np.ndarray:
        """
        Decode tokens back to image (for patch method only).

        Args:
            tokens: Patch embeddings or discrete tokens
            remove_cls_token: Whether to remove CLS token

        Returns:
            Reconstructed image
        """
        tokens = np.asarray(tokens)

        if tokens.ndim == 1:
            # Discrete tokens - cannot reconstruct
            return None

        if remove_cls_token:
            if tokens.ndim == 2:
                tokens = tokens[1:]  # Remove first token
            else:
                tokens = tokens[:, 1:]

        return self._unpatchify(tokens)

    def _unpatchify(self, patches: np.ndarray) -> np.ndarray:
        """Convert patches back to image."""
        if patches.ndim == 2:
            patches = patches[np.newaxis, ...]
            squeeze = True
        else:
            squeeze = False

        B = patches.shape[0]

        # Reshape from flat patches
        patches = patches.reshape(
            B,
            self.num_patches_h,
            self.num_patches_w,
            self.patch_size,
            self.patch_size,
            self.num_channels,
        )

        # Rearrange to image
        image = patches.transpose(0, 5, 1, 3, 2, 4)  # B, C, nH, pH, nW, pW
        image = image.reshape(B, self.num_channels, *self.image_size)

        if squeeze:
            image = image[0]

        return image

    def get_position_ids(self, batch_size: int = 1) -> np.ndarray:
        """
        Get position IDs for patches.

        Args:
            batch_size: Batch size

        Returns:
            Position IDs [B, num_patches + 1]
        """
        num_positions = self.num_patches + 1  # +1 for CLS
        positions = np.arange(num_positions)[np.newaxis, :]
        positions = np.repeat(positions, batch_size, axis=0)
        return positions

    def __call__(
        self, image: np.ndarray, return_tensors: Optional[str] = None, **kwargs
    ) -> Union[np.ndarray, dict[str, Any]]:
        """Preprocess image (alias for preprocess)."""
        return self.preprocess(image, return_tensors=return_tensors)


__all__ = ["ImageTokenizer"]
