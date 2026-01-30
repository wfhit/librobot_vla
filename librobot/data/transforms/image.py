"""Image transforms for data augmentation."""

from typing import Any, Union

import numpy as np


class ImageTransform:
    """Base class for image transforms."""

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply transform to sample."""
        if "images" in sample:
            sample["images"] = self.transform(sample["images"])
        return sample

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Transform image. Override in subclasses."""
        return image


class Resize(ImageTransform):
    """Resize image to target size."""

    def __init__(self, size: Union[int, tuple[int, int]]):
        """
        Args:
            size: Target size (H, W) or single int for square
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Resize image using bilinear interpolation."""
        try:
            from PIL import Image

            # Handle batch dimension
            if image.ndim == 4:
                return np.stack([self.transform(img) for img in image])

            # Handle CHW vs HWC
            if image.shape[0] in (1, 3, 4):  # CHW
                image = np.transpose(image, (1, 2, 0))
                was_chw = True
            else:
                was_chw = False

            pil_img = Image.fromarray((image * 255).astype(np.uint8))
            pil_img = pil_img.resize((self.size[1], self.size[0]), Image.BILINEAR)
            result = np.array(pil_img).astype(np.float32) / 255.0

            if was_chw:
                result = np.transpose(result, (2, 0, 1))

            return result
        except ImportError:
            return image


class RandomCrop(ImageTransform):
    """Random crop of image."""

    def __init__(
        self,
        size: Union[int, tuple[int, int]],
        padding: int = 0,
    ):
        """
        Args:
            size: Crop size (H, W) or single int
            padding: Padding before crop
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)
        self.padding = padding

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Apply random crop."""
        if image.ndim == 4:
            return np.stack([self.transform(img) for img in image])

        # Determine format
        if image.shape[0] in (1, 3, 4):  # CHW
            C, H, W = image.shape
            is_chw = True
        else:  # HWC
            H, W, C = image.shape
            is_chw = False

        # Add padding
        if self.padding > 0:
            if is_chw:
                image = np.pad(
                    image,
                    ((0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                    mode="reflect",
                )
                _, H, W = image.shape
            else:
                image = np.pad(
                    image,
                    ((self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                    mode="reflect",
                )
                H, W, _ = image.shape

        # Random crop position
        top = np.random.randint(0, max(1, H - self.size[0] + 1))
        left = np.random.randint(0, max(1, W - self.size[1] + 1))

        if is_chw:
            return image[:, top : top + self.size[0], left : left + self.size[1]]
        else:
            return image[top : top + self.size[0], left : left + self.size[1], :]


class CenterCrop(ImageTransform):
    """Center crop of image."""

    def __init__(self, size: Union[int, tuple[int, int]]):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)

    def transform(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 4:
            return np.stack([self.transform(img) for img in image])

        if image.shape[0] in (1, 3, 4):  # CHW
            _, H, W = image.shape
            top = (H - self.size[0]) // 2
            left = (W - self.size[1]) // 2
            return image[:, top : top + self.size[0], left : left + self.size[1]]
        else:  # HWC
            H, W, _ = image.shape
            top = (H - self.size[0]) // 2
            left = (W - self.size[1]) // 2
            return image[top : top + self.size[0], left : left + self.size[1], :]


class ColorJitter(ImageTransform):
    """Random color jitter augmentation."""

    def __init__(
        self,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.1,
        hue: float = 0.05,
    ):
        """
        Args:
            brightness: Brightness jitter factor
            contrast: Contrast jitter factor
            saturation: Saturation jitter factor
            hue: Hue jitter factor
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def transform(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 4:
            return np.stack([self.transform(img) for img in image])

        # Brightness
        if self.brightness > 0:
            factor = 1 + np.random.uniform(-self.brightness, self.brightness)
            image = image * factor

        # Contrast
        if self.contrast > 0:
            factor = 1 + np.random.uniform(-self.contrast, self.contrast)
            mean = np.mean(image)
            image = (image - mean) * factor + mean

        # Clip to valid range
        image = np.clip(image, 0, 1)

        return image


class RandomHorizontalFlip(ImageTransform):
    """Random horizontal flip."""

    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of flip
        """
        self.p = p

    def transform(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            if image.ndim == 4:
                return np.flip(image, axis=3)  # BCHW
            elif image.shape[0] in (1, 3, 4):  # CHW
                return np.flip(image, axis=2)
            else:  # HWC
                return np.flip(image, axis=1)
        return image


class Normalize(ImageTransform):
    """Normalize image with mean and std."""

    def __init__(
        self,
        mean: list[float] = [0.485, 0.456, 0.406],
        std: list[float] = [0.229, 0.224, 0.225],
    ):
        """
        Args:
            mean: Normalization mean per channel
            std: Normalization std per channel
        """
        self.mean = np.array(mean)
        self.std = np.array(std)

    def transform(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 4:  # BCHW
            mean = self.mean[np.newaxis, :, np.newaxis, np.newaxis]
            std = self.std[np.newaxis, :, np.newaxis, np.newaxis]
        elif image.shape[0] in (1, 3, 4):  # CHW
            mean = self.mean[:, np.newaxis, np.newaxis]
            std = self.std[:, np.newaxis, np.newaxis]
        else:  # HWC
            mean = self.mean
            std = self.std

        return (image - mean) / std


class ToTensor(ImageTransform):
    """Convert to tensor (PyTorch or NumPy CHW format)."""

    def __init__(self, device: str = "cpu"):
        self.device = device

    def transform(self, image: np.ndarray) -> np.ndarray:
        # Ensure float32
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Convert HWC to CHW if needed
        if image.ndim == 3 and image.shape[-1] in (1, 3, 4):
            image = np.transpose(image, (2, 0, 1))
        elif image.ndim == 4 and image.shape[-1] in (1, 3, 4):
            image = np.transpose(image, (0, 3, 1, 2))

        return image


__all__ = [
    "ImageTransform",
    "Resize",
    "RandomCrop",
    "CenterCrop",
    "ColorJitter",
    "RandomHorizontalFlip",
    "Normalize",
    "ToTensor",
]
