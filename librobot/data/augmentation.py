"""Data augmentation strategies for VLA training."""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from librobot.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation.
    
    Args:
        image_augmentations: List of image augmentation names
        action_augmentations: List of action augmentation names
        state_augmentations: List of state augmentation names
        probability: Global probability of applying augmentations
        seed: Random seed for reproducibility
    """
    image_augmentations: List[str] = field(default_factory=lambda: ["color_jitter", "random_crop"])
    action_augmentations: List[str] = field(default_factory=list)
    state_augmentations: List[str] = field(default_factory=list)
    probability: float = 0.5
    seed: Optional[int] = None


class AbstractAugmentation(ABC):
    """Abstract base class for augmentations."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    @abstractmethod
    def __call__(self, data: Any) -> Any:
        """Apply augmentation."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


# =============================================================================
# Image Augmentations
# =============================================================================

class ColorJitter(AbstractAugmentation):
    """Random color jitter augmentation."""
    
    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        p: float = 0.5,
    ):
        super().__init__(p)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return image
        
        try:
            import cv2
            
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            
            # Adjust hue
            if self.hue > 0:
                hsv[:, :, 0] += random.uniform(-self.hue, self.hue) * 180
                hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 180)
            
            # Adjust saturation
            if self.saturation > 0:
                hsv[:, :, 1] *= random.uniform(1 - self.saturation, 1 + self.saturation)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            
            # Adjust brightness
            if self.brightness > 0:
                hsv[:, :, 2] *= random.uniform(1 - self.brightness, 1 + self.brightness)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
            
            # Convert back to RGB
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
            # Adjust contrast
            if self.contrast > 0:
                factor = random.uniform(1 - self.contrast, 1 + self.contrast)
                mean = image.mean()
                image = np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)
            
            return image
            
        except ImportError:
            return image


class RandomCrop(AbstractAugmentation):
    """Random crop augmentation."""
    
    def __init__(
        self,
        crop_size: Union[int, Tuple[int, int]] = 224,
        scale: Tuple[float, float] = (0.8, 1.0),
        ratio: Tuple[float, float] = (0.75, 1.33),
        p: float = 0.5,
    ):
        super().__init__(p)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        self.crop_size = crop_size
        self.scale = scale
        self.ratio = ratio
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return image
        
        h, w = image.shape[:2]
        target_h, target_w = self.crop_size
        
        # Calculate crop area
        area = h * w
        target_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
        
        new_w = int(round(np.sqrt(target_area * aspect_ratio)))
        new_h = int(round(np.sqrt(target_area / aspect_ratio)))
        
        new_w = min(new_w, w)
        new_h = min(new_h, h)
        
        # Random position
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        
        # Crop
        cropped = image[top:top + new_h, left:left + new_w]
        
        # Resize to target size
        try:
            import cv2
            resized = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            return resized
        except ImportError:
            return cropped


class RandomFlip(AbstractAugmentation):
    """Random horizontal/vertical flip."""
    
    def __init__(
        self,
        horizontal: bool = True,
        vertical: bool = False,
        p: float = 0.5,
    ):
        super().__init__(p)
        self.horizontal = horizontal
        self.vertical = vertical
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return image
        
        if self.horizontal and random.random() < 0.5:
            image = np.fliplr(image).copy()
        
        if self.vertical and random.random() < 0.5:
            image = np.flipud(image).copy()
        
        return image


class RandomRotation(AbstractAugmentation):
    """Random rotation augmentation."""
    
    def __init__(
        self,
        degrees: float = 10.0,
        p: float = 0.5,
    ):
        super().__init__(p)
        self.degrees = degrees
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return image
        
        try:
            import cv2
            
            h, w = image.shape[:2]
            angle = random.uniform(-self.degrees, self.degrees)
            
            # Rotation matrix
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            
            return rotated
            
        except ImportError:
            return image


class GaussianNoise(AbstractAugmentation):
    """Add Gaussian noise to image."""
    
    def __init__(
        self,
        mean: float = 0.0,
        std: float = 0.05,
        p: float = 0.5,
    ):
        super().__init__(p)
        self.mean = mean
        self.std = std
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return image
        
        noise = np.random.normal(self.mean, self.std, image.shape)
        noisy = image.astype(np.float32) + noise * 255
        return np.clip(noisy, 0, 255).astype(np.uint8)


class GaussianBlur(AbstractAugmentation):
    """Apply Gaussian blur."""
    
    def __init__(
        self,
        kernel_size: Tuple[int, int] = (5, 9),
        sigma: Tuple[float, float] = (0.1, 2.0),
        p: float = 0.5,
    ):
        super().__init__(p)
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return image
        
        try:
            import cv2
            
            ksize = random.choice(range(self.kernel_size[0], self.kernel_size[1] + 1, 2))
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            
            return cv2.GaussianBlur(image, (ksize, ksize), sigma)
            
        except ImportError:
            return image


class Normalize(AbstractAugmentation):
    """Normalize image to [0, 1] or with mean/std."""
    
    def __init__(
        self,
        mean: Optional[Tuple[float, float, float]] = None,
        std: Optional[Tuple[float, float, float]] = None,
        p: float = 1.0,
    ):
        super().__init__(p)
        self.mean = np.array(mean) if mean else np.array([0.485, 0.456, 0.406])
        self.std = np.array(std) if std else np.array([0.229, 0.224, 0.225])
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        # Convert to float [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Normalize
        image = (image - self.mean) / self.std
        
        return image


class CutOut(AbstractAugmentation):
    """Random cutout augmentation."""
    
    def __init__(
        self,
        num_holes: int = 1,
        max_h_size: int = 32,
        max_w_size: int = 32,
        fill_value: int = 0,
        p: float = 0.5,
    ):
        super().__init__(p)
        self.num_holes = num_holes
        self.max_h_size = max_h_size
        self.max_w_size = max_w_size
        self.fill_value = fill_value
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return image
        
        h, w = image.shape[:2]
        image = image.copy()
        
        for _ in range(self.num_holes):
            y = random.randint(0, h)
            x = random.randint(0, w)
            
            y1 = max(0, y - self.max_h_size // 2)
            y2 = min(h, y + self.max_h_size // 2)
            x1 = max(0, x - self.max_w_size // 2)
            x2 = min(w, x + self.max_w_size // 2)
            
            image[y1:y2, x1:x2] = self.fill_value
        
        return image


# =============================================================================
# Action Augmentations
# =============================================================================

class ActionNoise(AbstractAugmentation):
    """Add noise to actions."""
    
    def __init__(
        self,
        noise_std: float = 0.01,
        p: float = 0.5,
    ):
        super().__init__(p)
        self.noise_std = noise_std
    
    def __call__(self, actions: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return actions
        
        noise = np.random.normal(0, self.noise_std, actions.shape)
        return actions + noise


class ActionScaling(AbstractAugmentation):
    """Scale actions randomly."""
    
    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        p: float = 0.5,
    ):
        super().__init__(p)
        self.scale_range = scale_range
    
    def __call__(self, actions: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return actions
        
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        return actions * scale


class ActionMixup(AbstractAugmentation):
    """Mixup augmentation for action sequences."""
    
    def __init__(
        self,
        alpha: float = 0.2,
        p: float = 0.5,
    ):
        super().__init__(p)
        self.alpha = alpha
    
    def __call__(self, actions_pair: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        if random.random() > self.p:
            return actions_pair[0]
        
        actions1, actions2 = actions_pair
        lam = np.random.beta(self.alpha, self.alpha)
        return lam * actions1 + (1 - lam) * actions2


# =============================================================================
# State Augmentations
# =============================================================================

class StateNoise(AbstractAugmentation):
    """Add noise to robot state."""
    
    def __init__(
        self,
        position_std: float = 0.001,
        velocity_std: float = 0.01,
        p: float = 0.5,
    ):
        super().__init__(p)
        self.position_std = position_std
        self.velocity_std = velocity_std
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return state
        
        # Assume first half is position, second half is velocity
        mid = len(state) // 2
        noise = np.zeros_like(state)
        noise[:mid] = np.random.normal(0, self.position_std, mid)
        noise[mid:] = np.random.normal(0, self.velocity_std, len(state) - mid)
        
        return state + noise


class StateDropout(AbstractAugmentation):
    """Randomly drop state dimensions."""
    
    def __init__(
        self,
        dropout_rate: float = 0.1,
        p: float = 0.5,
    ):
        super().__init__(p)
        self.dropout_rate = dropout_rate
    
    def __call__(self, state: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return state
        
        mask = np.random.random(state.shape) > self.dropout_rate
        return state * mask


# =============================================================================
# Composition and Pipeline
# =============================================================================

class Compose:
    """Compose multiple augmentations."""
    
    def __init__(self, augmentations: List[AbstractAugmentation]):
        self.augmentations = augmentations
    
    def __call__(self, data: Any) -> Any:
        for aug in self.augmentations:
            data = aug(data)
        return data
    
    def __repr__(self) -> str:
        augs_str = ", ".join(repr(aug) for aug in self.augmentations)
        return f"Compose([{augs_str}])"


class RandomChoice:
    """Randomly choose one augmentation from a list."""
    
    def __init__(self, augmentations: List[AbstractAugmentation]):
        self.augmentations = augmentations
    
    def __call__(self, data: Any) -> Any:
        aug = random.choice(self.augmentations)
        return aug(data)


class OneOf:
    """Apply one of the augmentations with given probability."""
    
    def __init__(
        self,
        augmentations: List[AbstractAugmentation],
        p: float = 0.5,
    ):
        self.augmentations = augmentations
        self.p = p
    
    def __call__(self, data: Any) -> Any:
        if random.random() > self.p:
            return data
        return random.choice(self.augmentations)(data)


class VLADataAugmentation:
    """Complete data augmentation pipeline for VLA training.
    
    Applies augmentations to images, actions, and states.
    
    Example:
        >>> augmenter = VLADataAugmentation(
        ...     image_augs=["color_jitter", "random_crop", "gaussian_blur"],
        ...     action_augs=["noise"],
        ...     state_augs=["noise"],
        ... )
        >>> augmented = augmenter({"image": img, "actions": act, "state": state})
    """
    
    def __init__(
        self,
        image_augs: Optional[List[str]] = None,
        action_augs: Optional[List[str]] = None,
        state_augs: Optional[List[str]] = None,
        p: float = 0.5,
    ):
        self.p = p
        
        # Build image augmentation pipeline
        image_aug_map = {
            "color_jitter": ColorJitter(p=0.8),
            "random_crop": RandomCrop(p=0.8),
            "random_flip": RandomFlip(p=0.5),
            "random_rotation": RandomRotation(p=0.3),
            "gaussian_noise": GaussianNoise(p=0.3),
            "gaussian_blur": GaussianBlur(p=0.3),
            "cutout": CutOut(p=0.3),
        }
        
        image_augs = image_augs or ["color_jitter", "random_crop"]
        self.image_pipeline = Compose([
            image_aug_map[name] for name in image_augs if name in image_aug_map
        ])
        
        # Build action augmentation pipeline
        action_aug_map = {
            "noise": ActionNoise(p=0.5),
            "scaling": ActionScaling(p=0.5),
        }
        
        action_augs = action_augs or []
        self.action_pipeline = Compose([
            action_aug_map[name] for name in action_augs if name in action_aug_map
        ])
        
        # Build state augmentation pipeline
        state_aug_map = {
            "noise": StateNoise(p=0.5),
            "dropout": StateDropout(p=0.3),
        }
        
        state_augs = state_augs or []
        self.state_pipeline = Compose([
            state_aug_map[name] for name in state_augs if name in state_aug_map
        ])
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Apply augmentations to a sample."""
        if random.random() > self.p:
            return sample
        
        result = sample.copy()
        
        # Augment image
        if "image" in result:
            result["image"] = self.image_pipeline(result["image"])
        
        if "images" in result:
            result["images"] = [self.image_pipeline(img) for img in result["images"]]
        
        # Augment actions
        if "actions" in result:
            result["actions"] = self.action_pipeline(result["actions"])
        
        # Augment state
        if "state" in result:
            result["state"] = self.state_pipeline(result["state"])
        
        if "proprioception" in result:
            result["proprioception"] = self.state_pipeline(result["proprioception"])
        
        return result


def create_augmentation_pipeline(
    config: Optional[AugmentationConfig] = None,
    **kwargs,
) -> VLADataAugmentation:
    """
    Factory function to create augmentation pipeline.
    
    Args:
        config: Augmentation configuration
        **kwargs: Override configuration parameters
        
    Returns:
        Configured augmentation pipeline
    """
    if config is None:
        config = AugmentationConfig(**kwargs)
    
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
    
    return VLADataAugmentation(
        image_augs=config.image_augmentations,
        action_augs=config.action_augmentations,
        state_augs=config.state_augmentations,
        p=config.probability,
    )


def get_default_train_augmentations() -> VLADataAugmentation:
    """Get default training augmentations."""
    return VLADataAugmentation(
        image_augs=["color_jitter", "random_crop", "gaussian_noise"],
        action_augs=["noise"],
        state_augs=["noise"],
        p=0.5,
    )


def get_strong_augmentations() -> VLADataAugmentation:
    """Get strong augmentations for improved generalization."""
    return VLADataAugmentation(
        image_augs=[
            "color_jitter", "random_crop", "random_rotation",
            "gaussian_noise", "gaussian_blur", "cutout"
        ],
        action_augs=["noise", "scaling"],
        state_augs=["noise", "dropout"],
        p=0.7,
    )


__all__ = [
    "AugmentationConfig",
    "AbstractAugmentation",
    "ColorJitter",
    "RandomCrop",
    "RandomFlip",
    "RandomRotation",
    "GaussianNoise",
    "GaussianBlur",
    "Normalize",
    "CutOut",
    "ActionNoise",
    "ActionScaling",
    "ActionMixup",
    "StateNoise",
    "StateDropout",
    "Compose",
    "RandomChoice",
    "OneOf",
    "VLADataAugmentation",
    "create_augmentation_pipeline",
    "get_default_train_augmentations",
    "get_strong_augmentations",
]
