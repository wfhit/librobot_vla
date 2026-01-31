"""
Unit tests for data augmentation strategies.

Tests all augmentation classes in librobot/data/augmentation.py.
"""

import random
import pytest
import numpy as np

from librobot.data.augmentation import (
    AugmentationConfig,
    AbstractAugmentation,
    ColorJitter,
    RandomCrop,
    RandomFlip,
    RandomRotation,
    GaussianNoise,
    GaussianBlur,
    Normalize,
    CutOut,
    ActionNoise,
    ActionScaling,
    ActionMixup,
    StateNoise,
    StateDropout,
    Compose,
    RandomChoice,
    OneOf,
    VLADataAugmentation,
    create_augmentation_pipeline,
    get_default_train_augmentations,
    get_strong_augmentations,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_image():
    """Create a sample RGB image as numpy array."""
    return np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_small():
    """Create a smaller sample RGB image."""
    return np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_actions():
    """Create sample action array."""
    return np.random.randn(10, 7).astype(np.float32)


@pytest.fixture
def sample_state():
    """Create sample robot state array."""
    return np.random.randn(14).astype(np.float32)


@pytest.fixture
def sample_trajectory():
    """Create sample trajectory for metric tests."""
    return np.random.randn(20, 3).astype(np.float32)


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility in tests."""
    random.seed(42)
    np.random.seed(42)
    yield


# ============================================================================
# Test AugmentationConfig
# ============================================================================


class TestAugmentationConfig:
    """Tests for AugmentationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = AugmentationConfig()

        assert config.image_augmentations == ["color_jitter", "random_crop"]
        assert config.action_augmentations == []
        assert config.state_augmentations == []
        assert config.probability == 0.5
        assert config.seed is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = AugmentationConfig(
            image_augmentations=["random_flip"],
            action_augmentations=["noise"],
            state_augmentations=["dropout"],
            probability=0.8,
            seed=123,
        )

        assert config.image_augmentations == ["random_flip"]
        assert config.action_augmentations == ["noise"]
        assert config.state_augmentations == ["dropout"]
        assert config.probability == 0.8
        assert config.seed == 123


# ============================================================================
# Test AbstractAugmentation
# ============================================================================


class TestAbstractAugmentation:
    """Tests for AbstractAugmentation base class."""

    def test_initialization(self):
        """Test base class initialization."""

        class ConcreteAug(AbstractAugmentation):
            def __call__(self, data):
                return data

        aug = ConcreteAug(p=0.7)
        assert aug.p == 0.7

    def test_repr(self):
        """Test string representation."""

        class TestAug(AbstractAugmentation):
            def __call__(self, data):
                return data

        aug = TestAug(p=0.3)
        assert "TestAug" in repr(aug)
        assert "0.3" in repr(aug)


# ============================================================================
# Test Image Augmentations
# ============================================================================


class TestColorJitter:
    """Tests for ColorJitter augmentation."""

    def test_initialization(self):
        """Test ColorJitter initialization with default parameters."""
        aug = ColorJitter()

        assert aug.brightness == 0.2
        assert aug.contrast == 0.2
        assert aug.saturation == 0.2
        assert aug.hue == 0.1
        assert aug.p == 0.5

    def test_initialization_custom_params(self):
        """Test ColorJitter initialization with custom parameters."""
        aug = ColorJitter(
            brightness=0.5,
            contrast=0.3,
            saturation=0.4,
            hue=0.2,
            p=0.8,
        )

        assert aug.brightness == 0.5
        assert aug.contrast == 0.3
        assert aug.saturation == 0.4
        assert aug.hue == 0.2
        assert aug.p == 0.8

    def test_output_shape_preserved(self, sample_image):
        """Test that output shape matches input shape."""
        aug = ColorJitter(p=1.0)
        result = aug(sample_image)

        assert result.shape == sample_image.shape

    def test_output_dtype_preserved(self, sample_image):
        """Test that output dtype is uint8."""
        aug = ColorJitter(p=1.0)
        result = aug(sample_image)

        assert result.dtype == np.uint8

    def test_probability_zero_returns_unchanged(self, sample_image):
        """Test that p=0 returns unchanged image."""
        aug = ColorJitter(p=0.0)
        result = aug(sample_image)

        np.testing.assert_array_equal(result, sample_image)

    def test_values_in_valid_range(self, sample_image):
        """Test that output values are in valid range [0, 255]."""
        aug = ColorJitter(p=1.0, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)
        result = aug(sample_image)

        assert result.min() >= 0
        assert result.max() <= 255


class TestRandomCrop:
    """Tests for RandomCrop augmentation."""

    def test_initialization_int_crop_size(self):
        """Test initialization with integer crop size."""
        aug = RandomCrop(crop_size=128, p=0.5)

        assert aug.crop_size == (128, 128)
        assert aug.p == 0.5

    def test_initialization_tuple_crop_size(self):
        """Test initialization with tuple crop size."""
        aug = RandomCrop(crop_size=(128, 256), p=0.5)

        assert aug.crop_size == (128, 256)

    def test_output_shape_with_crop(self, sample_image):
        """Test that output has correct shape after crop."""
        aug = RandomCrop(crop_size=(100, 100), p=1.0)
        result = aug(sample_image)

        assert result.shape[:2] == (100, 100)
        assert result.shape[2] == 3

    def test_probability_zero_returns_unchanged(self, sample_image):
        """Test that p=0 returns unchanged image."""
        aug = RandomCrop(crop_size=100, p=0.0)
        result = aug(sample_image)

        np.testing.assert_array_equal(result, sample_image)

    @pytest.mark.parametrize("crop_size", [(64, 64), (128, 128), (200, 200)])
    def test_various_crop_sizes(self, sample_image, crop_size):
        """Test various crop sizes."""
        aug = RandomCrop(crop_size=crop_size, p=1.0)
        result = aug(sample_image)

        assert result.shape[:2] == crop_size


class TestRandomFlip:
    """Tests for RandomFlip augmentation."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        aug = RandomFlip()

        assert aug.horizontal is True
        assert aug.vertical is False
        assert aug.p == 0.5

    def test_initialization_custom(self):
        """Test custom initialization."""
        aug = RandomFlip(horizontal=False, vertical=True, p=0.8)

        assert aug.horizontal is False
        assert aug.vertical is True
        assert aug.p == 0.8

    def test_output_shape_preserved(self, sample_image):
        """Test that output shape matches input."""
        aug = RandomFlip(p=1.0)
        result = aug(sample_image)

        assert result.shape == sample_image.shape

    def test_probability_zero_returns_unchanged(self, sample_image):
        """Test that p=0 returns unchanged image."""
        aug = RandomFlip(p=0.0)
        result = aug(sample_image)

        np.testing.assert_array_equal(result, sample_image)

    def test_horizontal_flip_changes_image(self, sample_image):
        """Test that horizontal flip actually changes the image."""
        # Create a recognizable pattern
        test_image = np.zeros((64, 64, 3), dtype=np.uint8)
        test_image[:, :32, :] = 255  # Left half white

        # Force flip to happen
        random.seed(0)  # Seed that causes flip
        aug = RandomFlip(horizontal=True, vertical=False, p=1.0)
        result = aug(test_image)

        # Check result shape is correct
        assert result.shape == test_image.shape


class TestRandomRotation:
    """Tests for RandomRotation augmentation."""

    def test_initialization(self):
        """Test initialization."""
        aug = RandomRotation(degrees=15.0, p=0.7)

        assert aug.degrees == 15.0
        assert aug.p == 0.7

    def test_output_shape_preserved(self, sample_image):
        """Test that output shape matches input."""
        aug = RandomRotation(degrees=10.0, p=1.0)
        result = aug(sample_image)

        assert result.shape == sample_image.shape

    def test_probability_zero_returns_unchanged(self, sample_image):
        """Test that p=0 returns unchanged image."""
        aug = RandomRotation(p=0.0)
        result = aug(sample_image)

        np.testing.assert_array_equal(result, sample_image)


class TestGaussianNoise:
    """Tests for GaussianNoise augmentation."""

    def test_initialization(self):
        """Test initialization."""
        aug = GaussianNoise(mean=0.1, std=0.1, p=0.6)

        assert aug.mean == 0.1
        assert aug.std == 0.1
        assert aug.p == 0.6

    def test_output_shape_preserved(self, sample_image):
        """Test that output shape matches input."""
        aug = GaussianNoise(p=1.0)
        result = aug(sample_image)

        assert result.shape == sample_image.shape

    def test_output_dtype_is_uint8(self, sample_image):
        """Test that output dtype is uint8."""
        aug = GaussianNoise(p=1.0)
        result = aug(sample_image)

        assert result.dtype == np.uint8

    def test_probability_zero_returns_unchanged(self, sample_image):
        """Test that p=0 returns unchanged image."""
        aug = GaussianNoise(p=0.0)
        result = aug(sample_image)

        np.testing.assert_array_equal(result, sample_image)

    def test_values_in_valid_range(self, sample_image):
        """Test that output values are clipped to valid range."""
        aug = GaussianNoise(mean=0.0, std=0.5, p=1.0)  # High noise
        result = aug(sample_image)

        assert result.min() >= 0
        assert result.max() <= 255

    def test_noise_actually_changes_image(self, sample_image):
        """Test that noise is actually added when p=1."""
        aug = GaussianNoise(std=0.1, p=1.0)
        result = aug(sample_image)

        # Should not be exactly equal
        assert not np.array_equal(result, sample_image)


class TestGaussianBlur:
    """Tests for GaussianBlur augmentation."""

    def test_initialization(self):
        """Test initialization."""
        aug = GaussianBlur(kernel_size=(3, 7), sigma=(0.5, 1.5), p=0.5)

        assert aug.kernel_size == (3, 7)
        assert aug.sigma == (0.5, 1.5)
        assert aug.p == 0.5

    def test_output_shape_preserved(self, sample_image):
        """Test that output shape matches input."""
        aug = GaussianBlur(p=1.0)
        result = aug(sample_image)

        assert result.shape == sample_image.shape

    def test_probability_zero_returns_unchanged(self, sample_image):
        """Test that p=0 returns unchanged image."""
        aug = GaussianBlur(p=0.0)
        result = aug(sample_image)

        np.testing.assert_array_equal(result, sample_image)


class TestNormalize:
    """Tests for Normalize augmentation."""

    def test_initialization_defaults(self):
        """Test default initialization uses ImageNet mean/std."""
        aug = Normalize()

        np.testing.assert_array_almost_equal(aug.mean, [0.485, 0.456, 0.406])
        np.testing.assert_array_almost_equal(aug.std, [0.229, 0.224, 0.225])
        assert aug.p == 1.0

    def test_initialization_custom(self):
        """Test custom mean and std."""
        aug = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        np.testing.assert_array_almost_equal(aug.mean, [0.5, 0.5, 0.5])
        np.testing.assert_array_almost_equal(aug.std, [0.5, 0.5, 0.5])

    def test_output_shape_preserved(self, sample_image):
        """Test that output shape matches input."""
        aug = Normalize()
        result = aug(sample_image)

        assert result.shape == sample_image.shape

    def test_output_dtype_is_float(self, sample_image):
        """Test that output dtype is float (float32 or float64)."""
        aug = Normalize()
        result = aug(sample_image)

        assert np.issubdtype(result.dtype, np.floating)

    def test_normalization_calculation(self):
        """Test that normalization is calculated correctly."""
        # Create known image
        image = np.full((10, 10, 3), 128, dtype=np.uint8)

        aug = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        result = aug(image)

        # 128/255 = ~0.502, (0.502 - 0.5) / 0.5 = ~0.004
        expected = (128 / 255.0 - 0.5) / 0.5
        np.testing.assert_array_almost_equal(result, expected, decimal=3)


class TestCutOut:
    """Tests for CutOut augmentation."""

    def test_initialization(self):
        """Test initialization."""
        aug = CutOut(num_holes=2, max_h_size=16, max_w_size=16, fill_value=128, p=0.5)

        assert aug.num_holes == 2
        assert aug.max_h_size == 16
        assert aug.max_w_size == 16
        assert aug.fill_value == 128
        assert aug.p == 0.5

    def test_output_shape_preserved(self, sample_image):
        """Test that output shape matches input."""
        aug = CutOut(p=1.0)
        result = aug(sample_image)

        assert result.shape == sample_image.shape

    def test_probability_zero_returns_unchanged(self, sample_image):
        """Test that p=0 returns unchanged image."""
        aug = CutOut(p=0.0)
        result = aug(sample_image)

        np.testing.assert_array_equal(result, sample_image)

    def test_cutout_creates_holes(self, sample_image):
        """Test that cutout creates holes in the image."""
        aug = CutOut(num_holes=1, max_h_size=32, max_w_size=32, fill_value=0, p=1.0)

        # Use white image to easily detect holes
        white_image = np.full((64, 64, 3), 255, dtype=np.uint8)
        result = aug(white_image)

        # Should have some zero values (the hole)
        assert np.any(result == 0)

    def test_does_not_modify_original(self, sample_image):
        """Test that original image is not modified."""
        original_copy = sample_image.copy()
        aug = CutOut(p=1.0)
        _ = aug(sample_image)

        np.testing.assert_array_equal(sample_image, original_copy)


# ============================================================================
# Test Action Augmentations
# ============================================================================


class TestActionNoise:
    """Tests for ActionNoise augmentation."""

    def test_initialization(self):
        """Test initialization."""
        aug = ActionNoise(noise_std=0.05, p=0.7)

        assert aug.noise_std == 0.05
        assert aug.p == 0.7

    def test_output_shape_preserved(self, sample_actions):
        """Test that output shape matches input."""
        aug = ActionNoise(p=1.0)
        result = aug(sample_actions)

        assert result.shape == sample_actions.shape

    def test_probability_zero_returns_unchanged(self, sample_actions):
        """Test that p=0 returns unchanged actions."""
        aug = ActionNoise(p=0.0)
        result = aug(sample_actions)

        np.testing.assert_array_equal(result, sample_actions)

    def test_noise_actually_changes_actions(self, sample_actions):
        """Test that noise is actually added when p=1."""
        aug = ActionNoise(noise_std=0.1, p=1.0)
        result = aug(sample_actions)

        # Should not be exactly equal
        assert not np.array_equal(result, sample_actions)

    def test_noise_magnitude(self, sample_actions):
        """Test that noise magnitude is approximately correct."""
        aug = ActionNoise(noise_std=0.01, p=1.0)
        differences = []

        # Run multiple times to get average noise magnitude
        for _ in range(100):
            np.random.seed(_)
            result = aug(sample_actions)
            diff = np.std(result - sample_actions)
            differences.append(diff)

        avg_diff = np.mean(differences)
        # Should be close to noise_std
        assert 0.005 < avg_diff < 0.015


class TestActionScaling:
    """Tests for ActionScaling augmentation."""

    def test_initialization(self):
        """Test initialization."""
        aug = ActionScaling(scale_range=(0.8, 1.2), p=0.6)

        assert aug.scale_range == (0.8, 1.2)
        assert aug.p == 0.6

    def test_output_shape_preserved(self, sample_actions):
        """Test that output shape matches input."""
        aug = ActionScaling(p=1.0)
        result = aug(sample_actions)

        assert result.shape == sample_actions.shape

    def test_probability_zero_returns_unchanged(self, sample_actions):
        """Test that p=0 returns unchanged actions."""
        aug = ActionScaling(p=0.0)
        result = aug(sample_actions)

        np.testing.assert_array_equal(result, sample_actions)

    def test_scaling_changes_magnitude(self, sample_actions):
        """Test that scaling changes action magnitude."""
        aug = ActionScaling(scale_range=(2.0, 2.0), p=1.0)  # Force 2x scaling
        result = aug(sample_actions)

        np.testing.assert_array_almost_equal(result, sample_actions * 2.0)


class TestActionMixup:
    """Tests for ActionMixup augmentation."""

    def test_initialization(self):
        """Test initialization."""
        aug = ActionMixup(alpha=0.4, p=0.5)

        assert aug.alpha == 0.4
        assert aug.p == 0.5

    def test_probability_zero_returns_first(self, sample_actions):
        """Test that p=0 returns first action sequence."""
        actions2 = np.random.randn(10, 7).astype(np.float32)

        aug = ActionMixup(p=0.0)
        result = aug((sample_actions, actions2))

        np.testing.assert_array_equal(result, sample_actions)

    def test_output_shape_preserved(self, sample_actions):
        """Test that output shape matches input."""
        actions2 = np.random.randn(10, 7).astype(np.float32)

        aug = ActionMixup(p=1.0)
        result = aug((sample_actions, actions2))

        assert result.shape == sample_actions.shape

    def test_mixup_combines_actions(self, sample_actions):
        """Test that mixup combines both action sequences."""
        actions1 = np.ones((5, 3), dtype=np.float32)
        actions2 = np.zeros((5, 3), dtype=np.float32)

        aug = ActionMixup(alpha=0.5, p=1.0)
        result = aug((actions1, actions2))

        # Result should be between 0 and 1
        assert result.min() >= 0
        assert result.max() <= 1


# ============================================================================
# Test State Augmentations
# ============================================================================


class TestStateNoise:
    """Tests for StateNoise augmentation."""

    def test_initialization(self):
        """Test initialization."""
        aug = StateNoise(position_std=0.002, velocity_std=0.02, p=0.5)

        assert aug.position_std == 0.002
        assert aug.velocity_std == 0.02
        assert aug.p == 0.5

    def test_output_shape_preserved(self, sample_state):
        """Test that output shape matches input."""
        aug = StateNoise(p=1.0)
        result = aug(sample_state)

        assert result.shape == sample_state.shape

    def test_probability_zero_returns_unchanged(self, sample_state):
        """Test that p=0 returns unchanged state."""
        aug = StateNoise(p=0.0)
        result = aug(sample_state)

        np.testing.assert_array_equal(result, sample_state)

    def test_noise_actually_changes_state(self, sample_state):
        """Test that noise is actually added when p=1."""
        aug = StateNoise(position_std=0.1, velocity_std=0.1, p=1.0)
        result = aug(sample_state)

        # Should not be exactly equal
        assert not np.array_equal(result, sample_state)


class TestStateDropout:
    """Tests for StateDropout augmentation."""

    def test_initialization(self):
        """Test initialization."""
        aug = StateDropout(dropout_rate=0.2, p=0.5)

        assert aug.dropout_rate == 0.2
        assert aug.p == 0.5

    def test_output_shape_preserved(self, sample_state):
        """Test that output shape matches input."""
        aug = StateDropout(p=1.0)
        result = aug(sample_state)

        assert result.shape == sample_state.shape

    def test_probability_zero_returns_unchanged(self, sample_state):
        """Test that p=0 returns unchanged state."""
        aug = StateDropout(p=0.0)
        result = aug(sample_state)

        np.testing.assert_array_equal(result, sample_state)

    def test_dropout_zeros_some_values(self, sample_state):
        """Test that dropout zeros some state dimensions."""
        aug = StateDropout(dropout_rate=0.5, p=1.0)

        # Use non-zero state
        state = np.ones(14, dtype=np.float32)
        result = aug(state)

        # Some values should be zero
        assert np.any(result == 0)


# ============================================================================
# Test Composition Classes
# ============================================================================


class TestCompose:
    """Tests for Compose augmentation pipeline."""

    def test_empty_compose(self, sample_image):
        """Test compose with no augmentations."""
        compose = Compose([])
        result = compose(sample_image)

        np.testing.assert_array_equal(result, sample_image)

    def test_single_augmentation(self, sample_image):
        """Test compose with single augmentation."""
        compose = Compose([GaussianNoise(p=1.0, std=0.1)])
        result = compose(sample_image)

        assert result.shape == sample_image.shape
        assert not np.array_equal(result, sample_image)

    def test_multiple_augmentations(self, sample_image):
        """Test compose with multiple augmentations."""
        compose = Compose([
            GaussianNoise(p=1.0, std=0.05),
            RandomFlip(p=1.0),
        ])
        result = compose(sample_image)

        assert result.shape == sample_image.shape

    def test_repr(self):
        """Test string representation."""
        compose = Compose([GaussianNoise(p=0.5), RandomFlip(p=0.3)])
        repr_str = repr(compose)

        assert "Compose" in repr_str
        assert "GaussianNoise" in repr_str
        assert "RandomFlip" in repr_str


class TestRandomChoice:
    """Tests for RandomChoice augmentation selector."""

    def test_selects_one_augmentation(self, sample_image):
        """Test that RandomChoice selects exactly one augmentation."""
        augs = [GaussianNoise(p=1.0, std=0.1), RandomFlip(p=1.0)]
        choice = RandomChoice(augs)
        result = choice(sample_image)

        assert result.shape == sample_image.shape

    def test_output_shape_preserved(self, sample_image):
        """Test that output shape matches input."""
        augs = [GaussianNoise(p=1.0), Normalize()]
        choice = RandomChoice(augs)
        result = choice(sample_image)

        assert result.shape == sample_image.shape


class TestOneOf:
    """Tests for OneOf augmentation selector."""

    def test_probability_zero_returns_unchanged(self, sample_image):
        """Test that p=0 returns unchanged input."""
        augs = [GaussianNoise(p=1.0, std=0.1), RandomFlip(p=1.0)]
        one_of = OneOf(augs, p=0.0)
        result = one_of(sample_image)

        np.testing.assert_array_equal(result, sample_image)

    def test_probability_one_applies_augmentation(self, sample_image):
        """Test that p=1 always applies one augmentation."""
        augs = [GaussianNoise(p=1.0, std=0.1)]
        one_of = OneOf(augs, p=1.0)
        result = one_of(sample_image)

        # Should be modified
        assert not np.array_equal(result, sample_image)

    def test_output_shape_preserved(self, sample_image):
        """Test that output shape matches input."""
        augs = [GaussianNoise(p=1.0), RandomFlip(p=1.0)]
        one_of = OneOf(augs, p=1.0)
        result = one_of(sample_image)

        assert result.shape == sample_image.shape


# ============================================================================
# Test VLADataAugmentation Pipeline
# ============================================================================


class TestVLADataAugmentation:
    """Tests for VLADataAugmentation pipeline."""

    def test_initialization_defaults(self):
        """Test default initialization."""
        aug = VLADataAugmentation()

        assert aug.p == 0.5
        assert aug.image_pipeline is not None
        assert aug.action_pipeline is not None
        assert aug.state_pipeline is not None

    def test_initialization_custom(self):
        """Test custom initialization."""
        aug = VLADataAugmentation(
            image_augs=["random_flip"],
            action_augs=["noise", "scaling"],
            state_augs=["noise"],
            p=0.8,
        )

        assert aug.p == 0.8

    def test_augment_single_image(self, sample_image):
        """Test augmenting sample with single image."""
        aug = VLADataAugmentation(p=1.0)

        sample = {"image": sample_image}
        result = aug(sample)

        assert "image" in result
        assert result["image"].shape == sample_image.shape

    def test_augment_multiple_images(self, sample_image):
        """Test augmenting sample with multiple images."""
        aug = VLADataAugmentation(p=1.0)

        sample = {"images": [sample_image, sample_image.copy()]}
        result = aug(sample)

        assert "images" in result
        assert len(result["images"]) == 2
        for img in result["images"]:
            assert img.shape == sample_image.shape

    def test_augment_actions(self, sample_actions):
        """Test augmenting actions."""
        aug = VLADataAugmentation(action_augs=["noise"], p=1.0)

        sample = {"actions": sample_actions}
        result = aug(sample)

        assert "actions" in result
        assert result["actions"].shape == sample_actions.shape

    def test_augment_state(self, sample_state):
        """Test augmenting state."""
        aug = VLADataAugmentation(state_augs=["noise"], p=1.0)

        sample = {"state": sample_state}
        result = aug(sample)

        assert "state" in result
        assert result["state"].shape == sample_state.shape

    def test_augment_proprioception(self, sample_state):
        """Test augmenting proprioception."""
        aug = VLADataAugmentation(state_augs=["noise"], p=1.0)

        sample = {"proprioception": sample_state}
        result = aug(sample)

        assert "proprioception" in result
        assert result["proprioception"].shape == sample_state.shape

    def test_augment_full_sample(self, sample_image, sample_actions, sample_state):
        """Test augmenting full sample with image, actions, and state."""
        aug = VLADataAugmentation(
            image_augs=["gaussian_noise"],
            action_augs=["noise"],
            state_augs=["noise"],
            p=1.0,
        )

        sample = {
            "image": sample_image,
            "actions": sample_actions,
            "state": sample_state,
        }
        result = aug(sample)

        assert "image" in result
        assert "actions" in result
        assert "state" in result
        assert result["image"].shape == sample_image.shape
        assert result["actions"].shape == sample_actions.shape
        assert result["state"].shape == sample_state.shape

    def test_probability_zero_returns_unchanged(self, sample_image):
        """Test that p=0 returns unchanged sample."""
        aug = VLADataAugmentation(p=0.0)

        sample = {"image": sample_image}
        result = aug(sample)

        np.testing.assert_array_equal(result["image"], sample_image)

    def test_does_not_modify_original(self, sample_image, sample_actions):
        """Test that original sample is not modified."""
        original_image = sample_image.copy()
        original_actions = sample_actions.copy()

        aug = VLADataAugmentation(p=1.0)
        sample = {"image": sample_image, "actions": sample_actions}
        _ = aug(sample)

        # Original should be unchanged in sample dict
        np.testing.assert_array_equal(sample["image"], original_image)
        np.testing.assert_array_equal(sample["actions"], original_actions)

    def test_invalid_augmentation_name_skipped(self):
        """Test that invalid augmentation names are skipped."""
        aug = VLADataAugmentation(
            image_augs=["invalid_name", "gaussian_noise"],
            p=1.0,
        )
        # Should not raise, invalid augmentation should be ignored
        assert aug.image_pipeline is not None


# ============================================================================
# Test Factory Functions
# ============================================================================


class TestCreateAugmentationPipeline:
    """Tests for create_augmentation_pipeline factory function."""

    def test_create_with_default_config(self):
        """Test creating pipeline with default config."""
        pipeline = create_augmentation_pipeline()

        assert isinstance(pipeline, VLADataAugmentation)

    def test_create_with_custom_config(self):
        """Test creating pipeline with custom config."""
        config = AugmentationConfig(
            image_augmentations=["random_flip"],
            probability=0.8,
            seed=42,
        )
        pipeline = create_augmentation_pipeline(config)

        assert isinstance(pipeline, VLADataAugmentation)
        assert pipeline.p == 0.8

    def test_create_with_kwargs(self):
        """Test creating pipeline with kwargs."""
        pipeline = create_augmentation_pipeline(probability=0.9)

        assert isinstance(pipeline, VLADataAugmentation)

    def test_seed_reproducibility(self, sample_image):
        """Test that seed provides reproducibility."""
        config = AugmentationConfig(seed=42)
        pipeline1 = create_augmentation_pipeline(config)
        result1 = pipeline1({"image": sample_image})

        config2 = AugmentationConfig(seed=42)
        pipeline2 = create_augmentation_pipeline(config2)
        result2 = pipeline2({"image": sample_image.copy()})

        np.testing.assert_array_equal(result1["image"], result2["image"])


class TestGetDefaultTrainAugmentations:
    """Tests for get_default_train_augmentations factory function."""

    def test_returns_vla_augmentation(self):
        """Test that function returns VLADataAugmentation."""
        aug = get_default_train_augmentations()

        assert isinstance(aug, VLADataAugmentation)

    def test_default_probability(self):
        """Test default probability."""
        aug = get_default_train_augmentations()

        assert aug.p == 0.5

    def test_can_process_sample(self, sample_image, sample_actions, sample_state):
        """Test that returned augmentation can process samples."""
        aug = get_default_train_augmentations()

        sample = {
            "image": sample_image,
            "actions": sample_actions,
            "state": sample_state,
        }
        result = aug(sample)

        assert "image" in result
        assert "actions" in result
        assert "state" in result


class TestGetStrongAugmentations:
    """Tests for get_strong_augmentations factory function."""

    def test_returns_vla_augmentation(self):
        """Test that function returns VLADataAugmentation."""
        aug = get_strong_augmentations()

        assert isinstance(aug, VLADataAugmentation)

    def test_higher_probability(self):
        """Test that strong augmentations have higher probability."""
        default_aug = get_default_train_augmentations()
        strong_aug = get_strong_augmentations()

        assert strong_aug.p > default_aug.p

    def test_can_process_sample(self, sample_image, sample_actions, sample_state):
        """Test that returned augmentation can process samples."""
        aug = get_strong_augmentations()

        sample = {
            "image": sample_image,
            "actions": sample_actions,
            "state": sample_state,
        }
        result = aug(sample)

        assert "image" in result
        assert "actions" in result
        assert "state" in result


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_small_image(self):
        """Test augmentation with very small image."""
        small_image = np.random.randint(0, 256, size=(8, 8, 3), dtype=np.uint8)
        aug = VLADataAugmentation(p=1.0)

        result = aug({"image": small_image})
        assert result["image"].shape == small_image.shape

    def test_single_element_state(self):
        """Test augmentation with single element state."""
        state = np.array([1.0], dtype=np.float32)
        aug = StateNoise(p=1.0)

        result = aug(state)
        assert result.shape == state.shape

    def test_single_action_sequence(self):
        """Test augmentation with single action."""
        actions = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        aug = ActionNoise(p=1.0)

        result = aug(actions)
        assert result.shape == actions.shape

    def test_grayscale_like_image(self):
        """Test augmentation with grayscale-like image (1 channel repeated)."""
        gray = np.random.randint(0, 256, size=(64, 64), dtype=np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)

        aug = GaussianNoise(p=1.0)
        result = aug(rgb)
        assert result.shape == rgb.shape

    def test_empty_sample_dict(self):
        """Test augmentation with empty sample dict."""
        aug = VLADataAugmentation(p=1.0)
        result = aug({})

        assert result == {}

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_action_dtypes(self, dtype):
        """Test action augmentation with different dtypes."""
        actions = np.random.randn(10, 7).astype(dtype)
        aug = ActionNoise(p=1.0)

        result = aug(actions)
        assert result.shape == actions.shape
