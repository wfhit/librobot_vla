"""
Unit tests for evaluation metrics.

Tests all metric classes in librobot/evaluation/metrics.py.
"""

import numpy as np
import pytest

from librobot.evaluation.metrics import (
    BENCHMARK_CONFIGS,
    MAE,
    MSE,
    EpisodeReturn,
    MetricBase,
    MetricCollection,
    PositionError,
    RotationError,
    Smoothness,
    SuccessRate,
    TrajectoryLength,
    create_metrics,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_predictions():
    """Create sample predictions array."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)


@pytest.fixture
def sample_targets():
    """Create sample targets array."""
    return np.array([1.1, 2.2, 2.9, 4.1, 4.8], dtype=np.float32)


@pytest.fixture
def sample_3d_predictions():
    """Create sample 3D position predictions."""
    return np.array([1.0, 2.0, 3.0], dtype=np.float32)


@pytest.fixture
def sample_3d_targets():
    """Create sample 3D position targets."""
    return np.array([1.1, 2.1, 3.1], dtype=np.float32)


@pytest.fixture
def sample_quaternion():
    """Create sample unit quaternion (w, x, y, z)."""
    # Normalized quaternion
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return q / np.linalg.norm(q)


@pytest.fixture
def sample_quaternion_rotated():
    """Create sample rotated unit quaternion."""
    # Small rotation around z-axis
    angle = 0.1  # radians
    q = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)], dtype=np.float32)
    return q / np.linalg.norm(q)


@pytest.fixture
def sample_trajectory():
    """Create sample trajectory."""
    # Simple linear trajectory in 3D
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def sample_actions():
    """Create sample action sequence."""
    return np.array(
        [
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def sample_rewards():
    """Create sample rewards sequence."""
    return np.array([1.0, 0.5, 0.8, 1.0, 0.2], dtype=np.float32)


# ============================================================================
# Test MetricBase
# ============================================================================


class TestMetricBase:
    """Tests for MetricBase class."""

    def test_initialization(self):
        """Test base metric initialization."""

        class ConcreteMetric(MetricBase):
            def update(self, *args, **kwargs):
                self._values.append(1.0)

        metric = ConcreteMetric("test_metric")

        assert metric.name == "test_metric"
        assert len(metric._values) == 0

    def test_reset(self):
        """Test metric reset."""

        class ConcreteMetric(MetricBase):
            def update(self, value):
                self._values.append(value)

        metric = ConcreteMetric("test")
        metric.update(1.0)
        metric.update(2.0)

        assert len(metric._values) == 2

        metric.reset()

        assert len(metric._values) == 0

    def test_compute_empty(self):
        """Test compute returns 0 when no values."""

        class ConcreteMetric(MetricBase):
            def update(self, value):
                self._values.append(value)

        metric = ConcreteMetric("test")

        assert metric.compute() == 0.0

    def test_compute_mean(self):
        """Test compute returns mean of values."""

        class ConcreteMetric(MetricBase):
            def update(self, value):
                self._values.append(value)

        metric = ConcreteMetric("test")
        metric.update(1.0)
        metric.update(2.0)
        metric.update(3.0)

        assert metric.compute() == 2.0

    def test_repr(self):
        """Test string representation."""

        class ConcreteMetric(MetricBase):
            def update(self, value):
                self._values.append(value)

        metric = ConcreteMetric("TestMetric")
        metric.update(0.1234)

        repr_str = repr(metric)

        assert "TestMetric" in repr_str
        assert "0.1234" in repr_str


# ============================================================================
# Test MSE Metric
# ============================================================================


class TestMSE:
    """Tests for MSE (Mean Squared Error) metric."""

    def test_initialization(self):
        """Test MSE initialization."""
        metric = MSE()

        assert metric.name == "MSE"

    def test_perfect_predictions(self):
        """Test MSE with perfect predictions (should be 0)."""
        metric = MSE()
        targets = np.array([1.0, 2.0, 3.0])

        metric.update(targets, targets)

        assert metric.compute() == 0.0

    def test_known_mse(self):
        """Test MSE with known values."""
        metric = MSE()
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([2.0, 3.0, 4.0])

        # Error = [1, 1, 1], squared = [1, 1, 1], mean = 1.0
        metric.update(predictions, targets)

        assert metric.compute() == 1.0

    def test_multiple_updates(self, sample_predictions, sample_targets):
        """Test MSE with multiple updates."""
        metric = MSE()

        metric.update(sample_predictions, sample_targets)
        metric.update(sample_predictions, sample_targets)

        # Both updates should have same MSE, so mean should equal single update
        expected_mse = np.mean((sample_predictions - sample_targets) ** 2)
        assert np.isclose(metric.compute(), expected_mse)

    def test_reset_clears_values(self, sample_predictions, sample_targets):
        """Test that reset clears accumulated values."""
        metric = MSE()
        metric.update(sample_predictions, sample_targets)

        metric.reset()

        assert metric.compute() == 0.0

    @pytest.mark.parametrize(
        "predictions,targets,expected_mse",
        [
            (np.array([0.0]), np.array([1.0]), 1.0),
            (np.array([0.0, 0.0]), np.array([2.0, 2.0]), 4.0),
            (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), 0.0),
        ],
    )
    def test_various_inputs(self, predictions, targets, expected_mse):
        """Test MSE with various inputs."""
        metric = MSE()
        metric.update(predictions, targets)

        assert np.isclose(metric.compute(), expected_mse)


# ============================================================================
# Test MAE Metric
# ============================================================================


class TestMAE:
    """Tests for MAE (Mean Absolute Error) metric."""

    def test_initialization(self):
        """Test MAE initialization."""
        metric = MAE()

        assert metric.name == "MAE"

    def test_perfect_predictions(self):
        """Test MAE with perfect predictions (should be 0)."""
        metric = MAE()
        targets = np.array([1.0, 2.0, 3.0])

        metric.update(targets, targets)

        assert metric.compute() == 0.0

    def test_known_mae(self):
        """Test MAE with known values."""
        metric = MAE()
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([2.0, 3.0, 4.0])

        # Error = [1, 1, 1], mean = 1.0
        metric.update(predictions, targets)

        assert metric.compute() == 1.0

    def test_negative_errors(self):
        """Test MAE handles negative errors correctly."""
        metric = MAE()
        predictions = np.array([3.0, 4.0, 5.0])
        targets = np.array([2.0, 3.0, 4.0])

        # Absolute error = [1, 1, 1], mean = 1.0
        metric.update(predictions, targets)

        assert metric.compute() == 1.0

    def test_multiple_updates(self, sample_predictions, sample_targets):
        """Test MAE with multiple updates."""
        metric = MAE()

        metric.update(sample_predictions, sample_targets)
        metric.update(sample_predictions, sample_targets)

        expected_mae = np.mean(np.abs(sample_predictions - sample_targets))
        assert np.isclose(metric.compute(), expected_mae)

    @pytest.mark.parametrize(
        "predictions,targets,expected_mae",
        [
            (np.array([0.0]), np.array([1.0]), 1.0),
            (np.array([0.0, 0.0]), np.array([2.0, 2.0]), 2.0),
            (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), 0.0),
            (np.array([5.0]), np.array([2.0]), 3.0),
        ],
    )
    def test_various_inputs(self, predictions, targets, expected_mae):
        """Test MAE with various inputs."""
        metric = MAE()
        metric.update(predictions, targets)

        assert np.isclose(metric.compute(), expected_mae)


# ============================================================================
# Test SuccessRate Metric
# ============================================================================


class TestSuccessRate:
    """Tests for SuccessRate metric."""

    def test_initialization_default_threshold(self):
        """Test SuccessRate initialization with default threshold."""
        metric = SuccessRate()

        assert metric.name == "Success Rate"
        assert metric.threshold == 0.05

    def test_initialization_custom_threshold(self):
        """Test SuccessRate initialization with custom threshold."""
        metric = SuccessRate(threshold=0.1)

        assert metric.threshold == 0.1

    def test_success_when_within_threshold(self):
        """Test success when achieved goal is within threshold."""
        metric = SuccessRate(threshold=0.1)
        achieved = np.array([1.0, 2.0, 3.0])
        desired = np.array([1.01, 2.01, 3.01])  # Very close

        metric.update(achieved, desired)

        assert metric.compute() == 1.0  # 100% success

    def test_failure_when_outside_threshold(self):
        """Test failure when achieved goal is outside threshold."""
        metric = SuccessRate(threshold=0.05)
        achieved = np.array([1.0, 2.0, 3.0])
        desired = np.array([2.0, 3.0, 4.0])  # Far away

        metric.update(achieved, desired)

        assert metric.compute() == 0.0  # 0% success

    def test_mixed_success_failure(self):
        """Test mixed success and failure."""
        metric = SuccessRate(threshold=0.1)

        # Success: distance ~0.017
        metric.update(np.array([1.0, 1.0, 1.0]), np.array([1.01, 1.01, 1.01]))

        # Failure: distance = sqrt(3) ~1.73
        metric.update(np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]))

        # 50% success rate
        assert metric.compute() == 0.5

    def test_exact_threshold_boundary(self):
        """Test behavior at exact threshold boundary."""
        metric = SuccessRate(threshold=0.1)
        achieved = np.array([0.0])
        desired = np.array([0.05])  # Distance = 0.05 < 0.1

        metric.update(achieved, desired)

        assert metric.compute() == 1.0

    @pytest.mark.parametrize(
        "threshold,distance,expected",
        [
            (0.1, 0.05, 1.0),  # Within threshold -> success
            (0.1, 0.15, 0.0),  # Outside threshold -> failure
            (0.5, 0.4, 1.0),  # Within larger threshold -> success
        ],
    )
    def test_various_thresholds(self, threshold, distance, expected):
        """Test various threshold scenarios."""
        metric = SuccessRate(threshold=threshold)
        achieved = np.array([0.0])
        desired = np.array([distance])

        metric.update(achieved, desired)

        assert metric.compute() == expected


# ============================================================================
# Test PositionError Metric
# ============================================================================


class TestPositionError:
    """Tests for PositionError metric."""

    def test_initialization(self):
        """Test PositionError initialization."""
        metric = PositionError()

        assert metric.name == "Position Error"

    def test_zero_error(self):
        """Test position error with same positions."""
        metric = PositionError()
        pos = np.array([1.0, 2.0, 3.0])

        metric.update(pos, pos)

        assert metric.compute() == 0.0

    def test_known_error(self):
        """Test position error with known values."""
        metric = PositionError()
        predicted = np.array([0.0, 0.0, 0.0])
        target = np.array([3.0, 4.0, 0.0])

        # Distance = sqrt(9 + 16) = 5.0
        metric.update(predicted, target)

        assert metric.compute() == 5.0

    def test_1d_position(self):
        """Test position error with 1D position."""
        metric = PositionError()
        predicted = np.array([0.0])
        target = np.array([5.0])

        metric.update(predicted, target)

        assert metric.compute() == 5.0

    def test_multiple_updates(self, sample_3d_predictions, sample_3d_targets):
        """Test position error with multiple updates."""
        metric = PositionError()

        metric.update(sample_3d_predictions, sample_3d_targets)
        metric.update(sample_3d_predictions, sample_3d_targets)

        expected_error = np.linalg.norm(sample_3d_predictions - sample_3d_targets)
        assert np.isclose(metric.compute(), expected_error)

    @pytest.mark.parametrize(
        "predicted,target,expected_error",
        [
            (np.array([0.0, 0.0]), np.array([3.0, 4.0]), 5.0),
            (np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0]), 0.0),
            (np.array([0.0]), np.array([1.0]), 1.0),
        ],
    )
    def test_various_positions(self, predicted, target, expected_error):
        """Test position error with various positions."""
        metric = PositionError()
        metric.update(predicted, target)

        assert np.isclose(metric.compute(), expected_error)


# ============================================================================
# Test RotationError Metric
# ============================================================================


class TestRotationError:
    """Tests for RotationError metric."""

    def test_initialization(self):
        """Test RotationError initialization."""
        metric = RotationError()

        assert metric.name == "Rotation Error"

    def test_zero_rotation_error(self, sample_quaternion):
        """Test rotation error with identical quaternions."""
        metric = RotationError()

        metric.update(sample_quaternion, sample_quaternion)

        assert np.isclose(metric.compute(), 0.0, atol=1e-5)

    def test_known_rotation_error(self):
        """Test rotation error with 90 degree rotation."""
        metric = RotationError()
        # Identity quaternion
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        # 90 degree rotation around z-axis
        q2 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])

        metric.update(q1, q2)

        # Should be approximately 90 degrees
        assert np.isclose(metric.compute(), 90.0, atol=1.0)

    def test_small_rotation(self, sample_quaternion, sample_quaternion_rotated):
        """Test rotation error with small rotation."""
        metric = RotationError()

        metric.update(sample_quaternion, sample_quaternion_rotated)

        # Should be a small angle (around 5.7 degrees for 0.1 radian rotation)
        assert metric.compute() < 10.0  # Less than 10 degrees

    def test_opposite_quaternions_same_rotation(self):
        """Test that opposite quaternions represent same rotation."""
        metric = RotationError()
        q1 = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = np.array([-1.0, 0.0, 0.0, 0.0])  # Same rotation

        metric.update(q1, q2)

        assert np.isclose(metric.compute(), 0.0, atol=1e-5)

    def test_multiple_updates(self, sample_quaternion, sample_quaternion_rotated):
        """Test rotation error with multiple updates."""
        metric = RotationError()

        metric.update(sample_quaternion, sample_quaternion_rotated)
        metric.update(sample_quaternion, sample_quaternion)  # Zero error

        # Mean of small error and zero
        assert metric.compute() < 10.0


# ============================================================================
# Test TrajectoryLength Metric
# ============================================================================


class TestTrajectoryLength:
    """Tests for TrajectoryLength metric."""

    def test_initialization(self):
        """Test TrajectoryLength initialization."""
        metric = TrajectoryLength()

        assert metric.name == "Trajectory Length"

    def test_straight_line_trajectory(self, sample_trajectory):
        """Test trajectory length of straight line."""
        metric = TrajectoryLength()

        metric.update(sample_trajectory)

        # 5 points, 4 segments of length 1.0 each
        assert metric.compute() == 4.0

    def test_single_point_trajectory(self):
        """Test trajectory with single point (no update)."""
        metric = TrajectoryLength()
        single_point = np.array([[0.0, 0.0, 0.0]])

        metric.update(single_point)

        # Length 1 trajectory shouldn't add anything
        assert metric.compute() == 0.0

    def test_two_point_trajectory(self):
        """Test trajectory with two points."""
        metric = TrajectoryLength()
        trajectory = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])

        metric.update(trajectory)

        # Distance = 5.0
        assert metric.compute() == 5.0

    def test_zigzag_trajectory(self):
        """Test zigzag trajectory length."""
        metric = TrajectoryLength()
        # Zigzag in 2D
        trajectory = np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 0.0],
                [3.0, 1.0],
            ]
        )

        metric.update(trajectory)

        # Each segment has length sqrt(2)
        expected_length = 3 * np.sqrt(2)
        assert np.isclose(metric.compute(), expected_length)

    def test_multiple_updates(self, sample_trajectory):
        """Test trajectory length with multiple updates."""
        metric = TrajectoryLength()

        metric.update(sample_trajectory)  # Length 4.0
        metric.update(sample_trajectory)  # Length 4.0

        # Mean = 4.0
        assert metric.compute() == 4.0


# ============================================================================
# Test Smoothness Metric
# ============================================================================


class TestSmoothness:
    """Tests for Smoothness metric."""

    def test_initialization(self):
        """Test Smoothness initialization."""
        metric = Smoothness()

        assert metric.name == "Smoothness"

    def test_constant_velocity(self):
        """Test smoothness of constant velocity motion (zero acceleration)."""
        metric = Smoothness()
        # Constant velocity: positions at 0, 1, 2, 3, 4
        actions = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])

        metric.update(actions)

        # Constant velocity means zero acceleration
        assert np.isclose(metric.compute(), 0.0, atol=1e-5)

    def test_jerky_motion(self):
        """Test smoothness of jerky motion."""
        metric = Smoothness()
        # Jerky motion with changing acceleration
        actions = np.array([[0.0], [1.0], [0.0], [1.0], [0.0]])

        metric.update(actions)

        # Non-zero smoothness (lower is smoother)
        assert metric.compute() > 0

    def test_short_sequence(self):
        """Test smoothness with sequence too short for second derivative."""
        metric = Smoothness()
        # Only 2 points - can't compute second derivative
        actions = np.array([[0.0], [1.0]])

        metric.update(actions)

        # Should not add any value
        assert metric.compute() == 0.0

    def test_multidimensional_actions(self, sample_actions):
        """Test smoothness with multidimensional actions."""
        metric = Smoothness()

        metric.update(sample_actions)

        # Linear increase has zero second derivative
        assert np.isclose(metric.compute(), 0.0, atol=1e-5)

    def test_multiple_updates(self, sample_actions):
        """Test smoothness with multiple updates."""
        metric = Smoothness()

        metric.update(sample_actions)  # Linear motion
        metric.update(sample_actions)

        # Both should have zero smoothness, mean = 0
        assert np.isclose(metric.compute(), 0.0, atol=1e-5)


# ============================================================================
# Test EpisodeReturn Metric
# ============================================================================


class TestEpisodeReturn:
    """Tests for EpisodeReturn metric."""

    def test_initialization_default_gamma(self):
        """Test EpisodeReturn initialization with default gamma."""
        metric = EpisodeReturn()

        assert metric.name == "Episode Return"
        assert metric.gamma == 0.99

    def test_initialization_custom_gamma(self):
        """Test EpisodeReturn initialization with custom gamma."""
        metric = EpisodeReturn(gamma=0.95)

        assert metric.gamma == 0.95

    def test_single_reward(self):
        """Test episode return with single reward."""
        metric = EpisodeReturn(gamma=0.99)
        rewards = np.array([1.0])

        metric.update(rewards)

        assert metric.compute() == 1.0

    def test_undiscounted_return(self):
        """Test episode return with gamma=1 (no discounting)."""
        metric = EpisodeReturn(gamma=1.0)
        rewards = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        metric.update(rewards)

        assert metric.compute() == 5.0

    def test_discounted_return(self):
        """Test episode return with discounting."""
        metric = EpisodeReturn(gamma=0.5)
        rewards = np.array([1.0, 1.0, 1.0])

        # Return = 1 + 0.5*1 + 0.25*1 = 1.75
        metric.update(rewards)

        assert np.isclose(metric.compute(), 1.75)

    def test_zero_gamma(self):
        """Test episode return with gamma=0 (only first reward counts)."""
        metric = EpisodeReturn(gamma=0.0)
        rewards = np.array([2.0, 5.0, 10.0])

        metric.update(rewards)

        # Only first reward counts with gamma=0
        assert metric.compute() == 2.0

    def test_multiple_episodes(self, sample_rewards):
        """Test episode return with multiple episodes."""
        metric = EpisodeReturn(gamma=0.99)

        metric.update(sample_rewards)
        metric.update(sample_rewards)

        # Mean of two identical episodes
        T = len(sample_rewards)
        discounts = 0.99 ** np.arange(T)
        expected = np.sum(sample_rewards * discounts)
        assert np.isclose(metric.compute(), expected)

    @pytest.mark.parametrize(
        "gamma,rewards,expected",
        [
            (1.0, np.array([1.0, 1.0, 1.0]), 3.0),
            (0.0, np.array([5.0, 1.0, 1.0]), 5.0),
            (0.5, np.array([4.0, 2.0]), 5.0),  # 4 + 0.5*2
        ],
    )
    def test_various_scenarios(self, gamma, rewards, expected):
        """Test episode return with various scenarios."""
        metric = EpisodeReturn(gamma=gamma)
        metric.update(rewards)

        assert np.isclose(metric.compute(), expected)


# ============================================================================
# Test MetricCollection
# ============================================================================


class TestMetricCollection:
    """Tests for MetricCollection class."""

    def test_initialization_default_metrics(self):
        """Test MetricCollection initialization with default metrics."""
        collection = MetricCollection()

        assert len(collection.metrics) == 4
        metric_names = [m.name for m in collection.metrics]
        assert "MSE" in metric_names
        assert "MAE" in metric_names
        assert "Success Rate" in metric_names
        assert "Smoothness" in metric_names

    def test_initialization_custom_metrics(self):
        """Test MetricCollection initialization with custom metrics."""
        metrics = [MSE(), MAE()]
        collection = MetricCollection(metrics)

        assert len(collection.metrics) == 2
        assert collection.metrics[0].name == "MSE"
        assert collection.metrics[1].name == "MAE"

    def test_reset_all_metrics(self, sample_predictions, sample_targets):
        """Test that reset clears all metrics."""
        collection = MetricCollection([MSE(), MAE()])

        collection.metrics[0].update(sample_predictions, sample_targets)
        collection.metrics[1].update(sample_predictions, sample_targets)

        collection.reset()

        for metric in collection.metrics:
            assert metric.compute() == 0.0

    def test_compute_all_metrics(self, sample_predictions, sample_targets):
        """Test computing all metrics."""
        collection = MetricCollection([MSE(), MAE()])

        collection.metrics[0].update(sample_predictions, sample_targets)
        collection.metrics[1].update(sample_predictions, sample_targets)

        results = collection.compute()

        assert "MSE" in results
        assert "MAE" in results
        # Check that values are numeric (could be numpy float or python float)
        assert np.issubdtype(type(results["MSE"]), np.floating) or isinstance(results["MSE"], float)
        assert np.issubdtype(type(results["MAE"]), np.floating) or isinstance(results["MAE"], float)

    def test_getitem(self):
        """Test accessing metric by name."""
        collection = MetricCollection([MSE(), MAE()])

        mse = collection["MSE"]
        mae = collection["MAE"]

        assert isinstance(mse, MSE)
        assert isinstance(mae, MAE)

    def test_getitem_missing_raises_error(self):
        """Test that accessing nonexistent metric raises KeyError."""
        collection = MetricCollection([MSE()])

        with pytest.raises(KeyError):
            _ = collection["NonexistentMetric"]

    def test_update_with_matching_kwargs(self, sample_predictions, sample_targets):
        """Test update passes correct kwargs to metrics."""
        collection = MetricCollection([MSE(), MAE()])

        collection.update(predictions=sample_predictions, targets=sample_targets)

        results = collection.compute()

        # Both MSE and MAE should have been updated
        assert results["MSE"] > 0
        assert results["MAE"] > 0

    def test_update_ignores_incompatible_metrics(self):
        """Test that update ignores metrics that can't accept kwargs."""
        collection = MetricCollection([MSE(), Smoothness()])

        # This should update MSE but ignore Smoothness (different signature)
        predictions = np.array([1.0, 2.0, 3.0])
        targets = np.array([1.1, 2.1, 3.1])

        collection.update(predictions=predictions, targets=targets)

        results = collection.compute()

        # MSE should be updated, Smoothness should be 0 (not updated)
        assert results["MSE"] > 0
        assert results["Smoothness"] == 0.0


# ============================================================================
# Test create_metrics Factory Function
# ============================================================================


class TestCreateMetrics:
    """Tests for create_metrics factory function."""

    def test_create_single_metric(self):
        """Test creating collection with single metric."""
        collection = create_metrics(["mse"])

        assert len(collection.metrics) == 1
        assert collection.metrics[0].name == "MSE"

    def test_create_multiple_metrics(self):
        """Test creating collection with multiple metrics."""
        collection = create_metrics(["mse", "mae", "success_rate"])

        assert len(collection.metrics) == 3
        metric_names = [m.name for m in collection.metrics]
        assert "MSE" in metric_names
        assert "MAE" in metric_names
        assert "Success Rate" in metric_names

    def test_create_all_metrics(self):
        """Test creating collection with all available metrics."""
        all_metric_names = [
            "mse",
            "mae",
            "success_rate",
            "position_error",
            "rotation_error",
            "trajectory_length",
            "smoothness",
            "episode_return",
        ]
        collection = create_metrics(all_metric_names)

        assert len(collection.metrics) == 8

    def test_case_insensitive(self):
        """Test that metric names are case insensitive."""
        collection1 = create_metrics(["MSE", "MAE"])
        collection2 = create_metrics(["mse", "mae"])

        assert len(collection1.metrics) == len(collection2.metrics)

    def test_unknown_metric_ignored(self):
        """Test that unknown metric names are ignored."""
        collection = create_metrics(["mse", "unknown_metric", "mae"])

        assert len(collection.metrics) == 2
        metric_names = [m.name for m in collection.metrics]
        assert "MSE" in metric_names
        assert "MAE" in metric_names

    def test_empty_list_returns_defaults(self):
        """Test creating collection with empty list returns default metrics."""
        # create_metrics passes empty list to MetricCollection, which uses defaults
        collection = create_metrics([])

        # MetricCollection uses default metrics when given empty list
        assert len(collection.metrics) == 4  # Default: MSE, MAE, SuccessRate, Smoothness

    @pytest.mark.parametrize(
        "metric_name,expected_class",
        [
            ("mse", MSE),
            ("mae", MAE),
            ("success_rate", SuccessRate),
            ("position_error", PositionError),
            ("rotation_error", RotationError),
            ("trajectory_length", TrajectoryLength),
            ("smoothness", Smoothness),
            ("episode_return", EpisodeReturn),
        ],
    )
    def test_individual_metrics(self, metric_name, expected_class):
        """Test creating individual metrics by name."""
        collection = create_metrics([metric_name])

        assert len(collection.metrics) == 1
        assert isinstance(collection.metrics[0], expected_class)


# ============================================================================
# Test BENCHMARK_CONFIGS
# ============================================================================


class TestBenchmarkConfigs:
    """Tests for BENCHMARK_CONFIGS constant."""

    def test_bridge_config_exists(self):
        """Test that bridge benchmark config exists."""
        assert "bridge" in BENCHMARK_CONFIGS

    def test_simpler_config_exists(self):
        """Test that simpler benchmark config exists."""
        assert "simpler" in BENCHMARK_CONFIGS

    def test_libero_config_exists(self):
        """Test that libero benchmark config exists."""
        assert "libero" in BENCHMARK_CONFIGS

    def test_bridge_config_structure(self):
        """Test bridge config has required fields."""
        config = BENCHMARK_CONFIGS["bridge"]

        assert "tasks" in config
        assert "metrics" in config
        assert "num_episodes" in config
        assert isinstance(config["tasks"], list)
        assert isinstance(config["metrics"], list)
        assert isinstance(config["num_episodes"], int)

    def test_simpler_config_structure(self):
        """Test simpler config has required fields."""
        config = BENCHMARK_CONFIGS["simpler"]

        assert "tasks" in config
        assert "metrics" in config
        assert "num_episodes" in config

    def test_libero_config_structure(self):
        """Test libero config has required fields."""
        config = BENCHMARK_CONFIGS["libero"]

        assert "tasks" in config
        assert "metrics" in config
        assert "num_episodes" in config


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_large_values(self):
        """Test metrics with very large values."""
        metric = MSE()
        predictions = np.array([1e10, 2e10])
        targets = np.array([1e10 + 1, 2e10 + 1])

        metric.update(predictions, targets)

        assert np.isfinite(metric.compute())

    def test_very_small_values(self):
        """Test metrics with very small values."""
        metric = MSE()
        predictions = np.array([1e-10, 2e-10])
        targets = np.array([1.1e-10, 2.1e-10])

        metric.update(predictions, targets)

        assert np.isfinite(metric.compute())
        assert metric.compute() >= 0

    def test_negative_values(self):
        """Test metrics with negative values."""
        metric = MAE()
        predictions = np.array([-5.0, -3.0, -1.0])
        targets = np.array([-4.0, -2.0, 0.0])

        metric.update(predictions, targets)

        # MAE should always be non-negative
        assert metric.compute() >= 0

    def test_single_element_arrays(self):
        """Test metrics with single element arrays."""
        mse = MSE()
        mae = MAE()

        predictions = np.array([1.0])
        targets = np.array([2.0])

        mse.update(predictions, targets)
        mae.update(predictions, targets)

        assert mse.compute() == 1.0
        assert mae.compute() == 1.0

    def test_high_dimensional_data(self):
        """Test position error with high dimensional data."""
        metric = PositionError()
        predicted = np.zeros(100)
        target = np.ones(100)

        metric.update(predicted, target)

        # Distance should be sqrt(100) = 10
        assert np.isclose(metric.compute(), 10.0)

    def test_long_trajectory(self):
        """Test trajectory length with many points."""
        metric = TrajectoryLength()
        # 1000 points along a line
        trajectory = np.zeros((1000, 3))
        trajectory[:, 0] = np.linspace(0, 100, 1000)

        metric.update(trajectory)

        # Total length should be approximately 100
        assert np.isclose(metric.compute(), 100.0, rtol=0.01)

    def test_metrics_are_independent(self):
        """Test that different metric instances don't share state."""
        mse1 = MSE()
        mse2 = MSE()

        mse1.update(np.array([1.0]), np.array([2.0]))

        assert mse1.compute() == 1.0
        assert mse2.compute() == 0.0  # Should be independent
