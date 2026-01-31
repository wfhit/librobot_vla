"""
Unit tests for the checkpoint utilities module.

Tests checkpoint saving, loading, and management functionality.
"""

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from librobot.utils.checkpoint import (
    Checkpoint,
    save_checkpoint,
    load_checkpoint,
)


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return nn.Linear(10, 5)


@pytest.fixture
def model_and_optimizer():
    """Create a model with optimizer for testing."""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer


@pytest.fixture
def checkpoint_dir(tmp_path):
    """Create a temporary checkpoint directory."""
    return tmp_path / "checkpoints"


class TestCheckpointClass:
    """Test suite for Checkpoint class."""

    def test_checkpoint_initialization(self, checkpoint_dir):
        """Test Checkpoint initialization."""
        checkpoint = Checkpoint(save_dir=checkpoint_dir)

        assert checkpoint.save_dir == checkpoint_dir
        assert checkpoint_dir.exists()

    def test_checkpoint_with_keep_last_n(self, checkpoint_dir):
        """Test Checkpoint with keep_last_n option."""
        checkpoint = Checkpoint(save_dir=checkpoint_dir, keep_last_n=5)

        assert checkpoint.keep_last_n == 5

    def test_checkpoint_with_save_best(self, checkpoint_dir):
        """Test Checkpoint with save_best option."""
        checkpoint = Checkpoint(
            save_dir=checkpoint_dir,
            save_best=True,
            metric_name="loss",
            mode="min",
        )

        assert checkpoint.save_best is True
        assert checkpoint.metric_name == "loss"
        assert checkpoint.mode == "min"

    def test_checkpoint_save_basic(self, checkpoint_dir, simple_model):
        """Test basic checkpoint saving."""
        checkpoint = Checkpoint(save_dir=checkpoint_dir)

        filepath = checkpoint.save(simple_model, epoch=1)

        assert filepath.exists()
        assert "epoch_1" in filepath.name

    def test_checkpoint_save_with_optimizer(self, checkpoint_dir, model_and_optimizer):
        """Test saving checkpoint with optimizer."""
        model, optimizer = model_and_optimizer
        checkpoint = Checkpoint(save_dir=checkpoint_dir)

        filepath = checkpoint.save(model, optimizer=optimizer, epoch=1)

        # Load and verify optimizer state is saved
        data = torch.load(filepath, weights_only=False)
        assert "optimizer_state_dict" in data

    def test_checkpoint_save_with_scheduler(self, checkpoint_dir, model_and_optimizer):
        """Test saving checkpoint with scheduler."""
        model, optimizer = model_and_optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        checkpoint = Checkpoint(save_dir=checkpoint_dir)

        filepath = checkpoint.save(model, optimizer=optimizer, scheduler=scheduler, epoch=1)

        data = torch.load(filepath, weights_only=False)
        assert "scheduler_state_dict" in data

    def test_checkpoint_save_with_metrics(self, checkpoint_dir, simple_model):
        """Test saving checkpoint with metrics."""
        checkpoint = Checkpoint(save_dir=checkpoint_dir)
        metrics = {"loss": 0.5, "accuracy": 0.9}

        filepath = checkpoint.save(simple_model, epoch=1, metrics=metrics)

        data = torch.load(filepath, weights_only=False)
        assert data["metrics"]["loss"] == 0.5
        assert data["metrics"]["accuracy"] == 0.9

    def test_checkpoint_save_with_metadata(self, checkpoint_dir, simple_model):
        """Test saving checkpoint with custom metadata."""
        checkpoint = Checkpoint(save_dir=checkpoint_dir)
        metadata = {"experiment": "test", "version": "1.0"}

        filepath = checkpoint.save(simple_model, epoch=1, metadata=metadata)

        data = torch.load(filepath, weights_only=False)
        assert data["metadata"]["experiment"] == "test"

    def test_checkpoint_save_with_step(self, checkpoint_dir, simple_model):
        """Test saving checkpoint with step number."""
        checkpoint = Checkpoint(save_dir=checkpoint_dir)

        filepath = checkpoint.save(simple_model, step=1000)

        assert "step_1000" in filepath.name

    def test_checkpoint_save_custom_filename(self, checkpoint_dir, simple_model):
        """Test saving with custom filename."""
        checkpoint = Checkpoint(save_dir=checkpoint_dir)

        filepath = checkpoint.save(simple_model, filename="custom_name.pt")

        assert filepath.name == "custom_name.pt"

    def test_checkpoint_creates_metadata_json(self, checkpoint_dir, simple_model):
        """Test that saving creates accompanying JSON metadata."""
        checkpoint = Checkpoint(save_dir=checkpoint_dir)

        filepath = checkpoint.save(simple_model, epoch=1, metrics={"loss": 0.5})

        json_path = filepath.with_suffix(".json")
        assert json_path.exists()

        with open(json_path) as f:
            metadata = json.load(f)
        assert metadata["epoch"] == 1
        assert metadata["metrics"]["loss"] == 0.5

    def test_checkpoint_load(self, checkpoint_dir, simple_model):
        """Test loading checkpoint."""
        checkpoint = Checkpoint(save_dir=checkpoint_dir)
        checkpoint.save(simple_model, epoch=1)

        data = checkpoint.load("checkpoint_epoch_1.pt")

        assert "model_state_dict" in data
        assert data["epoch"] == 1

    def test_checkpoint_load_into_model(self, checkpoint_dir, simple_model):
        """Test loading checkpoint into model."""
        checkpoint = Checkpoint(save_dir=checkpoint_dir)

        # Save original weights
        original_weights = simple_model.weight.clone()
        checkpoint.save(simple_model, epoch=1)

        # Modify model
        simple_model.weight.data.fill_(0)

        # Load checkpoint into model
        new_model = nn.Linear(10, 5)
        checkpoint.load("checkpoint_epoch_1.pt", model=new_model)

        assert torch.allclose(new_model.weight, original_weights)

    def test_checkpoint_load_into_optimizer(self, checkpoint_dir, model_and_optimizer):
        """Test loading checkpoint into optimizer."""
        model, optimizer = model_and_optimizer
        checkpoint = Checkpoint(save_dir=checkpoint_dir)

        # Run a step to modify optimizer state
        loss = model(torch.randn(1, 10)).sum()
        loss.backward()
        optimizer.step()

        checkpoint.save(model, optimizer=optimizer, epoch=1)

        # Create new optimizer and load
        new_optimizer = torch.optim.Adam(model.parameters())
        checkpoint.load("checkpoint_epoch_1.pt", optimizer=new_optimizer)

        # Optimizer state should be loaded
        assert len(new_optimizer.state) > 0

    def test_checkpoint_load_nonexistent_raises(self, checkpoint_dir):
        """Test loading nonexistent checkpoint raises error."""
        checkpoint = Checkpoint(save_dir=checkpoint_dir)

        with pytest.raises(FileNotFoundError):
            checkpoint.load("nonexistent.pt")

    def test_checkpoint_save_best(self, checkpoint_dir, simple_model):
        """Test that best checkpoint is saved."""
        checkpoint = Checkpoint(
            save_dir=checkpoint_dir,
            save_best=True,
            metric_name="loss",
            mode="min",
        )

        checkpoint.save(simple_model, epoch=1, metrics={"loss": 1.0})
        checkpoint.save(simple_model, epoch=2, metrics={"loss": 0.5})
        checkpoint.save(simple_model, epoch=3, metrics={"loss": 0.8})

        # Best checkpoint should exist
        best_path = checkpoint_dir / "best.pt"
        assert best_path.exists()

        # Best should be epoch 2 (lowest loss)
        data = checkpoint.load("best.pt")
        assert data["epoch"] == 2

    def test_checkpoint_save_best_max_mode(self, checkpoint_dir, simple_model):
        """Test best checkpoint with max mode."""
        checkpoint = Checkpoint(
            save_dir=checkpoint_dir,
            save_best=True,
            metric_name="accuracy",
            mode="max",
        )

        checkpoint.save(simple_model, epoch=1, metrics={"accuracy": 0.8})
        checkpoint.save(simple_model, epoch=2, metrics={"accuracy": 0.9})
        checkpoint.save(simple_model, epoch=3, metrics={"accuracy": 0.85})

        # Best should be epoch 2 (highest accuracy)
        data = checkpoint.load("best.pt")
        assert data["epoch"] == 2

    def test_checkpoint_load_best(self, checkpoint_dir, simple_model):
        """Test load_best method."""
        checkpoint = Checkpoint(
            save_dir=checkpoint_dir,
            save_best=True,
            metric_name="loss",
            mode="min",
        )

        checkpoint.save(simple_model, epoch=1, metrics={"loss": 1.0})
        checkpoint.save(simple_model, epoch=2, metrics={"loss": 0.5})

        data = checkpoint.load_best()

        assert data["epoch"] == 2

    def test_checkpoint_load_latest(self, checkpoint_dir, simple_model):
        """Test load_latest method."""
        checkpoint = Checkpoint(save_dir=checkpoint_dir)

        checkpoint.save(simple_model, epoch=1)
        checkpoint.save(simple_model, epoch=2)
        checkpoint.save(simple_model, epoch=3)

        data = checkpoint.load_latest()

        assert data["epoch"] == 3

    def test_checkpoint_load_latest_empty_raises(self, checkpoint_dir):
        """Test load_latest with no checkpoints raises error."""
        checkpoint = Checkpoint(save_dir=checkpoint_dir)

        with pytest.raises(FileNotFoundError):
            checkpoint.load_latest()

    def test_checkpoint_keep_last_n(self, checkpoint_dir, simple_model):
        """Test that only last N checkpoints are kept."""
        checkpoint = Checkpoint(save_dir=checkpoint_dir, keep_last_n=3)

        for i in range(5):
            checkpoint.save(simple_model, epoch=i)

        # Only 3 checkpoints should remain
        pt_files = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        assert len(pt_files) == 3

        # Should be epochs 2, 3, 4
        epochs = [int(f.stem.split("_")[-1]) for f in pt_files]
        assert sorted(epochs) == [2, 3, 4]

    def test_checkpoint_list_checkpoints(self, checkpoint_dir, simple_model):
        """Test listing checkpoints."""
        checkpoint = Checkpoint(save_dir=checkpoint_dir)

        checkpoint.save(simple_model, epoch=1, metrics={"loss": 1.0})
        checkpoint.save(simple_model, epoch=2, metrics={"loss": 0.5})

        info_list = checkpoint.list_checkpoints()

        assert len(info_list) == 2
        assert info_list[0]["epoch"] == 1
        assert info_list[1]["epoch"] == 2

    def test_checkpoint_timestamp_saved(self, checkpoint_dir, simple_model):
        """Test that timestamp is saved in checkpoint."""
        checkpoint = Checkpoint(save_dir=checkpoint_dir)

        filepath = checkpoint.save(simple_model, epoch=1)
        data = torch.load(filepath, weights_only=False)

        assert "timestamp" in data


class TestSaveCheckpointFunction:
    """Test suite for save_checkpoint convenience function."""

    def test_save_checkpoint_basic(self, tmp_path, simple_model):
        """Test basic save_checkpoint usage."""
        filepath = tmp_path / "checkpoint.pt"

        save_checkpoint(filepath, simple_model)

        assert filepath.exists()

    def test_save_checkpoint_with_optimizer(self, tmp_path, model_and_optimizer):
        """Test save_checkpoint with optimizer."""
        model, optimizer = model_and_optimizer
        filepath = tmp_path / "checkpoint.pt"

        save_checkpoint(filepath, model, optimizer=optimizer)

        data = torch.load(filepath, weights_only=False)
        assert "optimizer_state_dict" in data

    def test_save_checkpoint_with_kwargs(self, tmp_path, simple_model):
        """Test save_checkpoint with additional kwargs."""
        filepath = tmp_path / "checkpoint.pt"

        save_checkpoint(filepath, simple_model, epoch=10, extra_data="test")

        data = torch.load(filepath, weights_only=False)
        assert data["epoch"] == 10
        assert data["extra_data"] == "test"

    def test_save_checkpoint_creates_dirs(self, tmp_path, simple_model):
        """Test save_checkpoint creates parent directories."""
        filepath = tmp_path / "nested" / "dir" / "checkpoint.pt"

        save_checkpoint(filepath, simple_model)

        assert filepath.exists()


class TestLoadCheckpointFunction:
    """Test suite for load_checkpoint convenience function."""

    def test_load_checkpoint_basic(self, tmp_path, simple_model):
        """Test basic load_checkpoint usage."""
        filepath = tmp_path / "checkpoint.pt"
        torch.save({"model_state_dict": simple_model.state_dict()}, filepath)

        data = load_checkpoint(filepath)

        assert "model_state_dict" in data

    def test_load_checkpoint_into_model(self, tmp_path, simple_model):
        """Test load_checkpoint into model."""
        filepath = tmp_path / "checkpoint.pt"
        original_weights = simple_model.weight.clone()
        torch.save({"model_state_dict": simple_model.state_dict()}, filepath)

        new_model = nn.Linear(10, 5)
        load_checkpoint(filepath, model=new_model)

        assert torch.allclose(new_model.weight, original_weights)

    def test_load_checkpoint_into_optimizer(self, tmp_path, model_and_optimizer):
        """Test load_checkpoint into optimizer."""
        model, optimizer = model_and_optimizer
        filepath = tmp_path / "checkpoint.pt"

        # Create some optimizer state
        loss = model(torch.randn(1, 10)).sum()
        loss.backward()
        optimizer.step()

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            filepath,
        )

        new_optimizer = torch.optim.Adam(model.parameters())
        load_checkpoint(filepath, optimizer=new_optimizer)

        assert len(new_optimizer.state) > 0

    def test_load_checkpoint_map_location(self, tmp_path, simple_model):
        """Test load_checkpoint with map_location."""
        filepath = tmp_path / "checkpoint.pt"
        torch.save({"model_state_dict": simple_model.state_dict()}, filepath)

        data = load_checkpoint(filepath, map_location="cpu")

        assert data is not None


class TestCheckpointRoundtrip:
    """Test complete checkpoint save/load roundtrips."""

    def test_model_roundtrip(self, tmp_path):
        """Test model weights are preserved through save/load."""
        model1 = nn.Linear(100, 50)
        filepath = tmp_path / "roundtrip.pt"

        save_checkpoint(filepath, model1)

        model2 = nn.Linear(100, 50)
        load_checkpoint(filepath, model=model2)

        # Weights should match
        assert torch.allclose(model1.weight, model2.weight)
        assert torch.allclose(model1.bias, model2.bias)

    def test_training_state_roundtrip(self, tmp_path):
        """Test complete training state roundtrip."""
        model = nn.Linear(10, 5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        # Simulate training step
        for _ in range(3):
            loss = model(torch.randn(4, 10)).sum()
            loss.backward()
            optimizer.step()
            scheduler.step()

        checkpoint = Checkpoint(save_dir=tmp_path)
        checkpoint.save(
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=3,
            metrics={"loss": 0.1},
        )

        # Load into new instances
        new_model = nn.Linear(10, 5)
        new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)
        new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=1)

        data = checkpoint.load_latest(
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
        )

        assert data["epoch"] == 3
        assert data["metrics"]["loss"] == 0.1
        assert torch.allclose(model.weight, new_model.weight)


class TestCheckpointEdgeCases:
    """Test edge cases and error handling."""

    def test_checkpoint_with_string_path(self, tmp_path, simple_model):
        """Test Checkpoint with string path."""
        checkpoint = Checkpoint(save_dir=str(tmp_path / "checkpoints"))

        filepath = checkpoint.save(simple_model, epoch=1)

        assert filepath.exists()

    def test_checkpoint_absolute_path_load(self, checkpoint_dir, simple_model):
        """Test loading with absolute path."""
        checkpoint = Checkpoint(save_dir=checkpoint_dir)
        filepath = checkpoint.save(simple_model, epoch=1)

        # Load using absolute path
        data = checkpoint.load(filepath)

        assert data["epoch"] == 1

    def test_checkpoint_persists_across_instances(self, checkpoint_dir, simple_model):
        """Test that checkpoint list persists across Checkpoint instances."""
        checkpoint1 = Checkpoint(save_dir=checkpoint_dir)
        checkpoint1.save(simple_model, epoch=1)
        checkpoint1.save(simple_model, epoch=2)

        # Create new instance pointing to same directory
        checkpoint2 = Checkpoint(save_dir=checkpoint_dir)

        # Should be able to list and load previous checkpoints
        info = checkpoint2.list_checkpoints()
        assert len(info) == 2

    def test_auto_filename_with_no_epoch_or_step(self, checkpoint_dir, simple_model):
        """Test auto-generated filename when no epoch or step provided."""
        checkpoint = Checkpoint(save_dir=checkpoint_dir)

        filepath = checkpoint.save(simple_model)

        assert filepath.exists()
        assert "checkpoint_" in filepath.name
