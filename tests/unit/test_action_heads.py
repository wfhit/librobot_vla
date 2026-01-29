"""Test action heads."""

import pytest
import torch

from librobot.models.action_heads import DiffusionTransformerHead, MLPOFTHead


def test_mlp_oft_head():
    """Test MLP OFT action head."""
    batch_size = 2
    input_dim = 512
    action_dim = 6
    action_horizon = 10

    head = MLPOFTHead(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dim=256,
        num_layers=3,
        action_horizon=action_horizon,
    )

    # Test forward pass
    features = torch.randn(batch_size, input_dim)
    output = head(features)

    assert "actions" in output
    assert output["actions"].shape == (batch_size, action_horizon, action_dim)

    # Test with ground truth actions
    actions = torch.randn(batch_size, action_horizon, action_dim)
    output = head(features, actions=actions)

    assert "loss" in output
    assert output["loss"].ndim == 0  # Scalar loss

    # Test predict
    predicted = head.predict(features, num_samples=1)
    assert predicted.shape == (batch_size, 1, action_horizon, action_dim)


def test_diffusion_head():
    """Test diffusion transformer head."""
    batch_size = 2
    input_dim = 512
    action_dim = 6
    action_horizon = 10

    head = DiffusionTransformerHead(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dim=256,
        num_layers=3,
        action_horizon=action_horizon,
        num_diffusion_steps=50,
        num_inference_steps=5,
    )

    # Test forward pass (requires actions for training)
    features = torch.randn(batch_size, input_dim)
    actions = torch.randn(batch_size, action_horizon, action_dim)
    output = head(features, actions=actions)

    assert "loss" in output
    assert output["loss"].ndim == 0

    # Test predict
    predicted = head.predict(features, num_samples=2)
    assert predicted.shape == (batch_size, 2, action_horizon, action_dim)
