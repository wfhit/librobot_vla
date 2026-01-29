#!/usr/bin/env python3
"""
Demo script showing the LibroBot VLA framework in action.

This demonstrates:
1. Building a complete VLA model programmatically
2. Using the registry system
3. Forward and backward passes
4. Configuration integration
"""

import torch
import torch.nn as nn

from librobot.models.action_heads import DiffusionTransformerHead, MLPOFTHead
from librobot.models.encoders import MLPEncoder
from librobot.models.frameworks import GR00TStyleFramework
from librobot.robots import WheelLoaderRobot
from librobot.utils import load_config, set_seed, setup_logging, REGISTRY

# Setup
logger = setup_logging(level="INFO")
set_seed(42)


class MockVLM(nn.Module):
    """Mock VLM for demonstration."""

    def __init__(self, hidden_dim=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Simple projection from flattened images
        self.projection = nn.Sequential(
            nn.Linear(3 * 224 * 224, 1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_dim),
        )

    def forward(self, images, text=None, **kwargs):
        batch_size = images.shape[0]
        # Flatten images (assuming shape: [batch, num_cams, 3, 224, 224])
        x = images.reshape(batch_size, -1)
        features = self.projection(x)
        return {"hidden_states": features}

    def freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False


def main():
    logger.info("=" * 60)
    logger.info("LibroBot VLA Framework Demo")
    logger.info("=" * 60)

    # 1. Show available components
    logger.info("\n1. Available Components:")
    logger.info(f"   Action Heads: {REGISTRY.list('action_head')}")
    logger.info(f"   Frameworks: {REGISTRY.list('framework')}")
    logger.info(f"   Robots: {REGISTRY.list('robot')}")

    # 2. Load robot definition
    logger.info("\n2. Robot Definition:")
    robot = WheelLoaderRobot()
    logger.info(f"   Name: {robot.name}")
    logger.info(f"   Action dim: {robot.action_dim}")
    logger.info(f"   State dim: {robot.state_dim}")
    logger.info(f"   Actions: {robot.action_names}")

    # 3. Build model components
    logger.info("\n3. Building Model Components:")

    # VLM (frozen)
    vlm = MockVLM(hidden_dim=512)
    vlm.freeze_parameters()
    logger.info(f"   ✓ VLM: Mock VLM (frozen)")

    # State encoder
    state_encoder = MLPEncoder(
        input_dim=robot.state_dim,
        output_dim=256,
        hidden_dims=[256, 256],
    )
    logger.info(f"   ✓ State Encoder: MLP (22 → 256)")

    # Action head
    action_head = DiffusionTransformerHead(
        input_dim=512 + 256,  # VLM + state encoder
        action_dim=robot.action_dim,
        hidden_dim=512,
        num_layers=4,
        action_horizon=10,
        num_diffusion_steps=50,
        num_inference_steps=5,
    )
    logger.info(f"   ✓ Action Head: Diffusion Transformer")

    # 4. Build VLA framework
    logger.info("\n4. Building VLA Framework:")
    framework = GR00TStyleFramework(
        vlm=vlm,
        action_head=action_head,
        state_encoder=state_encoder,
        fusion_method="concat",
    )
    logger.info(f"   ✓ Framework: GR00T-style")

    # Show parameter counts
    params = framework.get_trainable_parameters()
    logger.info(f"   Trainable parameters:")
    for name, count in params.items():
        logger.info(f"     {name}: {count:,}")

    # 5. Demonstrate forward pass (training)
    logger.info("\n5. Training Forward Pass:")
    batch_size = 2

    # Create dummy inputs
    images = torch.randn(batch_size, 1, 3, 224, 224)  # Single camera
    instruction = ["Load material from pile", "Dump material in truck"]
    state = torch.randn(batch_size, robot.state_dim)
    actions = torch.randn(batch_size, 10, robot.action_dim)

    # Forward pass
    output = framework(images, instruction, state=state, actions=actions)
    logger.info(f"   Input shape: images={images.shape}, state={state.shape}")
    logger.info(f"   Output keys: {list(output.keys())}")
    logger.info(f"   Loss: {output['loss'].item():.4f}")

    # 6. Demonstrate inference
    logger.info("\n6. Inference:")
    with torch.no_grad():
        predicted_actions = framework.predict(
            images, instruction, state=state, num_samples=3
        )
    logger.info(f"   Predicted actions: {predicted_actions.shape}")
    logger.info(f"   Shape: [batch={batch_size}, samples=3, horizon=10, action_dim={robot.action_dim}]")

    # 7. Show action denormalization
    logger.info("\n7. Action Denormalization:")
    sample_action = predicted_actions[0, 0, 0].numpy()  # First sample, first timestep
    logger.info(f"   Normalized action: {sample_action}")
    denorm_action = robot.denormalize_action(sample_action)
    logger.info(f"   Denormalized action: {denorm_action}")
    logger.info(f"   Action names: {robot.action_names}")

    # 8. Load and show config
    logger.info("\n8. Configuration System:")
    config = load_config("configs/defaults.yaml")
    logger.info(f"   Loaded defaults.yaml")
    logger.info(f"   Seed: {config.seed}")
    logger.info(f"   Batch size: {config.training.batch_size}")
    logger.info(f"   Learning rate: {config.training.optimizer.lr}")

    logger.info("\n" + "=" * 60)
    logger.info("Demo completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
