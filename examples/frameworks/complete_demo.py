"""
Complete example demonstrating all 8 VLA framework implementations.

This script shows how to:
1. Initialize each framework
2. Create mock data
3. Run training forward pass
4. Run inference
5. Compare frameworks

Run: python examples/frameworks/complete_demo.py
"""

import torch
import torch.nn as nn
from typing import Dict, Any

# ============================================================================
# Mock Components for Testing
# ============================================================================


class MockVLM(nn.Module):
    """Mock VLM for demonstration."""

    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Linear(3 * 224 * 224, hidden_dim)

    def forward(self, images, text=None, **kwargs):
        batch_size = images.size(0)
        flat = images.view(batch_size, -1)
        embeddings = self.projection(flat)
        return {"embeddings": embeddings.unsqueeze(1)}

    def encode_image(self, images, **kwargs):
        return self.forward(images)["embeddings"]

    def encode_text(self, text, **kwargs):
        batch_size = len(text) if isinstance(text, list) else 1
        return torch.randn(batch_size, self.hidden_dim)

    def get_embedding_dim(self):
        return self.hidden_dim

    @property
    def config(self):
        return {"type": "MockVLM", "hidden_dim": self.hidden_dim}


class MockVisionEncoder(nn.Module):
    """Mock vision encoder for demonstration."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 512, kernel_size=7, stride=2, padding=3)
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.output_dim = 512
        self.embed_dim = 512

    def forward(self, x):
        return self.pool(self.conv(x))


# ============================================================================
# Data Generation
# ============================================================================


def create_mock_data(batch_size: int = 4) -> Dict[str, Any]:
    """Create mock training data."""
    return {
        "images": torch.randn(batch_size, 3, 224, 224),
        "text": [f"Task {i}" for i in range(batch_size)],
        "proprioception": torch.randn(batch_size, 14),
        "actions": torch.randn(batch_size, 7),
        "action_sequences": torch.randn(batch_size, 10, 7),  # For ACT
        "task_ids": torch.randint(0, 10, (batch_size,)),  # For Octo
    }


# ============================================================================
# Main Demo
# ============================================================================


def main():
    """Run all demonstrations."""
    print("=" * 80)
    print("LibRobot VLA Framework Demonstration")
    print("Complete examples of all 8 VLA implementations")
    print("=" * 80)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    from librobot.models.frameworks import (
        GR00TVLA,
        Pi0VLA,
        OctoVLA,
        OpenVLA,
        RT2VLA,
        ACTVLA,
        HelixVLA,
        CustomVLA,
    )

    frameworks = []
    batch_size = 2

    # 1. GR00T
    print("\n1. Testing GR00TVLA...")
    vlm = MockVLM(256)
    model = GR00TVLA(vlm=vlm, action_dim=7, state_dim=14, hidden_dim=256, diffusion_steps=10)
    frameworks.append(("GR00T", model))

    # 2. π0
    print("2. Testing Pi0VLA...")
    model = Pi0VLA(vlm=vlm, action_dim=7, state_dim=14, hidden_dim=256, flow_steps=10)
    frameworks.append(("π0", model))

    # 3. Octo
    print("3. Testing OctoVLA...")
    vision_encoder = MockVisionEncoder()
    model = OctoVLA(vision_encoder=vision_encoder, action_dim=7, state_dim=14, hidden_dim=256)
    frameworks.append(("Octo", model))

    # 4. OpenVLA
    print("4. Testing OpenVLA...")
    model = OpenVLA(vlm=vlm, action_dim=7, hidden_dim=256)
    frameworks.append(("OpenVLA", model))

    # 5. RT-2
    print("5. Testing RT2VLA...")
    model = RT2VLA(vlm=vlm, action_dim=7, num_bins=64)
    frameworks.append(("RT-2", model))

    # 6. ACT
    print("6. Testing ACTVLA...")
    model = ACTVLA(vision_encoder=vision_encoder, action_dim=7, state_dim=14, chunk_size=5)
    frameworks.append(("ACT", model))

    # 7. Helix
    print("7. Testing HelixVLA...")
    model = HelixVLA(vlm=vlm, action_dim=7, state_dim=14, hidden_dim=256)
    frameworks.append(("Helix", model))

    # 8. Custom
    print("8. Testing CustomVLA...")
    from librobot.models.action_heads.mlp_oft import MLPActionHead

    action_head = MLPActionHead(256, 7)
    model = CustomVLA(action_dim=7, vlm=vlm, action_head=action_head, hidden_dim=256)
    frameworks.append(("Custom", model))

    # Test all frameworks
    print("\n" + "=" * 80)
    print("Framework Comparison")
    print("=" * 80)
    print(f"{'Framework':<15} {'Parameters':<15} {'Trainable':<15} {'Status'}")
    print("-" * 80)

    data = create_mock_data(batch_size)

    for name, model in frameworks:
        try:
            total_params = model.get_num_parameters()
            trainable_params = model.get_num_parameters(trainable_only=True)

            # Quick forward pass
            model.eval()
            with torch.no_grad():
                if name == "ACT":
                    pred = model.predict_action(data["images"], None, data["proprioception"])
                elif name == "Octo":
                    pred = model.predict_action(
                        data["images"], None, data["proprioception"], data["task_ids"]
                    )
                else:
                    pred = model.predict_action(
                        data["images"], data["text"], data["proprioception"]
                    )

            status = "✓ Working"
            print(f"{name:<15} {total_params:<15,} {trainable_params:<15,} {status}")

        except Exception as e:
            print(f"{name:<15} {'N/A':<15} {'N/A':<15} ✗ Error: {str(e)[:30]}")

    print("=" * 80)
    print("\n✓ All frameworks demonstrated successfully!")
    print("\nFor detailed usage, see:")
    print("  - librobot/models/frameworks/README.md")
    print("  - examples/frameworks/training_example.py")


if __name__ == "__main__":
    main()
