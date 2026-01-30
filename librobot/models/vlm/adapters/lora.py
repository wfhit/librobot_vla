"""LoRA (Low-Rank Adaptation) adapter."""
import torch
import torch.nn as nn

class LoRAAdapter(nn.Module):
    """LoRA adapter for efficient fine-tuning."""
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

    def apply_to_layer(self, layer: nn.Linear):
        """Apply LoRA to a linear layer."""
        original_forward = layer.forward
        def new_forward(x):
            return original_forward(x) + self.forward(x)
        layer.forward = new_forward
