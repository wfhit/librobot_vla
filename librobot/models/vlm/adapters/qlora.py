"""QLoRA with quantization."""
import torch
import torch.nn as nn
from .lora import LoRAAdapter

class QLoRAAdapter(LoRAAdapter):
    """QLoRA with 4-bit quantization."""
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16, bits: int = 4):
        super().__init__(in_features, out_features, rank, alpha)
        self.bits = bits
    
    def quantize_layer(self, layer: nn.Linear):
        """Quantize base layer weights."""
        weight = layer.weight.data
        scale = weight.abs().max() / (2 ** (self.bits - 1) - 1)
        quantized = (weight / scale).round().clamp(-(2 ** (self.bits - 1)), 2 ** (self.bits - 1) - 1)
        layer.weight.data = quantized * scale
        layer.weight.requires_grad = False
