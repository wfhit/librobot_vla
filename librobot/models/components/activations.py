"""Common activation functions for neural networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation.

    GELU(x) = x * Φ(x) where Φ is the cumulative distribution function of the standard Gaussian.
    Used in BERT, GPT, and many modern transformers.

    Args:
        approximate: Whether to use tanh approximation (faster but less accurate)
    """

    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate=self.approximate)


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU) activation.

    SwiGLU(x) = Swish(W1 @ x) ⊙ (W2 @ x)
    Used in PaLM and LLaMA models for better performance.

    Args:
        dim: Input dimension
        hidden_dim: Hidden dimension (typically 4 * dim)
        bias: Whether to use bias in linear layers
    """

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.w1(x)) * self.w2(x)


class GeGLU(nn.Module):
    """
    GELU-Gated Linear Unit (GeGLU) activation.

    GeGLU(x) = GELU(W1 @ x) ⊙ (W2 @ x)
    Used in some transformer variants.

    Args:
        dim: Input dimension
        hidden_dim: Hidden dimension (typically 4 * dim)
        bias: Whether to use bias in linear layers
    """

    def __init__(self, dim: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.w1(x)) * self.w2(x)


class Mish(nn.Module):
    """
    Mish activation function.

    Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Smooth, non-monotonic activation used in some vision models.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class QuickGELU(nn.Module):
    """
    Quick GELU approximation.

    QuickGELU(x) = x * σ(1.702 * x)
    Faster approximation used in CLIP and other models.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


def get_activation(name: str, **kwargs) -> nn.Module:
    """
    Get activation function by name.

    Args:
        name: Activation name ('relu', 'gelu', 'silu', 'swish', 'mish', 'tanh', 'sigmoid', etc.)
        **kwargs: Additional arguments for activation

    Returns:
        Activation module

    Raises:
        ValueError: If activation name is not recognized
    """
    name = name.lower()

    activations = {
        "relu": nn.ReLU,
        "gelu": GELU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,  # SiLU and Swish are the same
        "mish": Mish,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "softplus": nn.Softplus,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "prelu": nn.PReLU,
        "quickgelu": QuickGELU,
        "identity": nn.Identity,
    }

    if name not in activations:
        raise ValueError(
            f"Unknown activation: {name}. Available activations: {list(activations.keys())}"
        )

    return activations[name](**kwargs)


__all__ = [
    "GELU",
    "SwiGLU",
    "GeGLU",
    "Mish",
    "QuickGELU",
    "get_activation",
]
