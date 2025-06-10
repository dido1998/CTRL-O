"""Utility functions applied to tensors."""
import torch


@torch.jit.script
def standardize(x: torch.Tensor, dim: int = -1, eps: float = 1e-5) -> torch.Tensor:
    """Standardize (zero mean, unit variance) the input."""
    mean = x.mean(dim=dim, keepdim=True)
    var = x.var(dim=dim, keepdim=True, unbiased=False)
    x = (x - mean) / (var + eps) ** 0.5
    return x


@torch.jit.script
def pixel_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-4) -> torch.Tensor:
    """Normalize input such that it has length sqrt(D)."""
    return x / torch.sqrt(torch.mean(x**2, dim=dim, keepdim=True) + eps)
