from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import nn

import ocl.typing


class IdentityMapping(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ConcatConditioning(nn.Module):
    def __init__(self, dim: int, conditioning_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim + conditioning_dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)
        )

    def forward(self, z: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        z = torch.cat((z, conditioning), dim=-1)
        z = self.mlp(z)
        return z


class EncodeDualConditioning(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.language = nn.Linear(4096, dim // 2)
        self.point = nn.Linear(2, dim // 2)

    def forward(
        self, language: torch.Tensor, point: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        language = self.language(language) * mask.unsqueeze(-1)
        point = self.point(point) * mask.unsqueeze(-1)
        return torch.cat((language, point), dim=-1)


class EncodeLangConditioning(nn.Module):
    def __init__(self, dim: int, lang_dim: int):
        super().__init__()

        self.language = nn.Linear(lang_dim, dim)

    def forward(self, language: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        language = self.language(language.float()) * mask.unsqueeze(-1)
        return language


class EncodePointConditioning(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.point = nn.Linear(2, dim)

    def forward(self, point: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        point = self.point(point) * mask.unsqueeze(-1)
        return point


class EncodeDualConditioningPE(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.language = nn.Linear(4096, dim // 2)

    def forward(
        self, language: torch.Tensor, point: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        language = self.language(language) * mask.unsqueeze(-1)
        point = point * mask.unsqueeze(-1)
        return torch.cat((language, point), dim=-1)
