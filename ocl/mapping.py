from copy import deepcopy
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from torch import nn

import ocl.typing
from ocl.neural_networks.convenience import build_transformer_encoder


class MLPMapping(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.LayerNorm(2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )

    def forward(self, x: ocl.typing.FeatureExtractorOutput) -> ocl.typing.FeatureExtractorOutput:
        x_copy = ocl.typing.FeatureExtractorOutput(
            features=x.features.clone(), positions=x.positions.clone()
        )
        x_copy.features = self.mlp(x.features)
        return x_copy


class IdentityMapping(nn.Module):
    def forward(self, x: ocl.typing.FeatureExtractorOutput) -> torch.Tensor:
        return x


class EncCondnMapping(nn.Module):
    def __init__(self, dim: int, num_layers: int, num_heads: int):
        super().__init__()
        self.encoder = build_transformer_encoder(dim, dim, num_layers, num_heads)
        self.language = nn.Linear(4096, dim // 2)
        self.point = nn.Linear(2, dim // 2)

    def forward(
        self,
        x: ocl.typing.FeatureExtractorOutput,
        language: torch.Tensor,
        point: torch.Tensor,
        mask: torch.Tensor,
    ) -> ocl.typing.FeatureExtractorOutput:
        x_copy = ocl.typing.FeatureExtractorOutput(
            features=x.features.clone(), positions=x.positions.clone()
        )
        language_emb = self.language(language) * mask.unsqueeze(-1)
        point_emb = self.point(point) * mask.unsqueeze(-1)
        condn_emb = torch.cat([language_emb, point_emb], dim=-1)
        enc_inp = torch.cat([condn_emb, x.features], dim=1)
        x_copy.features = self.encoder(enc_inp)[:, condn_emb.shape[1] :]

        return x_copy


class DetachedMLPMapping(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.LayerNorm(2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )

    def forward(self, x: ocl.typing.FeatureExtractorOutput) -> ocl.typing.FeatureExtractorOutput:
        x_copy = ocl.typing.FeatureExtractorOutput(
            features=x.features.clone(), positions=x.positions.clone()
        )
        x_copy.features = self.mlp(x.features.detach())
        return x_copy
