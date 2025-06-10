from typing import Callable, Optional, Tuple

import torch
from torch import nn


class PointPredictionHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, 2))

    def forward(self, attn: torch.Tensor, x: torch.Tensor) -> torch.Tensor:

        x = torch.bmm(attn, x)
        x = torch.sigmoid(self.mlp(x))
        return x


class AttentionAggregationHead(nn.Module):
    def __init__(self, dim: int, only_masks: bool = False, stop_gradient: bool = False):
        super().__init__()
        self.only_masks = only_masks
        self.stop_gradient = stop_gradient

    def forward(self, attn: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.stop_gradient:
            attn = attn.detach()
            x = x.detach()
        if self.only_masks:
            x = attn
        else:
            x = torch.bmm(attn, x)
        return x


class EncodeConditioningHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.language = nn.Linear(4096, dim // 2)
        self.point = nn.Linear(2, dim // 2)

    def forward(self, language: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
        language = self.language(language)
        point = self.point(point)
        return torch.cat((language, point), dim=-1)


class CategoryPredictionHead(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, num_classes))

    def forward(self, attn: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = torch.bmm(attn, x)
        x = self.mlp(x)
        return x


class SlotProjectorHead(nn.Module):
    def __init__(
        self, dim: int, embedding_dim: int, hidden_dim: Optional[int] = None, linear: bool = False
    ):
        super().__init__()
        self.linear = linear
        if hidden_dim is None:
            hidden_dim = 4 * dim
        if not linear:
            self.tranform = nn.Sequential(
                nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, embedding_dim)
            )
        else:
            self.tranform = nn.Linear(dim, embedding_dim)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        projected_slots = self.tranform(slots)
        return projected_slots


class DualEmbeddingHead(nn.Module):
    """Dual embedding of language embedding and bbox centroid."""

    def __init__(
        self,
        embedding_dim: int,
        linear: bool = False,
        language_dim: int = 4096,
        point_dim: int = 2,
    ):
        super().__init__()
        self.language = nn.Linear(language_dim, embedding_dim // 2)
        self.point = nn.Linear(point_dim, embedding_dim // 2)
        self.linear = linear
        if not linear:
            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.GELU(),
                nn.Linear(embedding_dim, embedding_dim),
            )
        else:
            self.mlp = None

    def forward(
        self,
        name_embedding: torch.Tensor,
        point_embedding: torch.Tensor,
    ) -> torch.Tensor:

        emb_point = self.point(point_embedding)
        # name_embedding normailzed to 1
        name_embedding = name_embedding / torch.norm(name_embedding, dim=-1, keepdim=True)
        emb_lang = self.language(name_embedding)
        embedding = torch.cat([emb_lang, emb_point], dim=-1)
        if self.mlp is not None:
            embedding = self.mlp(embedding)
        return embedding


class LangEmbeddingHead(nn.Module):
    """Dual embedding of language embedding and bbox centroid."""

    def __init__(self, embedding_dim: int, lang_dim: int, linear: bool = False):
        super().__init__()
        self.language = nn.Linear(lang_dim, embedding_dim)
        self.linear = linear
        if not linear:
            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.GELU(),
                nn.Linear(embedding_dim, embedding_dim),
            )
        else:
            self.mlp = None

    def forward(
        self,
        name_embedding: torch.Tensor,
    ) -> torch.Tensor:

        # name_embedding normailzed to 1
        name_embedding = name_embedding / torch.norm(name_embedding + 1e-4, dim=-1, keepdim=True)
        emb_lang = self.language(name_embedding.float())
        embedding = emb_lang
        if self.mlp is not None:
            embedding = self.mlp(embedding)
        return embedding


class PointEmbeddingHead(nn.Module):
    """Dual embedding of language embedding and bbox centroid."""

    def __init__(self, embedding_dim: int, linear: bool = False):
        super().__init__()
        self.point = nn.Linear(2, embedding_dim)
        self.linear = linear
        if not linear:
            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.GELU(),
                nn.Linear(embedding_dim, embedding_dim),
            )
        else:
            self.mlp = None

    def forward(
        self,
        point_embedding: torch.Tensor,
    ) -> torch.Tensor:

        emb_point = self.point(point_embedding)
        embedding = emb_point
        if self.mlp is not None:
            embedding = self.mlp(embedding)
        return embedding
