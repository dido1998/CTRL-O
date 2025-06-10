"""Implementation of different positional embeddings."""
from typing import Optional, Tuple

import torch
from torch import nn


class SoftPositionEmbed(nn.Module):
    """Embeding of positions using convex combination of learnable tensors.

    This assumes that the input positions are between 0 and 1.
    """

    def __init__(
        self, n_spatial_dims: int, feature_dim: int, cnn_channel_order=False, savi_style=False
    ):
        """__init__.

        Args:
            n_spatial_dims (int): Number of spatial dimensions.
            feature_dim (int): Dimensionality of the input features.
            cnn_channel_order (bool): Assume features are in CNN channel order (i.e. C x H x W).
            savi_style (bool): Use savi style positional encoding, where positions are normalized
                between -1 and 1 and a single dense layer is used for embedding.
        """
        super().__init__()
        self.savi_style = savi_style
        n_features = n_spatial_dims if savi_style else 2 * n_spatial_dims
        self.dense = nn.Linear(in_features=n_features, out_features=feature_dim)
        self.cnn_channel_order = cnn_channel_order

    def forward(self, inputs: torch.Tensor, positions: torch.Tensor):
        if self.savi_style:
            # Rescale positional encoding to -1 to 1
            positions = (positions - 0.5) * 2
        else:
            positions = torch.cat([positions, 1 - positions], axis=-1)
        emb_proj = self.dense(positions)
        if self.cnn_channel_order:
            emb_proj = emb_proj.permute(*range(inputs.ndim - 3), -1, -3, -2)
        return inputs + emb_proj


class LearnedAdditivePositionalEmbed(nn.Module):
    """Add positional encoding as in SLATE."""

    def __init__(self, max_len, d_model, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model), requires_grad=True)
        nn.init.trunc_normal_(self.pe)

    def forward(self, input):
        T = input.shape[1]
        return self.dropout(input + self.pe[:, :T])


class DummyPositionEmbed(nn.Module):
    """Embedding that just passes through inputs without adding any positional embeddings."""

    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, positions: torch.Tensor):
        return inputs


def get_2d_sincos_pos_embed(
    embed_dim: int,
    grid_size: Tuple[int, int],
    temperature: float = 10000,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Get 2D sin-cos positional embedding.

    Adapted from MoCo-v3 under CC-NC 4.0:
    https://github.com/facebookresearch/moco-v3/blob/main/vits.py

    Args:
        embed_dim: Size of embedding. Must be divisible by 4.
        grid_size: Tuple of (height, width) specifying grid size.
        temperature: Temperature.
        device: Optional device.

    Returns:
        Tensor of shape (height * width, embed_dim).
    """
    h, w = grid_size
    grid_w = torch.arange(w, dtype=torch.float32, device=device)
    grid_h = torch.arange(h, dtype=torch.float32, device=device)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="xy")

    assert (
        embed_dim % 4 == 0
    ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1.0 / (temperature**omega)
    out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
    out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
    pos_emb = torch.cat(
        (torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)), dim=1
    )

    return pos_emb
