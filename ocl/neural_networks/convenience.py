"""Convenience functions for the construction neural networks using config."""
from typing import Callable, List, Optional, Union

import torch
from torch import nn

from ocl.neural_networks.extensions import TransformerDecoderWithAttention
from ocl.neural_networks.wrappers import Residual


class ReLUSquared(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return nn.functional.relu(x, inplace=self.inplace) ** 2


class ReGLU(nn.Module):
    """ReLU + GLU activation.

    See https://arxiv.org/abs/2002.05202.
    """

    @staticmethod
    def reglu(x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] % 2 == 0
        a, b = x.chunk(2, dim=-1)
        return a * nn.functional.relu(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reglu(x)


class GEGLU(nn.Module):
    """GELU + GLU activation.

    See https://arxiv.org/abs/2002.05202.
    """

    @staticmethod
    def geglu(x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] % 2 == 0
        a, b = x.chunk(2, dim=-1)
        return a * nn.functional.gelu(b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.geglu(x)


def get_activation_fn(name: str, inplace: bool = True, leaky_relu_slope: Optional[float] = None):
    if callable(name):
        return name

    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=inplace)
    elif name == "relu_squared":
        return ReLUSquared(inplace=inplace)
    elif name == "leaky_relu":
        if leaky_relu_slope is None:
            raise ValueError("Slope of leaky ReLU was not defined")
        return nn.LeakyReLU(leaky_relu_slope, inplace=inplace)
    elif name == "gelu":
        return nn.GELU()
    elif name == "glu":
        return nn.GLU()
    elif name == "geglu":
        return GEGLU()
    elif name == "reglu":
        return ReGLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "identity":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation function {name}")


def build_mlp(
    input_dim: int,
    output_dim: int,
    features: List[int],
    activation_fn: Union[str, Callable] = "relu",
    final_activation_fn: Optional[Union[str, Callable]] = None,
    initial_layer_norm: bool = False,
    residual: bool = False,
) -> nn.Sequential:
    layers = []
    current_dim = input_dim
    if initial_layer_norm:
        layers.append(nn.LayerNorm(current_dim))

    for n_features in features:
        layers.append(nn.Linear(current_dim, n_features))
        nn.init.zeros_(layers[-1].bias)
        layers.append(get_activation_fn(activation_fn))
        if activation_fn.lower().endswith("glu"):
            current_dim = n_features // 2
        else:
            current_dim = n_features

    layers.append(nn.Linear(current_dim, output_dim))
    nn.init.zeros_(layers[-1].bias)
    if final_activation_fn is not None:
        layers.append(get_activation_fn(final_activation_fn))

    if residual:
        return Residual(nn.Sequential(*layers))
    return nn.Sequential(*layers)


class MLP(nn.Module):
    """Simple MLP class with the same signature as `build_mlp`."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.module = build_mlp(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.module(input)


def build_two_layer_mlp(
    input_dim,
    output_dim,
    hidden_dim,
    activation_fn: str = "relu",
    initial_layer_norm: bool = False,
    residual: bool = False,
):
    """Build a two layer MLP, with optional initial layer norm.

    Separate class as this type of construction is used very often for slot attention and
    transformers.
    """
    return build_mlp(
        input_dim,
        output_dim,
        [hidden_dim],
        activation_fn=activation_fn,
        initial_layer_norm=initial_layer_norm,
        residual=residual,
    )


def build_residual_mlp(
    input_dim: int,
    output_dim: int,
    embedding_dim: int,
    n_blocks: int,
    activation_fn: Union[str, Callable] = "relu",
    final_activation_fn: Optional[Union[str, Callable]] = None,
    initial_linear: bool = True,
    initial_layer_norm: bool = False,
    final_layer_norm: bool = False,
):
    layers = []
    if initial_linear:
        layers.append(nn.Linear(input_dim, embedding_dim, bias=not initial_layer_norm))
        nn.init.xavier_uniform_(layers[-1].weight)
        if hasattr(layers[-1], "bias"):
            nn.init.zeros_(layers[-1].bias)
    else:
        assert input_dim == embedding_dim

    if initial_layer_norm:
        layers.append(nn.LayerNorm(embedding_dim))

    for _ in range(n_blocks):
        block = Residual(
            nn.Sequential(
                nn.LayerNorm(embedding_dim, elementwise_affine=False),
                nn.Linear(embedding_dim, 4 * embedding_dim),
                get_activation_fn(activation_fn),
                nn.Linear(4 * embedding_dim, embedding_dim),
            )
        )
        layers.append(block)
        nn.init.xavier_uniform_(block.module[-3].weight, nn.init.calculate_gain("relu"))
        nn.init.zeros_(block.module[-3].bias)
        nn.init.xavier_uniform_(block.module[-1].weight)
        nn.init.zeros_(block.module[-1].bias)

    if final_layer_norm:
        layers.append(nn.LayerNorm(embedding_dim, elementwise_affine=False))

    layers.append(nn.Linear(embedding_dim, output_dim))
    nn.init.xavier_uniform_(layers[-1].weight)
    nn.init.zeros_(layers[-1].bias)

    if final_activation_fn is not None:
        layers.append(get_activation_fn(final_activation_fn))

    return nn.Sequential(*layers)


def build_transformer_encoder(
    input_dim: int,
    output_dim: int,
    n_layers: int,
    n_heads: int,
    hidden_dim: Optional[int] = None,
    dropout: float = 0.0,
    activation_fn: Union[str, Callable] = "relu",
    layer_norm_eps: float = 1e-5,
    use_output_transform: bool = True,
):
    if hidden_dim is None:
        hidden_dim = 4 * input_dim

    layers = []
    for _ in range(n_layers):
        layers.append(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation=activation_fn,
                layer_norm_eps=layer_norm_eps,
                batch_first=True,
                norm_first=True,
            )
        )

    if use_output_transform:
        layers.append(nn.LayerNorm(input_dim, eps=layer_norm_eps))
        output_transform = nn.Linear(input_dim, output_dim, bias=True)
        nn.init.xavier_uniform_(output_transform.weight)
        nn.init.zeros_(output_transform.bias)
        layers.append(output_transform)

    return nn.Sequential(*layers)


def build_transformer_decoder(
    input_dim: int,
    output_dim: int,
    n_layers: int,
    n_heads: int,
    hidden_dim: Optional[int] = None,
    dropout: float = 0.0,
    activation_fn: Union[str, Callable] = "relu",
    layer_norm_eps: float = 1e-5,
    return_attention_weights: bool = False,
    attention_weight_type: Union[int, str] = -1,
    **kwargs,
):
    if hidden_dim is None:
        hidden_dim = 4 * input_dim

    decoder_layer = nn.TransformerDecoderLayer(
        d_model=input_dim,
        nhead=n_heads,
        dim_feedforward=hidden_dim,
        dropout=dropout,
        activation=activation_fn,
        layer_norm_eps=layer_norm_eps,
        batch_first=True,
        norm_first=True,
    )

    if return_attention_weights:
        return TransformerDecoderWithAttention(
            decoder_layer,
            n_layers,
            return_attention_weights=True,
            attention_weight_type=attention_weight_type,
        )
    else:
        return nn.TransformerDecoder(decoder_layer, n_layers)


class RecurrentGatedCell(nn.Module):
    """Recurrent gated cell.

    Computes h + W_x x * sigmoid(W_gh h + W_gx x).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        input_gating: bool = True,
        hidden_gating: bool = True,
        update_transform: bool = True,
        convex_update: bool = False,
    ):
        super().__init__()
        self.input_gating = input_gating
        self.hidden_gating = hidden_gating
        self.update_transform = update_transform
        self.convex_update = convex_update
        if not input_gating and not hidden_gating:
            raise ValueError("At least one of `input_gating` or `hidden_gating` needs to be active")

        if update_transform:
            if input_gating:
                self.to_update = nn.Linear(input_size, 2 * hidden_size)
            else:
                self.to_update = nn.Linear(input_size, hidden_size)
        else:
            if input_size != hidden_size:
                raise ValueError("If `update_transform==False`, input_size must equal hidden_size")
            if input_gating:
                self.to_update = nn.Linear(input_size, hidden_size)
            else:
                self.to_update = nn.Identity()

        if hidden_gating:
            self.gating_hidden = nn.Linear(hidden_size, hidden_size)
        else:
            self.gating_hidden = None

    def forward(self, input: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if self.update_transform:
            if self.input_gating:
                update, gate = self.to_update(input).chunk(2, dim=-1)
            else:
                update = self.to_update(input)
                gate = torch.zeros_like(update)
        else:
            update = input
            if self.input_gating:
                gate = self.to_update(input)
            else:
                gate = torch.zeros_like(update)

        if self.hidden_gating:
            gate = gate + self.gating_hidden(hidden)

        gating = nn.functional.sigmoid(gate)
        if self.convex_update:
            return (1 - gating) * hidden + gating * update
        else:
            return hidden + gating * update
