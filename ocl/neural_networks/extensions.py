"""Extensions of existing layers to implement additional functionality."""
from typing import Callable, Optional, Union

import torch
from torch import nn

from ocl.neural_networks import modules


class TransformerDecoderWithAttention(nn.TransformerDecoder):
    """Modified nn.TransformerDecoder class that returns attention weights over memory."""

    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        return_attention_weights=False,
        attention_weight_type: Union[int, str] = "mean",
    ):
        super(TransformerDecoderWithAttention, self).__init__(decoder_layer, num_layers, norm)

        if return_attention_weights:
            self.attention_hooks = []
            for layer in self.layers:
                self.attention_hooks.append(self._prepare_layer(layer))
        else:
            self.attention_hooks = None

        if isinstance(attention_weight_type, int):
            if attention_weight_type >= num_layers or attention_weight_type < -num_layers:
                raise ValueError(
                    f"Index {attention_weight_type} exceeds number of layers {num_layers}"
                )
        elif attention_weight_type != "mean":
            raise ValueError("`weights` needs to be a number or 'mean'.")
        self.weights = attention_weight_type

    def _prepare_layer(self, layer):
        assert isinstance(layer, nn.TransformerDecoderLayer)

        def _mha_block(self, x, mem, attn_mask, key_padding_mask, is_causal):
            x = self.multihead_attn(
                x,
                mem,
                mem,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                is_causal=is_causal,
                need_weights=True,
            )[0]
            return self.dropout2(x)

        # Patch _mha_block method to compute attention weights
        layer._mha_block = _mha_block.__get__(layer, nn.TransformerDecoderLayer)

        class AttentionHook:
            def __init__(self):
                self._attention = None

            def pop(self) -> torch.Tensor:
                assert self._attention is not None, "Forward was not called yet!"
                attention = self._attention
                self._attention = None
                return attention

            def __call__(self, module, inp, outp):
                self._attention = outp[1]

        hook = AttentionHook()
        layer.multihead_attn.register_forward_hook(hook)
        return hook

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = tgt

        for mod in self.layers:
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        if self.norm is not None:
            output = self.norm(output)

        if self.attention_hooks is not None:
            attentions = []
            for hook in self.attention_hooks:
                attentions.append(hook.pop())

            if self.weights == "mean":
                attentions = torch.stack(attentions, dim=-1)
                # Take mean over all layers
                attention = attentions.mean(dim=-1)
            else:
                attention = attentions[self.weights]

            return output, attention.transpose(1, 2)
        else:
            return output


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Like torch.nn.TransformerEncoderLayer, but with customizations."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dim_kv: Optional[int] = None,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = torch.nn.functional.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = False,
        initial_residual_scale: Optional[float] = None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            device=device,
            dtype=dtype,
        )
        self.self_attn = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            bias=qkv_bias,
            kdim=dim_kv,
            vdim=dim_kv,
            batch_first=True,
            **factory_kwargs,
        )
        if initial_residual_scale is not None:
            self.scale1 = modules.LayerScale(d_model, init_values=initial_residual_scale)
            self.scale2 = modules.LayerScale(d_model, init_values=initial_residual_scale)
        else:
            self.scale1 = nn.Identity()
            self.scale2 = nn.Identity()

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        keys: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> torch.Tensor:
        keys = keys if keys is not None else x
        values = values if values is not None else x
        x, attn = self.self_attn(
            x,
            keys,
            values,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=return_weights,
        )
        x = self.dropout1(x)

        if return_weights:
            return x, attn
        else:
            return x

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = src
        if self.norm_first:
            x = x + self.scale1(
                self._sa_block(
                    self.norm1(x), src_mask, src_key_padding_mask, keys=memory, values=memory
                )
            )
            x = x + self.scale2(self._ff_block(self.norm2(x)))
        else:
            x = self.norm1(
                x
                + self.scale1(
                    self._sa_block(x, src_mask, src_key_padding_mask, keys=memory, values=memory)
                )
            )
            x = self.norm2(x + self.scale2(self._ff_block(x)))

        return x


class TransformerEncoder(nn.Module):
    """TransformerEncoder like torch.nn.TransformerEncoder, but supporting cross-attention mode."""

    def __init__(
        self,
        dim: int,
        n_blocks: int,
        n_heads: int,
        memory_dim: Optional[int] = None,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = "relu",
        hidden_dim: Optional[int] = None,
        initial_residual_scale: Optional[float] = None,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * dim

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    dim,
                    n_heads,
                    dim_feedforward=hidden_dim,
                    dim_kv=memory_dim,
                    qkv_bias=qkv_bias,
                    dropout=dropout,
                    activation=activation,
                    layer_norm_eps=1e-05,
                    batch_first=True,
                    norm_first=True,
                    initial_residual_scale=initial_residual_scale,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(
        self,
        inp: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        memory: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = inp

        for block in self.blocks:
            x = block(x, mask, key_padding_mask, memory)

        return x
