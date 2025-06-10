# slightly modified from the https://github.com/singhgautam/slate/blob/master/transformer.py
import torch
import torch.nn as nn


def linear(in_features, out_features, bias=True, weight_init="xavier", gain=1.0):

    m = nn.Linear(in_features, out_features, bias)

    if weight_init == "kaiming":
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
    else:
        nn.init.xavier_uniform_(m.weight, gain)

    if bias:
        nn.init.zeros_(m.bias)

    return m


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0, gain=1.0):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads

        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        self.proj_q = linear(d_model, d_model, bias=False)
        self.proj_k = linear(d_model, d_model, bias=False)
        self.proj_v = linear(d_model, d_model, bias=False)
        self.proj_o = linear(d_model, d_model, bias=False, gain=gain)

    def forward(self, q, k, v, attn_mask=None):
        """
        q: batch_size x target_len x d_model
        k: batch_size x source_len x d_model
        v: batch_size x source_len x d_model
        attn_mask: target_len x source_len
        return: batch_size x target_len x d_model
        """
        B, T, _ = q.shape
        _, S, _ = k.shape

        q = self.proj_q(q).view(B, T, self.num_heads, -1).transpose(1, 2)
        k = self.proj_k(k).view(B, S, self.num_heads, -1).transpose(1, 2)
        v = self.proj_v(v).view(B, S, self.num_heads, -1).transpose(1, 2)

        q = q * (q.shape[-1] ** (-0.5))
        attn = torch.matmul(q, k.transpose(-1, -2))

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        output = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
        output = self.proj_o(output)
        output = self.o


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        dropout=0.0,
        gain=1.0,
        is_first=False,
    ):
        super().__init__()

        self.is_first = is_first

        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, gain)

        # Removed the triangular mask, allowing all patches to attend to each other
        # No need for self.self_attn_mask anymore

        self.encoder_decoder_attn_layer_norm = nn.LayerNorm(d_model)
        self.encoder_decoder_attn = MultiHeadAttention(d_model, num_heads, dropout, gain)

        self.ffn_layer_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, input, encoder_output):
        """
        input: batch_size x target_len x d_model
        encoder_output: batch_size x source_len x d_model
        return: batch_size x target_len x d_model
        """

        if self.is_first:
            input = self.self_attn_layer_norm(input)
            x = self.self_attn(input, input, input)  # No mask applied
            input = input + x
        else:
            x = self.self_attn_layer_norm(input)
            x = self.self_attn(x, x, x)  # No mask applied
            input = input + x

        x = self.encoder_decoder_attn_layer_norm(input)
        x = self.encoder_decoder_attn(x, encoder_output, encoder_output)
        input = input + x

        x = self.ffn_layer_norm(input)
        x = self.ffn(x)
        return input + x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_blocks,
        d_model,
        num_heads,
        output_dim,
        dropout=0.0,
    ):
        super().__init__()

        block = TransformerDecoderBlock
        if num_blocks > 0:
            gain = (3 * num_blocks) ** (-0.5)
            self.blocks = nn.ModuleList(
                [block(d_model, num_heads, dropout, gain, is_first=True)]
                + [
                    block(d_model, num_heads, dropout, gain, is_first=False)
                    for _ in range(num_blocks - 1)
                ]
            )
        else:
            self.blocks = nn.ModuleList()

        self.layer_norm = nn.LayerNorm(d_model)
        self.final_linear = nn.Linear(d_model, output_dim)  # Final layer to match output dimensions

    def forward(self, input, encoder_output):
        """
        input: batch_size x target_len x d_model
        encoder_output: batch_size x source_len x d_model, *slots*
        return: batch_size x target_len x output_dim
        """
        for block in self.blocks:
            input = block(input, encoder_output)

        # Normalize and apply final linear layer to produce output_dim features
        input = self.layer_norm(input)
        output = self.final_linear(input)
        return output
