import torch

from ocl.neural_networks import extensions


def test_transformer_encoder():
    bs, seq_len_q, dim_q = 2, 3, 4
    seq_len_kv, dim_kv = 4, 3

    q = torch.randn(bs, seq_len_q, dim_q)
    kv = torch.randn(bs, seq_len_kv, dim_kv)

    # Self attention mode
    transf = extensions.TransformerEncoder(dim=dim_q, n_blocks=2, n_heads=1)
    outp = transf(q)
    assert outp.shape == (bs, seq_len_q, dim_q)

    # Cross attention mode
    transf = extensions.TransformerEncoder(dim=dim_q, n_blocks=2, n_heads=1, memory_dim=dim_kv)
    outp = transf(q, memory=kv)
    assert outp.shape == (bs, seq_len_q, dim_q)
