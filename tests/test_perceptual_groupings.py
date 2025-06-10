import pytest
import torch

import ocl
from ocl import perceptual_grouping


@pytest.mark.parametrize("n_heads", [1, 4])
@pytest.mark.parametrize("kvq_dim", [None, 16])
@pytest.mark.parametrize(
    "use_gru,slot_update",
    [
        (False, None),
        (False, "residual"),
        (True, None),
    ],
)
@pytest.mark.parametrize("use_implicit_differentiation", [False, True])
@pytest.mark.parametrize("use_cosine_attention", [False, True])
@pytest.mark.parametrize("test_masking", [False, True])
def test_slot_attention(
    n_heads,
    kvq_dim,
    use_gru,
    slot_update,
    use_implicit_differentiation,
    use_cosine_attention,
    test_masking,
):
    bs, n_inputs, n_slots = 2, 5, 3
    inp_dim, slot_dim = 12, 8

    if slot_update == "residual":
        upd_dim = kvq_dim if kvq_dim else slot_dim

        class Update(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.transf = torch.nn.Linear(upd_dim, slot_dim)

            def forward(self, upd, hidden):
                return hidden + self.transf(upd)

        slot_update = Update()
    else:
        slot_update = None

    slot_attention = perceptual_grouping.SlotAttention(
        dim=slot_dim,
        feature_dim=inp_dim,
        iters=2,
        n_heads=n_heads,
        kvq_dim=kvq_dim,
        slot_update=slot_update,
        use_gru=use_gru,
        use_implicit_differentiation=use_implicit_differentiation,
        use_cosine_attention=use_cosine_attention,
        eps=0.0,
    )

    inputs = torch.randn(bs, n_inputs, inp_dim)
    slots = torch.randn(bs, n_slots, slot_dim)

    if test_masking:
        mask = torch.zeros(bs, n_slots, dtype=torch.bool)
        mask[0, 1] = True
        mask[0, 2] = True
        mask[1, 0] = True
    else:
        mask = None

    upd_slots, attn, _ = slot_attention(inputs, slots, mask)

    assert upd_slots.shape == (bs, n_slots, slot_dim)
    assert attn.shape == (bs, n_slots, n_inputs)

    if test_masking:
        # First slot should get all attention (averaged over heads)
        assert torch.allclose(attn[0, 0], torch.ones_like(attn[0, 0]) / n_heads)
        assert torch.allclose(attn[0, 1], torch.zeros_like(attn[0, 1]))
        assert torch.allclose(attn[0, 2], torch.zeros_like(attn[0, 2]))
        # Second and third slot should get all attention (averaged over heads)
        assert torch.allclose(attn[1, 0], torch.zeros_like(attn[1, 0]))
        assert torch.allclose(attn[1, 1] + attn[1, 2], torch.ones_like(attn[1, 1]) / n_heads)


@pytest.mark.parametrize("n_blocks", [1, 2])
def test_slot_attention_grouping(n_blocks):
    bs, n_inputs, n_slots = 2, 5, 3
    inp_dim, slot_dim = 12, 8

    grouping = perceptual_grouping.SlotAttentionGrouping(
        feature_dim=inp_dim,
        object_dim=slot_dim,
        n_blocks=n_blocks,
        ff_mlp=lambda: torch.nn.Linear(slot_dim, slot_dim),
    )

    inputs = ocl.typing.FeatureExtractorOutput(
        features=torch.randn(bs, n_inputs, inp_dim),
        positions=None,
    )
    slots = torch.randn(bs, n_slots, slot_dim)
    outp = grouping(inputs, slots)

    assert outp.objects.shape == (bs, n_slots, slot_dim)
    assert outp.feature_attributions.shape == (bs, n_slots, n_inputs)
