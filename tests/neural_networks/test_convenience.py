import pytest
import torch

from ocl.neural_networks import convenience


@pytest.mark.parametrize(
    "input_gating,hidden_gating",
    [
        (False, True),
        (True, True),
        (True, True),
    ],
)
@pytest.mark.parametrize(
    "update_transform,hidden_size",
    [
        (False, 4),
        (True, 3),
    ],
)
@pytest.mark.parametrize("convex_update", [False, True])
def test_recurrent_gated_cell(
    hidden_size, input_gating, hidden_gating, update_transform, convex_update
):
    bs = 2
    input_size = 4
    cell = convenience.RecurrentGatedCell(
        input_size, hidden_size, input_gating, hidden_gating, update_transform, convex_update
    )

    hidden = torch.randn(bs, hidden_size)
    inp = torch.randn(bs, input_size)
    out = cell(inp, hidden)
    assert out.shape == hidden.shape
