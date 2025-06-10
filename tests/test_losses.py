import pytest
import torch

from ocl import losses


@pytest.mark.parametrize(
    "tensor_shape,target_shape,except_dim,expected_shape", [((3, 2), (3, 5, 3, 3), 1, (3, 2, 3, 3))]
)
def test_masked_reconstruction_loss_make_gatherable(
    tensor_shape, target_shape, except_dim, expected_shape
):
    tensor = torch.zeros(*tensor_shape)
    outp = losses.MaskedReconstructionLoss.make_gatherable(tensor, target_shape, except_dim)
    assert outp.shape == expected_shape
