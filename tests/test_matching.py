import pytest
import torch
import math
from ocl.matching import MaskMatching

# Create a fixture for setting up the MaskMatching instance
@pytest.fixture
def mask_matching():
    return MaskMatching()

def test_basic_functionality(mask_matching):
    """Test the basic functionality with simple inputs."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred_masks = torch.tensor([[[[1, 0], [0, 1]]]], dtype=torch.float32, device=device)
    true_masks = torch.tensor([[[[1, 0], [0, 1]]]], dtype=torch.float32, device=device)
    selected_indices = torch.tensor([[0]], dtype=torch.int64, device=device)

    result = mask_matching(pred_masks, true_masks, selected_indices)

    expected_slots_indices = torch.tensor([[0]], dtype=torch.int32, device=device)
    expected_gt_masks_indices = torch.tensor([[0]], dtype=torch.int32, device=device)

    assert torch.equal(result.slots_indecies, expected_slots_indices)
    assert torch.equal(result.gt_masks_indecies, expected_gt_masks_indices)

def test_with_padding(mask_matching):
    """Test the case where padding is involved."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred_masks = torch.tensor([[[[1, 0], [0, 1]], [[0, 0], [0, 0]]]], dtype=torch.float32, device=device)
    true_masks = torch.tensor([[[[1, 0], [0, 1]], [[0, 0], [0, 0]]]], dtype=torch.float32, device=device)
    selected_indices = torch.tensor([[0, -1]], dtype=torch.int64, device=device)

    result = mask_matching(pred_masks, true_masks, selected_indices)

    expected_slots_indices = torch.tensor([[0, -1]], dtype=torch.int32, device=device)
    expected_gt_masks_indices = torch.tensor([[0, -1]], dtype=torch.int32, device=device)

    assert torch.equal(result.slots_indecies, expected_slots_indices)
    assert torch.equal(result.gt_masks_indecies, expected_gt_masks_indices)

def test_non_trivial_iou_matching(mask_matching):
    """Test with non-trivial IoU matching scenario."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create predicted masks and true masks with partial overlaps
    pred_masks = torch.tensor(
        [
            [
                [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],  # Prediction 0
                [[0, 0, 1, 0], [0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]] ,   # Prediction 1
                [[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1]],  # Prediction 2

            ]
        ], 
        dtype=torch.float32, device=device
    )
    
    true_masks = torch.tensor(
        [
            [
                [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]],  # Ground truth 0
                [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],  # Ground truth 1
                [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]]   # Ground truth 2
            ]
        ],
        dtype=torch.float32, device=device
    )

    selected_indices = torch.tensor([[0, 1, 2]], dtype=torch.int64, device=device)

    result = mask_matching(pred_masks, true_masks, selected_indices)

    # The expected indices are not necessarily in order due to potential equal IoUs,
    # but the optimal assignment should match the indices as per IoU maximization.
    # Here, the Hungarian algorithm should match each prediction to the ground truth
    # with the highest IoU:
    # Prediction 0 to Ground Truth 0
    # Prediction 1 to Ground Truth 2
    # Prediction 2 to Ground Truth 2
    expected_slots_indices = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
    expected_gt_masks_indices = torch.tensor([[0, 2, 1]], dtype=torch.int32, device=device)

    assert torch.equal(result.slots_indecies, expected_slots_indices), "Slots indices do not match expected result."
    assert torch.equal(result.gt_masks_indecies, expected_gt_masks_indices), "GT masks indices do not match expected result."

def test_single_object(mask_matching):
    """Test the single-object case."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred_masks = torch.tensor([[[[1, 1], [0, 0]]]], dtype=torch.float32, device=device)
    true_masks = torch.tensor([[[[1, 1], [0, 0]]]], dtype=torch.float32, device=device)
    selected_indices = torch.tensor([[0]], dtype=torch.int64, device=device)

    result = mask_matching(pred_masks, true_masks, selected_indices)

    expected_slots_indices = torch.tensor([[0]], dtype=torch.int32, device=device)
    expected_gt_masks_indices = torch.tensor([[0]], dtype=torch.int32, device=device)

    assert torch.equal(result.slots_indecies, expected_slots_indices)
    assert torch.equal(result.gt_masks_indecies, expected_gt_masks_indices)


if __name__ == "__main__":
    pytest.main()