"""Methods for matching between sets of elements."""
import dataclasses
from typing import Optional, Tuple, Type

import numpy as np
import torch
from einops import rearrange
from scipy.optimize import linear_sum_assignment
from torchtyping import TensorType

# Avoid errors due to flake:
batch_size = None
n_elements = None

CostMatrix = Type[TensorType["batch_size", "n_elements", "n_elements"]]
AssignmentMatrix = Type[TensorType["batch_size", "n_elements", "n_elements"]]
CostVector = Type[TensorType["batch_size"]]


@dataclasses.dataclass
class MaskMatchingOutput:
    slots_indecies: TensorType["batch_size", "n_objects"]  # noqa: F821
    gt_masks_indecies: TensorType["batch_size", "n_objects"]  # noqa: F821
    matched_gt_masks: Optional[
        TensorType["batch_size", "n_objects", "height", "width"]  # noqa: F821
    ] = None
    matched_pred_masks: Optional[
        TensorType["batch_size", "n_objects", "height", "width"]
    ] = None  # noqa: F821


class Matcher(torch.nn.Module):
    """Matcher base class to define consistent interface."""

    def forward(self, C: CostMatrix) -> Tuple[AssignmentMatrix, CostVector]:
        pass


class CPUHungarianMatcher(Matcher):
    """Implementaiton of a cpu hungarian matcher using scipy.optimize.linear_sum_assignment."""

    def forward(self, C: CostMatrix) -> Tuple[AssignmentMatrix, CostVector]:
        X = torch.zeros_like(C)
        C_cpu: np.ndarray = C.detach().cpu().numpy()
        for i, cost_matrix in enumerate(C_cpu):
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            X[i][row_ind, col_ind] = 1.0
        return X, (C * X).sum(dim=(1, 2))


class MaskMatching(torch.nn.Module):
    def __init__(self, save_matched_masks: bool = True):
        super().__init__()
        self.save_matched_masks = save_matched_masks

    def forward(
        self, pred_masks: torch.Tensor, true_masks: torch.Tensor, selected_indices: torch.Tensor
    ) -> MaskMatchingOutput:
        """ "Args:
        prediction: Predicted mask of shape (B, C, H, W), where C is the
            number of objects.
        target: Ground truth mask of shape (B, K, H, W), where K is the
            number of gt_masks.
        """
        iou_empty = 0.0
        all_pred_idxs = []
        all_true_idxs = []

        # this is similar and partially copied from
        # IoU metrics computation with the difference that
        # we are using the selected_indices first to get the gt masks
        batch_idx = (
            torch.arange(pred_masks.shape[0], device=pred_masks.device)
            .unsqueeze(-1)
            .repeat(1, selected_indices.shape[1])
        )
        n_obj = selected_indices.shape[1]
        batch_idx = batch_idx.reshape(-1)
        selected_indices_flat = selected_indices.reshape(-1)
        indices = torch.argmax(pred_masks, dim=1)
        pred_masks = torch.nn.functional.one_hot(indices, num_classes=pred_masks.shape[1])
        pred_masks = rearrange(pred_masks, "b h w c -> b c h w")
        # -1 in selected_indices as padding would be removed later
        true_masks = true_masks[batch_idx, selected_indices_flat.long()].reshape(pred_masks.shape)

        for pred_mask, true_mask, selected_index in zip(pred_masks, true_masks, selected_indices):
            true_mask = true_mask[selected_index != -1]
            assert true_mask.shape[0] <= pred_mask.shape[0]
            pred_mask = pred_mask.flatten(-2, -1)
            true_mask = true_mask.flatten(-2, -1)
            assert pred_mask.ndim == 2
            assert true_mask.ndim == 2

            pred_mask = pred_mask.unsqueeze(1).to(torch.bool)
            true_mask = true_mask.unsqueeze(0).to(torch.bool)

            intersection = torch.sum(pred_mask & true_mask, dim=-1).to(torch.float64)
            union = torch.sum(pred_mask | true_mask, dim=-1).to(torch.float64)
            pairwise_iou = intersection / union
            pairwise_iou[union == 0] = iou_empty

            pred_idxs, true_idxs = linear_sum_assignment(pairwise_iou.cpu(), maximize=True)
            pred_idxs_t = torch.as_tensor(pred_idxs, dtype=torch.int64, device=pairwise_iou.device)
            true_idxs_t = torch.as_tensor(true_idxs, dtype=torch.int64, device=pairwise_iou.device)
            assert pred_idxs_t.shape[0] == true_idxs_t.shape[0]

            # n_obj size -1s
            pred_idxs_padded = -1 * torch.ones(n_obj, dtype=torch.int32, device=pred_idxs_t.device)
            pred_idxs_padded[: pred_idxs_t.shape[0]] = pred_idxs_t
            true_idxs_padded = -1 * torch.ones(n_obj, dtype=torch.int32, device=pred_idxs_t.device)
            true_idxs_padded[: pred_idxs_t.shape[0]] = true_idxs_t
            all_pred_idxs.append(pred_idxs_padded)
            all_true_idxs.append(true_idxs_padded)

        pred_idxs_padded = torch.stack(all_pred_idxs)
        true_idxs_padded = torch.stack(all_true_idxs)

        if self.save_matched_masks:
            batch_idx = (
                torch.arange(pred_masks.shape[0], device=pred_masks.device)
                .unsqueeze(-1)
                .repeat(1, pred_masks.shape[1])
            )
            batch_idx = batch_idx.reshape(-1)
            # to get the indecies of the matched masks
            # in the original COCO dataset one should use the selected_indices
            empty_selected_masks = selected_indices.reshape(-1) == -1
            empty_masks_true = true_idxs_padded.reshape(-1) == -1
            empty_masks_pred = pred_idxs_padded.reshape(-1) == -1
            slot_masks_shape = pred_masks.shape
            assert (empty_selected_masks == empty_masks_true).all()
            assert (empty_masks_true == empty_masks_pred).all()

            matched_gt_masks = true_masks[batch_idx, true_idxs_padded.reshape(-1)]
            matched_gt_masks[empty_masks_true] = 0
            matched_gt_masks = matched_gt_masks.reshape(slot_masks_shape)

            matched_pred_masks = pred_masks[batch_idx, pred_idxs_padded.reshape(-1)]
            matched_pred_masks[empty_masks_pred] = 0
            matched_pred_masks = matched_pred_masks.reshape(slot_masks_shape)
        else:
            matched_gt_masks = None
            matched_pred_masks = None
        output = MaskMatchingOutput(
            slots_indecies=pred_idxs_padded,
            gt_masks_indecies=true_idxs_padded,
            matched_gt_masks=matched_gt_masks,
            matched_pred_masks=matched_pred_masks,
        )
        return output
