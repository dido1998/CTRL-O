from math import log
from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F
from torchvision.ops import generalized_box_iou

import ocl.typing
from ocl.matching import CPUHungarianMatcher
from ocl.utils import tensor_ops
from ocl.utils.bboxes import box_cxcywh_to_xyxy


class DiagonalContrastiveLoss(nn.Module):
    def __init__(
        self,
        l2_normalize: bool = True,
        weight: float = 1.0,
        symmetric: bool = False,
        temp: float = 0.1,
        batch_contrastive: bool = True,
    ):
        super().__init__()
        self.l2_normalize = l2_normalize
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.weight = weight
        self.temp = temp
        self.symmetric = symmetric
        self.batch_contrastive = batch_contrastive

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        contrastive_loss_mask: torch.Tensor,
    ):
        if self.l2_normalize:
            x1 = F.normalize(x1, dim=-1, p=2)
            x2 = F.normalize(x2, dim=-1, p=2)
        if self.batch_contrastive:
            if self.symmetric:
                x1 = x1[contrastive_loss_mask == 1]
                x2 = x2[contrastive_loss_mask == 1]
                logits_x1x2 = x1 @ x2.T / self.temp
                logits_x2x1 = logits_x1x2.T
                batch_size = x1.size(0)
                labels = torch.arange(batch_size, device=x1.device)
                # Mask self-comparisons
                loss = (self.loss_fn(logits_x1x2, labels) + self.loss_fn(logits_x2x1.T, labels)) / 2
                return self.weight * loss.mean()
            else:
                x2 = x2[contrastive_loss_mask == 1]

                pos_x1 = x1[contrastive_loss_mask == 1]
                neg_x1 = x1[contrastive_loss_mask == 0]
                x_1 = torch.cat([pos_x1, neg_x1], dim=0)
                logits_x2x1 = x2 @ x_1.T / self.temp
                logits_x1x2 = pos_x1 @ x2.T / self.temp
                batch_size = pos_x1.size(0)
                labels = torch.arange(batch_size, device=x1.device)

                # Mask self-comparisons
                loss = (self.loss_fn(logits_x2x1, labels) + self.loss_fn(logits_x1x2, labels)) / 2
                return self.weight * loss.mean()
        else:
            logits_x1x2 = torch.bmm(x1, x2.permute(0, 2, 1)) / self.temp
            logits_x2x1 = torch.bmm(x2, x1.permute(0, 2, 1)) / self.temp
            # make masked values -inf
            logits_x2x1.masked_fill_((contrastive_loss_mask == 0).unsqueeze(1), float(-1e10))
            logits_x1x2.masked_fill_((contrastive_loss_mask == 0).unsqueeze(1), float(-1e10))

            num_slots = logits_x1x2.shape[1]
            bs = logits_x1x2.shape[0]
            labels = (
                torch.arange(num_slots, device=logits_x1x2.device)
                .unsqueeze(0)
                .repeat(bs, 1)
                .reshape(-1)
            )

            contrastive_loss_mask = contrastive_loss_mask.reshape(-1)

            if self.symmetric:
                loss = (
                    self.loss_fn(logits_x1x2.view(-1, num_slots), labels.view(-1))
                    + self.loss_fn(logits_x2x1.view(-1, num_slots), labels.view(-1))
                ) / 2
            else:
                loss = self.loss_fn(logits_x1x2.view(-1, num_slots), labels.view(-1))

            return (self.weight * loss * contrastive_loss_mask).sum() / contrastive_loss_mask.sum()


class ReconstructionLoss(nn.Module):
    """Simple reconstruction loss."""

    def __init__(
        self,
        loss_type: str,
        weight: float = 1.0,
        normalize_prediction: bool = False,
        normalize_target: bool = False,
        normalization: str = "standardize",
    ):
        """Initialize ReconstructionLoss.

        Args:
            loss_type: One of `mse`, `mse_sum`, `l1`, `cosine_loss`, `cross_entropy_sum`.
            weight: Weight of loss, output is multiplied with this value.
            normalize_prediction: Normalize prediction over last dimension prior to computing loss.
            normalize_target: Normalize target over last dimension prior to computing loss.
            normalization: Type of normalization to apply.
        """
        super().__init__()
        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "mse_sum":
            # Used for slot_attention and video slot attention.
            self.loss_fn = (
                lambda x1, x2: nn.functional.mse_loss(x1, x2, reduction="sum") / x1.shape[0]
            )
        elif loss_type == "l1":
            self.loss_name = "l1_loss"
            self.loss_fn = nn.functional.l1_loss
        elif loss_type == "smooth_l1":
            self.loss_name = "smooth_l1_loss"
            self.loss_fn = nn.functional.smooth_l1_loss
        elif loss_type == "cosine":
            self.loss_name = "cosine_loss"
            self.loss_fn = lambda x1, x2: -nn.functional.cosine_similarity(x1, x2, dim=-1).mean()
        elif loss_type == "cross_entropy_sum":
            # Used for SLATE, average is over the first (batch) dim only.
            self.loss_name = "cross_entropy_sum_loss"
            self.loss_fn = (
                lambda x1, x2: nn.functional.cross_entropy(
                    x1.reshape(-1, x1.shape[-1]), x2.reshape(-1, x2.shape[-1]), reduction="sum"
                )
                / x1.shape[0]
            )
        else:
            raise ValueError(
                f"Unknown loss {loss_type}. Valid choices are (mse, l1, cosine, cross_entropy)."
            )
        # If weight is callable use it to determine scheduling otherwise use constant value.
        self.weight = weight
        self.normalize_prediction = normalize_prediction
        self.normalize_target = normalize_target
        if normalization == "standardize":
            self.normalize_fn = tensor_ops.standardize
        elif normalization == "l2":
            self.normalize_fn = lambda x: F.normalize(x, dim=-1, eps=1e-8)
        else:
            raise ValueError(f"Unknown normalization {normalization}")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss.

        Args:
            input: Prediction / input tensor.
            target: Target tensor.

        Returns:
            The reconstruction loss.
        """
        target = target.detach()
        if self.normalize_target:
            target = self.normalize_fn(target)

        if self.normalize_prediction:
            input = self.normalize_fn(input)

        loss = self.loss_fn(input, target)
        return self.weight * loss


class ControlMaskReconstructionLoss(nn.Module):
    def __init__(self, loss_type: str, weight: float = 1.0):
        super().__init__()
        self.loss_fn = nn.BCELoss(reduction="none")

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, control_mask: torch.Tensor
    ) -> torch.Tensor:
        b, n, h, w = input.shape
        batch_idx = (
            torch.arange(input.shape[0], device=input.device)
            .unsqueeze(-1)
            .repeat(1, control_mask.shape[1])
        )
        batch_idx = batch_idx.reshape(-1)
        control_mask = control_mask.reshape(-1)

        target = target[batch_idx, control_mask.long()]
        target = target.reshape(b, n, h, w)
        target[target > 0] = 1

        loss_mask = torch.ones_like(control_mask)
        loss_mask[control_mask == -1] = 0

        loss_mask = loss_mask.reshape(b, n)

        if target.dtype != input.dtype:
            target = target.to(input.dtype)

        loss = self.loss_fn(input, target)

        return (loss * loss_mask.unsqueeze(-1).unsqueeze(-1)).mean()


class PointPredictionLoss(nn.Module):
    def __init__(self, iter_start: int = 50000, weight: float = 1.0):
        super().__init__()
        self.iter_start = iter_start
        self.loss_fn = nn.MSELoss(reduction="none")
        self.weight = weight
        self.cur_iter = 0

    def forward(
        self, point_preds: torch.Tensor, target: torch.Tensor, control_mask: torch.Tensor
    ) -> torch.Tensor:
        new_target = target
        if self.cur_iter < self.iter_start:
            self.cur_iter += 1
            return torch.tensor(0.0, device=point_preds.device)
        else:
            loss = self.loss_fn(point_preds, new_target)
            self.cur_iter += 1

        loss_mask = torch.ones_like(control_mask)
        loss_mask[control_mask == -1] = 0

        return self.weight * (loss * loss_mask.unsqueeze(-1)).mean()


class CategoryPredictionLoss(nn.Module):
    def __init__(self, iter_start: int = 50000, weight: float = 1.0):
        super().__init__()
        self.iter_start = iter_start
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.weight = weight
        self.cur_iter = 0

    def forward(
        self, category_preds: torch.Tensor, target: torch.Tensor, control_mask: torch.Tensor
    ) -> torch.Tensor:
        new_target = target.long()
        if self.cur_iter < self.iter_start:
            self.cur_iter += 1
            return torch.tensor(0.0, device=category_preds.device)
        else:
            loss = self.loss_fn(category_preds.permute(0, 2, 1), new_target)
            self.cur_iter += 1
        loss_mask = torch.ones_like(control_mask)
        loss_mask[control_mask == -1] = 0

        return self.weight * (loss * loss_mask).mean()


class EmbeddingRefereceLoss(nn.Module):
    """
    CLIP style loss function for matching slot embeddings with "control" embeddings.

    Args:
        temp (float, optional): Temperature scaling factor for the logits. Default is 0.1.
        symmetric (bool, optional): Whether to apply a symmetric loss by computing
            the cross-entropy loss in both directions (slot-to-control and control-to-slot).
            Default is True.

    Forward:
        Args:
            slot_embeddings (torch.Tensor): Tensor of shape (batch_size, num_slots, dim)
                containing the embeddings of the slots.
            ctrl_embeddings (torch.Tensor): Tensor of shape (batch_size, num_slots, dim)
                containing the control embeddings (e.g., lang or point embeddings).
            ctrl_indices (torch.Tensor): Tensor of shape (batch_size, num_slots) containing the
                indices to select the relevant control embeddings for comparison.
            slots_indices (torch.Tensor): Tensor of shape (batch_size, num_slots) containing the
                indices to select the relevant slot embeddings for comparison.

        Returns:
            torch.Tensor: Scalar CE loss value computed by comparing slot embeddings with control embeddings.

    Notes:
        - The loss compares slot and control embeddings using a bi-directional (symmetric) cross-entropy loss,
          optionally scaled by a temperature factor.
        - The `ctrl_indices` and `slots_indices` are used to match the relevant embeddings.
          For example, mask matching matches those embeddings have maximal IoU overlap with the embedding masks.
        - Invalid indices (where index is -1) are ignored during the loss computation.
    """

    def __init__(
        self,
        temp: float = 0.1,
        symmetric: bool = True,
        batch_contrastive: bool = False,
        weight: float = 1.0,
    ):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.temp = temp
        self.symmetric = symmetric
        self.batch_contrastive = batch_contrastive
        self.weight = weight

    def forward(
        self,
        slot_embeddings: torch.Tensor,
        ctrl_embeddings: torch.Tensor,
        ctrl_indecies: torch.Tensor,
        slots_indecies: torch.Tensor,
    ) -> torch.Tensor:
        assert slot_embeddings.shape == ctrl_embeddings.shape
        batch_size, num_slots, dim = slot_embeddings.size()
        batch_idx = (
            torch.arange(batch_size, device=slot_embeddings.device)
            .unsqueeze(-1)
            .expand(batch_size, num_slots)
        )
        batch_idx = batch_idx.reshape(-1)
        ctrl_indecies_flat = ctrl_indecies.reshape(-1).long()
        slots_indecies_flat = slots_indecies.reshape(-1).long()
        assert ((ctrl_indecies_flat == -1) == (slots_indecies_flat == -1)).all()
        # not valid slots and gt masks should be ignored in loss and metrics
        mask_index = ctrl_indecies != -1

        # select gt embeddings that match the slot embeddings
        ctrl_embeddings = ctrl_embeddings[batch_idx, ctrl_indecies_flat]
        ctrl_embeddings = ctrl_embeddings.reshape([batch_size, num_slots, dim])

        # we need to permute slot embeddings also
        # for the case when ctrl_embeddings number is less then num_slots
        # otherwise their slot_indecies are just [0, 1, 2, ...]
        slot_embeddings = slot_embeddings[batch_idx, slots_indecies_flat]
        slot_embeddings = slot_embeddings.reshape([batch_size, num_slots, dim])
        slot_embeddings = F.normalize(slot_embeddings, dim=-1, p=2)
        ctrl_embeddings = F.normalize(ctrl_embeddings, dim=-1, p=2)
        if self.batch_contrastive:
            slot_embeddings = slot_embeddings[mask_index]
            ctrl_embeddings = ctrl_embeddings[mask_index]
            logits_sc = slot_embeddings @ ctrl_embeddings.T / self.temp
            logits_cs = ctrl_embeddings @ slot_embeddings.T / self.temp
            batch_size = slot_embeddings.size(0)
            labels = torch.arange(batch_size, device=slot_embeddings.device)
            # Mask self-comparisons
            loss = (self.loss_fn(logits_sc, labels) + self.loss_fn(logits_cs.T, labels)) / 2
            return self.weight * loss.mean()
        else:
            logits_sc = torch.bmm(slot_embeddings, ctrl_embeddings.permute(0, 2, 1)) / self.temp
            logits_cs = torch.bmm(ctrl_embeddings, slot_embeddings.permute(0, 2, 1)) / self.temp
            # make masked values -inf
            logits_sc.masked_fill_((mask_index == 0).unsqueeze(1), float(-1e10))
            logits_cs.masked_fill_((mask_index == 0).unsqueeze(1), float(-1e10))

            mask_index = mask_index.reshape(-1)
            # after matching, the taget is identity matrix
            labels = torch.arange(num_slots).expand(batch_size, num_slots).to(slot_embeddings.device)
            if self.symmetric:
                loss = (
                    self.loss_fn(logits_sc.view(-1, num_slots), labels.view(-1))
                    + self.loss_fn(logits_cs.view(-1, num_slots), labels.view(-1))
                ) / 2
            else:
                loss = self.loss_fn(logits_cs.view(-1, num_slots), labels.view(-1))

            return (loss * mask_index.float()).sum() / mask_index.sum()


class MaskedReconstructionLoss(ReconstructionLoss):
    """Reconstruction loss where only parts of the target are used in the loss.

    Loss receives an additional argument `indices` that specifies the indexes along dimension
    `index_dim` which should be used from the target in the loss computation. Shape of `indices`
    should match target's shape from the left except at `index_dim`, but is made to broadcast from
    the right.
    """

    def __init__(self, *args, index_dim: int, **kwargs):
        """Initialize.

        Args:
            index_dim: dimension of target along which elements are selected.
        """
        super().__init__(*args, **kwargs)
        self.index_dim = index_dim

    @staticmethod
    def make_gatherable(tensor: torch.Tensor, shape: torch.Size, except_dim: int) -> torch.Tensor:
        """Expand tensor to target size such that it can be used in a gather.

        Args:
            tensor: Tensor to expand
            shape: Target shape. Needs to match with tensor's shape from the left except at dimension
                specified by `except_dim`, and have greater or equal dimensions than tensor's shape.
            except_dim: Dimension where tensor's shape and target shape do not need to match.
        """
        if len(shape) < tensor.ndim:
            raise ValueError(
                "Target shape can not have less dimensions than tensor, but "
                f"{len(shape)} (target) < {tensor.ndim} (tensor)"
            )
        tensor_shape = tensor.shape
        view_shape = []
        expanded_shape = []
        for idx, size in enumerate(shape):
            if idx < len(tensor_shape):
                tensor_size = tensor_shape[idx]
            else:
                tensor_size = None

            if tensor_size is not None:
                if idx == except_dim or tensor_size == size:
                    view_shape.append(tensor_size)
                    expanded_shape.append(tensor_size)
                else:
                    raise ValueError(
                        "Target shape and tensor need to have shapes that match from "
                        "the left except at dim `except_dim`, but found shapes "
                        f"{shape} and {tensor.shape}"
                    )
            else:
                view_shape.append(1)
                expanded_shape.append(size)

        return tensor.view(*view_shape).expand(expanded_shape)

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if indices is not None:
            indices = self.make_gatherable(indices, shape=target.shape, except_dim=self.index_dim)
            target = torch.gather(target, dim=self.index_dim, index=indices)

        if input.shape != target.shape:
            raise ValueError(
                f"Shape of input ({input.shape}) and target ({target.shape}) after "
                "masking do not match."
            )

        return super().forward(input, target)


class MAELoss(nn.Module):
    """Reconstruction loss as used in masked auto-encoders.

    Expects full sized predictions and targets as input, but only those patches specified by the
    mask are taken into account by the loss. Supports images and already patched data (e.g features).
    """

    def __init__(
        self,
        weight: float = 1.0,
        normalize_target: bool = False,
        patchify_target: bool = True,
        patch_size: Optional[int] = None,
        eval_masking: bool = False,
    ):
        super().__init__()
        self.weight = weight
        self.normalize_target = normalize_target
        self.patchify_target = patchify_target
        self.patch_size = patch_size
        if patchify_target and patch_size is None:
            raise ValueError("If `patchify_target == True`, `patch_size` needs to be defined")
        self.eval_masking = eval_masking
        self.loss_fn = torch.nn.MSELoss(reduction="none")

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, masks: torch.Tensor
    ) -> torch.Tensor:
        target = target.detach()
        if self.patchify_target:
            target = rearrange(
                target,
                "b c (h1 h2) (w1 w2) -> b (h1 w1) (h2 w2 c)",
                h2=self.patch_size,
                w2=self.patch_size,
            )
        if self.normalize_target:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True, unbiased=False)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = self.loss_fn(input, target)
        loss = loss.mean(-1)  # Average per-patch

        if self.training or self.eval_masking:
            loss = (loss * masks).sum() / masks.sum()  # Compute loss only from masked patches
        else:
            loss = loss.mean()

        return self.weight * loss


class LatentDupplicateSuppressionLoss(nn.Module):
    """Latent Dupplicate Suppression Loss.

    Inspired by: Li et al, Duplicate latent representation suppression
      for multi-object variational autoencoders, BMVC 2021
    """

    def __init__(
        self,
        weight: float,
        eps: float = 1e-08,
    ):
        """Initialize LatentDupplicateSuppressionLoss.

        Args:
            weight: Weight of loss, output is multiplied with this value.
            eps: Small value to avoid division by zero in cosine similarity computation.
        """
        super().__init__()
        self.weight = weight
        self.similarity = nn.CosineSimilarity(dim=-1, eps=eps)

    def forward(self, grouping: ocl.typing.PerceptualGroupingOutput) -> float:
        """Compute latent dupplicate suppression loss.

        This also takes into account the `is_empty` tensor of
        [ocl.typing.PerceptualGroupingOutput][].

        Args:
            grouping: Grouping to use for loss computation.

        Returns:
            The weighted loss.
        """
        if grouping.objects.dim() == 4:
            # Build large tensor of reconstructed video.
            objects = grouping.objects
            bs, n_frames, n_objects, n_features = objects.shape

            off_diag_indices = torch.triu_indices(
                n_objects, n_objects, offset=1, device=objects.device
            )

            sq_similarities = (
                self.similarity(
                    objects[:, :, off_diag_indices[0], :], objects[:, :, off_diag_indices[1], :]
                )
                ** 2
            )

            if grouping.is_empty is not None:
                p_not_empty = 1.0 - grouping.is_empty
                # Assume that the probability of of individual objects being present is independent,
                # thus the probability of both being present is the product of the individual
                # probabilities.
                p_pair_present = (
                    p_not_empty[..., off_diag_indices[0]] * p_not_empty[..., off_diag_indices[1]]
                )
                # Use average expected penalty as loss for each frame.
                losses = (sq_similarities * p_pair_present) / torch.sum(
                    p_pair_present, dim=-1, keepdim=True
                )
            else:
                losses = sq_similarities.mean(dim=-1)

            return self.weight * losses.sum() / (bs * n_frames)
        elif grouping.objects.dim() == 3:
            # Build large tensor of reconstructed image.
            objects = grouping.objects
            bs, n_objects, n_features = objects.shape

            off_diag_indices = torch.triu_indices(
                n_objects, n_objects, offset=1, device=objects.device
            )

            sq_similarities = (
                self.similarity(
                    objects[:, off_diag_indices[0], :], objects[:, off_diag_indices[1], :]
                )
                ** 2
            )

            if grouping.is_empty is not None:
                p_not_empty = 1.0 - grouping.is_empty
                # Assume that the probability of of individual objects being present is independent,
                # thus the probability of both being present is the product of the individual
                # probabilities.
                p_pair_present = (
                    p_not_empty[..., off_diag_indices[0]] * p_not_empty[..., off_diag_indices[1]]
                )
                # Use average expected penalty as loss for each frame.
                losses = (sq_similarities * p_pair_present) / torch.sum(
                    p_pair_present, dim=-1, keepdim=True
                )
            else:
                losses = sq_similarities.mean(dim=-1)

            return self.weight * losses.sum() / bs
        else:
            raise ValueError("Incompatible input format.")


def _focal_loss(
    inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, mean_in_dim1=True
):
    """Loss used in RetinaNet for dense detection. # noqa: D411.

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if mean_in_dim1:
        return loss.mean(1).sum() / num_boxes
    else:
        return loss.sum() / num_boxes


def _compute_detr_cost_matrix(
    outputs,
    targets,
    use_focal=True,
    class_weight: float = 1,
    bbox_weight: float = 1,
    giou_weight: float = 1,
):
    """Compute cost matrix between outputs instances and target instances.

    Params:
        outputs: This is a dict that contains at least these entries:
                "pred_logits": Tensor of dim [batch_size, num_queries, num_classes]
                            with the classification logits
                "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the
                            predicted box coordinates

        targets: a list of targets (len(targets) = batch_size), where each target is a instance:
                "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                        ground-truth objects in the target) containing the class labels
                "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

    Returns:
        costMatrix: A iter of tensors of size [num_outputs, num_targets].
    """
    with torch.no_grad():
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if use_focal:
            out_prob = outputs["pred_logits"].flatten(0, 1)
        else:
            AssertionError("only support focal for now.")
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost.
        if use_focal:
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
        else:
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(  # noqa: F821
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)  # noqa: F821
        )

        # Final cost matrix
        C = bbox_weight * cost_bbox + class_weight * cost_class + giou_weight * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]

        return C.split(sizes, -1)


class CLIPLoss(nn.Module):
    """Contrastive CLIP loss.

    Reference:
        Radford et al.,
        Learning transferable visual models from natural language supervision,
        ICML 2021
    """

    def __init__(
        self,
        normalize_inputs: bool = True,
        learn_scale: bool = True,
        max_temperature: Optional[float] = None,
    ):
        """Initiailize CLIP loss.

        Args:
            normalize_inputs: Normalize both inputs based on mean and variance.
            learn_scale: Learn scaling factor of dot product.
            max_temperature: Maximum temperature of scaling.
        """
        super().__init__()
        self.normalize_inputs = normalize_inputs
        if learn_scale:
            self.logit_scale = nn.Parameter(torch.zeros([]) * log(1 / 0.07))  # Same init as CLIP.
        else:
            self.register_buffer("logit_scale", torch.zeros([]))  # exp(0) = 1, i.e. no scaling.
        self.max_temperature = max_temperature

    def forward(
        self,
        first: ocl.typing.PooledFeatures,
        second: ocl.typing.PooledFeatures,
        model: Optional[pl.LightningModule] = None,
    ) -> Tuple[float, Dict[str, torch.Tensor]]:
        """Compute CLIP loss.

        Args:
            first: First tensor.
            second: Second tensor.
            model: Pytorch lighting model. This is needed in order to perform
                multi-gpu / multi-node communication independent of the backend.

        Returns:
            - Computed loss
            - Dict with keys `similarity` (containing local similarities)
                and `temperature` (containing the current temperature).
        """
        # Collect all representations.
        if self.normalize_inputs:
            first = first / first.norm(dim=-1, keepdim=True)
            second = second / second.norm(dim=-1, keepdim=True)

        temperature = self.logit_scale.exp()
        if self.max_temperature:
            temperature = torch.clamp_max(temperature, self.max_temperature)

        if model is not None and hasattr(model, "trainer") and model.trainer.world_size > 1:
            # Running on multiple GPUs.
            global_rank = model.global_rank
            all_first_rep, all_second_rep = model.all_gather([first, second], sync_grads=True)
            world_size, batch_size = all_first_rep.shape[:2]
            labels = (
                torch.arange(batch_size, dtype=torch.long, device=first.device)
                + batch_size * global_rank
            )
            # Flatten the GPU dim into batch.
            all_first_rep = all_first_rep.flatten(0, 1)
            all_second_rep = all_second_rep.flatten(0, 1)

            # Compute inner product for instances on the current GPU.
            logits_per_first = temperature * first @ all_second_rep.t()
            logits_per_second = temperature * second @ all_first_rep.t()

            # For visualization purposes, return the cosine similarities on the local batch.
            similarities = (
                1
                / temperature
                * logits_per_first[:, batch_size * global_rank : batch_size * (global_rank + 1)]
            )
            # shape = [local_batch_size, global_batch_size]
        else:
            batch_size = first.shape[0]
            labels = torch.arange(batch_size, dtype=torch.long, device=first.device)
            # When running with only a single GPU we can save some compute time by reusing
            # computations.
            logits_per_first = temperature * first @ second.t()
            logits_per_second = logits_per_first.t()
            similarities = 1 / temperature * logits_per_first

        return (
            (F.cross_entropy(logits_per_first, labels) + F.cross_entropy(logits_per_second, labels))
            / 2,
            {"similarities": similarities, "temperature": temperature},
        )


def _compute_detr_seg_const_matrix(
    predicts,
    targets,
):
    """Compute cost matrix between outputs instances and target instances.

    Returns:
        costMatrix: A iter of tensors of size [num_outputs, num_targets].
    """
    # filter out valid targets
    npr, h, w = predicts.shape
    nt = targets.shape[0]

    predicts = repeat(predicts, "npr h w -> (npr repeat) h w", repeat=nt)
    targets = repeat(targets, "nt h w -> (repeat nt) h w", repeat=npr)

    cost = F.binary_cross_entropy(predicts, targets.float(), reduction="none").mean(-1).mean(-1)
    cost = rearrange(cost, "(npr nt) -> npr nt", npr=npr, nt=nt)
    return cost


class DETRSegLoss(nn.Module):
    """DETR inspired loss for segmentation.

    This loss computes a hungarian matching of segmentation masks between a prediction and
    a target.  The loss is then a linear combination of the CE loss between matched masks
    and a foreground prediction classification.

    Reference:
        Carion et al., End-to-End Object Detection with Transformers, ECCV 2020
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        ignore_background: bool = True,
        foreground_weight: float = 1.0,
        foreground_matching_weight: float = 1.0,
        global_loss: bool = True,
    ):
        """Initialize DETRSegLoss.

        Args:
            loss_weight: Loss weight
            ignore_background: Ignore background masks.
            foreground_weight: Contribution weight of foreground classification loss.
            foreground_matching_weight: Contribution weight of foreground classification
                to matching.
            global_loss: Use average loss over all instances of all gpus.  This is
                particularly useful when training with sparse labels.
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.ignore_background = ignore_background
        self.foreground_weight = foreground_weight
        self.foreground_matching_weight = foreground_matching_weight
        self.global_loss = global_loss
        self.matcher = CPUHungarianMatcher()

    def forward(
        self,
        input_mask: ocl.typing.ObjectFeatureAttributions,
        target_mask: ocl.typing.ObjectFeatureAttributions,
        foreground_logits: Optional[torch.Tensor] = None,
        model: Optional[pl.LightningModule] = None,
    ) -> float:
        """Compute DETR segmentation loss.

        Args:
            input_mask: Input/predicted masks
            target_mask: Target masks
            foreground_logits: Forground prediction logits
            model: Pytorch lighting model. This is needed in order to perform
                multi-gpu / multi-node communication independent of the backend.

        Returns:
            The computed loss.
        """
        target_mask = target_mask.detach() > 0
        device = target_mask.device

        # A nan mask is not considered.
        valid_targets = ~(target_mask.isnan().all(-1).all(-1)).any(-1)
        # Discard first dimension mask as it is background.
        if self.ignore_background:
            # Assume first class in masks is background.
            if len(target_mask.shape) > 4:  # Video data (bs, frame, classes, w, h).
                target_mask = target_mask[:, :, 1:]
            else:  # Image data (bs, classes, w, h).
                target_mask = target_mask[:, 1:]

        targets = target_mask[valid_targets]
        predictions = input_mask[valid_targets]
        if foreground_logits is not None:
            foreground_logits = foreground_logits[valid_targets]

        total_loss = torch.tensor(0.0, device=device)
        num_samples = 0

        # Iterate through each clip. Might think about if parallelable
        for i, (prediction, target) in enumerate(zip(predictions, targets)):
            # Filter empty masks.
            target = target[target.sum(-1).sum(-1) > 0]

            # Compute matching.
            costMatrixSeg = _compute_detr_seg_const_matrix(
                prediction,
                target,
            )
            # We cannot rely on the matched cost for computing the loss due to
            # normalization issues between segmentation component (normalized by
            # number of matches) and classification component (normalized by
            # number of predictions). Thus compute both components separately
            # after deriving the matching matrix.
            if foreground_logits is not None and self.foreground_matching_weight != 0.0:
                # Positive classification component.
                logits = foreground_logits[i]
                costMatrixTotal = (
                    costMatrixSeg
                    + self.foreground_weight
                    * F.binary_cross_entropy_with_logits(
                        logits, torch.ones_like(logits), reduction="none"
                    ).detach()
                )
            else:
                costMatrixTotal = costMatrixSeg

            # Matcher takes a batch but we are doing this one by one.
            matching_matrix = self.matcher(costMatrixTotal.unsqueeze(0))[0].squeeze(0)
            n_matches = min(predictions.shape[0], target.shape[0])
            if n_matches > 0:
                instance_cost = (costMatrixSeg * matching_matrix).sum(-1).sum(-1) / n_matches
            else:
                instance_cost = torch.tensor(0.0, device=device)

            if foreground_logits is not None:
                ismatched = (matching_matrix > 0).any(-1)
                logits = foreground_logits[i].squeeze(-1)
                instance_cost += self.foreground_weight * F.binary_cross_entropy_with_logits(
                    logits, ismatched.float(), reduction="mean"
                )

            total_loss += instance_cost
            # Normalize by number of matches.
            num_samples += 1

        if (
            model is not None
            and hasattr(model, "trainer")
            and model.trainer.world_size > 1
            and self.global_loss
        ):
            # As data is sparsely labeled return the average loss over all GPUs.
            # This should make the loss a mit more smooth.
            all_losses, sample_counts = model.all_gather([total_loss, num_samples], sync_grads=True)
            total_count = sample_counts.sum()
            if total_count > 0:
                total_loss = all_losses.sum() / total_count
            else:
                total_loss = torch.tensor(0.0, device=device)

            return total_loss * self.loss_weight
        else:
            if num_samples == 0:
                # Avoid division by zero if a batch does not contain any labels.
                return torch.tensor(0.0, device=targets.device)

            total_loss /= num_samples
            total_loss *= self.loss_weight
            return total_loss


class EM_rec_loss(nn.Module):
    def __init__(
        self,
        loss_weight: float = 20,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.loss_fn = lambda x1, x2: nn.functional.mse_loss(x1, x2, reduction="none")

    def forward(
        self,
        segmentations: torch.Tensor,  # rollout_decode.masks
        masks: torch.Tensor,  # decoder.masks
        reconstructions: torch.Tensor,
        rec_tgt: torch.Tensor,
        masks_vis: torch.Tensor,
        attn_index: torch.Tensor,
    ):
        b, f, c, h, w = segmentations.shape
        _, _, n_slots, n_buffer = attn_index.shape

        segmentations = (
            segmentations.reshape(-1, n_buffer, h, w).unsqueeze(1).repeat(1, n_slots, 1, 1, 1)
        )
        masks = masks.reshape(-1, n_slots, h, w).unsqueeze(2).repeat(1, 1, n_buffer, 1, 1)
        masks = masks > 0.5
        masks_vis = (
            masks_vis.reshape(-1, n_slots, h, w)
            .unsqueeze(2)
            .unsqueeze(3)
            .repeat(1, 1, n_buffer, 3, 1, 1)
        )
        masks_vis = masks_vis > 0.5
        attn_index = attn_index.reshape(-1, n_slots, n_buffer)
        rec_tgt = (
            rec_tgt.reshape(-1, 3, h, w)
            .unsqueeze(1)
            .unsqueeze(2)
            .repeat(1, n_slots, n_buffer, 1, 1, 1)
        )
        reconstructions = (
            reconstructions.reshape(-1, n_buffer, 3, h, w)
            .unsqueeze(1)
            .repeat(1, n_slots, 1, 1, 1, 1)
        )
        rec_pred = reconstructions * masks_vis
        rec_tgt_ = rec_tgt * masks_vis
        loss = torch.sum(
            F.binary_cross_entropy(segmentations, masks.float(), reduction="none"), (-1, -2)
        ) / (h * w) + 0.1 * torch.sum(self.loss_fn(rec_pred, rec_tgt_), (-3, -2, -1))
        total_loss = torch.sum(attn_index * loss, (0, 1, 2)) / (b * f * n_slots * n_buffer)
        return (total_loss) * self.loss_weight
