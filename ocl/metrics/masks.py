"""Metrics related to the evaluation of masks."""
from typing import Optional, Tuple

import scipy.optimize
import torch
import torchmetrics

from ocl.metrics.utils import adjusted_rand_index, fg_adjusted_rand_index, tensor_to_one_hot
from ocl.utils.resizing import resize_patches_to_image


class ARIMetric(torchmetrics.Metric):
    """Computes ARI metric."""

    def __init__(
        self,
        foreground: bool = True,
        convert_target_one_hot: bool = False,
        ignore_overlaps: bool = False,
        min_true_classes: int = 0,
    ):
        super().__init__()
        self.foreground = foreground
        self.convert_target_one_hot = convert_target_one_hot
        self.ignore_overlaps = ignore_overlaps
        self.min_true_classes = min_true_classes
        self.add_state(
            "values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self, prediction: torch.Tensor, target: torch.Tensor, ignore: Optional[torch.Tensor] = None
    ):
        """Update this metric.

        Args:
            prediction: Predicted mask of shape (B, C, H, W) or (B, F, C, H, W), where C is the
                number of classes.
            target: Ground truth mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the
                number of classes.
            ignore: Ignore mask of shape (B, 1, H, W) or (B, 1, K, H, W)
        """
        if prediction.ndim == 5:
            # Merge frames, height and width to single dimension.
            prediction = prediction.transpose(1, 2).flatten(-3, -1)
            target = target.transpose(1, 2).flatten(-3, -1)
            if ignore is not None:
                ignore = ignore.to(torch.bool).transpose(1, 2).flatten(-3, -1)
        elif prediction.ndim == 4:
            # Merge height and width to single dimension.
            prediction = prediction.flatten(-2, -1)
            target = target.flatten(-2, -1)
            if ignore is not None:
                ignore = ignore.to(torch.bool).flatten(-2, -1)
        else:
            raise ValueError(f"Incorrect input shape: f{prediction.shape}")

        if self.ignore_overlaps:
            overlaps = (target > 0).sum(1, keepdim=True) > 1
            if ignore is None:
                ignore = overlaps
            else:
                ignore = ignore | overlaps

        if ignore is not None:
            assert ignore.ndim == 3 and ignore.shape[1] == 1
            prediction = prediction.clone()
            prediction[ignore.expand_as(prediction)] = 0
            target = target.clone()
            target[ignore.expand_as(target)] = 0

        # Make channels / gt labels the last dimension.
        prediction = prediction.transpose(-2, -1)
        target = target.transpose(-2, -1)

        if self.convert_target_one_hot:
            target_oh = tensor_to_one_hot(target, dim=2)
            # For empty pixels (all values zero), one-hot assigns 1 to the first class, correct for
            # this (then it is technically not one-hot anymore).
            target_oh[:, :, 0][target.sum(dim=2) == 0] = 0
            target = target_oh

        # Should be either 0 (empty, padding) or 1 (single object).
        assert torch.all(target.sum(dim=-1) < 2), "Issues with target format, mask non-exclusive"

        if self.foreground:
            ari = fg_adjusted_rand_index(prediction, target)
        else:
            ari = adjusted_rand_index(prediction, target)

        if self.min_true_classes > 0:
            num_true_classes = (target.sum(1) > 0).sum(-1)
            ari = ari[num_true_classes >= self.min_true_classes]

        self.values += ari.sum()
        self.total += len(ari)

    def compute(self) -> torch.Tensor:
        return self.values / self.total


class PatchARIMetric(ARIMetric):
    """Computes ARI metric assuming patch masks as input."""

    def __init__(
        self,
        foreground=True,
        resize_masks_mode: str = "bilinear",
        **kwargs,
    ):
        super().__init__(foreground=foreground, **kwargs)
        self.resize_masks_mode = resize_masks_mode

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        """Update this metric.

        Args:
            prediction: Predicted mask of shape (B, C, P) or (B, F, C, P), where C is the
                number of classes and P the number of patches.
            target: Ground truth mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the
                number of classes.
        """
        h, w = target.shape[-2:]
        assert h == w

        prediction_resized = resize_patches_to_image(
            prediction, size=h, resize_mode=self.resize_masks_mode
        )

        return super().update(prediction=prediction_resized, target=target)


class BindingHits(torchmetrics.Metric):
    def __init__(
        self,
        use_threshold: bool = False,
        threshold: float = 0.5,
        matching: str = "hungarian",
        average_per_image: bool = True,
        compute_discovery_fraction: bool = False,
        correct_localization: bool = False,
        compute_panoptic_quality: bool = False,
        compute_panoptic_segmentation_quality: bool = False,
        compute_panoptic_recognition_quality: bool = False,
        discovery_threshold: float = 0.5,
        ignore_background: bool = False,
        ignore_overlaps: bool = False,
        filter_void_predictions: bool = False,
    ):
        super().__init__()
        self.use_threshold = use_threshold
        self.threshold = threshold
        self.discovery_threshold = discovery_threshold
        self.average_per_image = average_per_image
        self.compute_discovery_fraction = compute_discovery_fraction
        self.correct_localization = correct_localization
        if compute_discovery_fraction and correct_localization:
            raise ValueError(
                "Only one of `compute_discovery_fraction` and `correct_localization` can be enabled."
            )
        self.compute_panoptic_quality = compute_panoptic_quality
        self.compute_panoptic_segmentation_quality = compute_panoptic_segmentation_quality
        self.compute_panoptic_recognition_quality = compute_panoptic_recognition_quality
        is_panoptic = (
            compute_panoptic_quality
            or compute_panoptic_segmentation_quality
            or compute_panoptic_recognition_quality
        )
        if is_panoptic:
            if matching != "threshold":
                raise ValueError("For panoptic metrics, matching must be 'threshold'")
            if average_per_image:
                raise ValueError("For panoptic metrics, average_per_image must be False")

        matchings = ("hungarian", "best_overlap", "threshold")
        if matching not in matchings:
            raise ValueError(f"Unknown matching type {matching}. Valid values are {matchings}.")
        self.matching = matching
        self.ignore_background = ignore_background
        self.ignore_overlaps = ignore_overlaps
        self.filter_void_predictions = filter_void_predictions

        self.add_state("hits", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        selected_indices: torch.Tensor,
        ignore: Optional[torch.Tensor] = None,
    ):
        """Update this metric.

        Args:
            prediction: Predicted mask of shape (B, C, H, W) or (B, F, C, H, W), where C is the
                number of classes. Assumes class probabilities as inputs.
            target: Ground truth mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the
                number of classes.
            ignore: Ignore mask of shape (B, 1, H, W) or (B, 1, K, H, W)
        """
        if prediction.ndim == 5:
            # Merge frames, height and width to single dimension.
            predictions = prediction.transpose(1, 2).flatten(-3, -1)
            targets = target.transpose(1, 2).flatten(-3, -1)
            if ignore is not None:
                ignore = ignore.to(torch.bool).transpose(1, 2).flatten(-3, -1)
        elif prediction.ndim == 4:
            # Merge height and width to single dimension.
            predictions = prediction.flatten(-2, -1)
            targets = target.flatten(-2, -1)
            if ignore is not None:
                ignore = ignore.to(torch.bool).flatten(-2, -1)
        else:
            raise ValueError(f"Incorrect input shape: f{prediction.shape}")

        if self.use_threshold:
            predictions = predictions > self.threshold
        else:
            indices = torch.argmax(predictions, dim=1)
            predictions = torch.nn.functional.one_hot(indices, num_classes=predictions.shape[1])
            predictions = predictions.transpose(1, 2)

        if self.ignore_background:
            targets = targets[:, 1:]

        targets = targets > 0  # Ensure masks are binary

        if self.ignore_overlaps:
            overlaps = targets.sum(1, keepdim=True) > 1
            if ignore is None:
                ignore = overlaps
            else:
                ignore = ignore | overlaps

        predictions_orig = predictions
        if ignore is not None:
            assert ignore.ndim == 3 and ignore.shape[1] == 1
            predictions[ignore.expand_as(predictions)] = 0
            targets[ignore.expand_as(targets)] = 0

        targets_empty = targets.sum(dim=1) == 0  # B x P

        # Should be either 0 (empty, padding) or 1 (single object).
        # assert torch.all(targets.sum(dim=1) < 2), "Issues with target format, mask non-exclusive"

        for pred, pred_orig, target, target_empty, indices in zip(
            predictions, predictions_orig, targets, targets_empty, selected_indices
        ):
            if len(target) == 0:
                continue  # Skip elements without any target mask

            assert pred.ndim == 2
            assert target.ndim == 2

            pred_mask = pred.unsqueeze(1).to(torch.bool)
            true_mask = target.unsqueeze(0).to(torch.bool)

            intersection = torch.sum(pred_mask & true_mask, dim=-1).to(torch.float64)
            union = torch.sum(pred_mask | true_mask, dim=-1).to(torch.float64)
            pairwise_iou = intersection / union
            n_pred_classes, n_true_classes = pairwise_iou.shape

            # Remove NaN from divide-by-zero: class does not occur, and class was not predicted.
            pairwise_iou[union == 0] = 0.0

            pred_idxs, true_idxs = scipy.optimize.linear_sum_assignment(
                pairwise_iou.cpu(), maximize=True
            )

            pred_idxs_t = torch.as_tensor(pred_idxs, dtype=torch.int64, device=pairwise_iou.device)
            true_idxs_t = torch.as_tensor(true_idxs, dtype=torch.int64, device=pairwise_iou.device)

            for i in range(min(len(true_idxs_t), len(indices))):
                if indices[i] != -1:
                    if true_idxs_t[i] == indices[i]:
                        self.hits += 1

                    self.total += 1

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.zeros_like(self.hits)
        else:
            return self.hits / self.total


class UnsupervisedMaskIoUMetric(torchmetrics.Metric):
    """Computes IoU metric for segmentation masks when correspondences to ground truth are not known.

    Offers different matching approaches to compute the assignment between predicted classes and
    ground truth classes.

    Args:
        use_threshold: If `True`, convert predicted class probabilities to mask using a threshold.
            If `False`, class probabilities are turned into mask using a softmax instead.
        threshold: Value to use for thresholding masks.
        matching: Approach to match predicted to ground truth classes. For "hungarian", computes
            assignment that maximizes total IoU between all classes. For "best_overlap", uses the
            predicted class with maximum overlap for each ground truth class. Using "best_overlap"
            leads to the "average best overlap" metric. For "threshold", counts every pair with IoU
            larger than `discovery_threshold` as a match.
        compute_discovery_fraction: Instead of the IoU, compute the fraction of ground truth classes
            that were "discovered", meaning that they have an IoU greater than some threshold.
        correct_localization: Instead of the IoU, compute the fraction of images on which at least
            one ground truth class was correctly localised, meaning that they have an IoU
            greater than some threshold.
        compute_panoptic_quality: Compute panoptic quality.
        compute_panoptic_segmentation_quality: Compute panoptic segmentation quality.
        compute_panoptic_recognition_quality: Compute panoptic recognition quality.
        discovery_threshold: Minimum IoU to count a class as discovered/correctly localized.
        ignore_background: If true, assume class at index 0 of ground truth masks is background class
            that is removed before computing IoU.
        ignore_overlaps: If true, remove points where ground truth masks has overlappign classes from
            predictions and ground truth masks.
        filter_void_predictions: If true, do not count predictions as false positives that have
            significant fraction of empty pixels.
    """

    def __init__(
        self,
        use_threshold: bool = False,
        threshold: float = 0.5,
        matching: str = "hungarian",
        average_per_image: bool = True,
        compute_discovery_fraction: bool = False,
        correct_localization: bool = False,
        compute_panoptic_quality: bool = False,
        compute_panoptic_segmentation_quality: bool = False,
        compute_panoptic_recognition_quality: bool = False,
        discovery_threshold: float = 0.5,
        ignore_background: bool = False,
        ignore_overlaps: bool = False,
        filter_void_predictions: bool = False,
    ):
        super().__init__()
        self.use_threshold = use_threshold
        self.threshold = threshold
        self.discovery_threshold = discovery_threshold
        self.average_per_image = average_per_image
        self.compute_discovery_fraction = compute_discovery_fraction
        self.correct_localization = correct_localization
        if compute_discovery_fraction and correct_localization:
            raise ValueError(
                "Only one of `compute_discovery_fraction` and `correct_localization` can be enabled."
            )
        self.compute_panoptic_quality = compute_panoptic_quality
        self.compute_panoptic_segmentation_quality = compute_panoptic_segmentation_quality
        self.compute_panoptic_recognition_quality = compute_panoptic_recognition_quality
        is_panoptic = (
            compute_panoptic_quality
            or compute_panoptic_segmentation_quality
            or compute_panoptic_recognition_quality
        )
        if is_panoptic:
            if matching != "threshold":
                raise ValueError("For panoptic metrics, matching must be 'threshold'")
            if average_per_image:
                raise ValueError("For panoptic metrics, average_per_image must be False")

        matchings = ("hungarian", "best_overlap", "threshold")
        if matching not in matchings:
            raise ValueError(f"Unknown matching type {matching}. Valid values are {matchings}.")
        self.matching = matching
        self.ignore_background = ignore_background
        self.ignore_overlaps = ignore_overlaps
        self.filter_void_predictions = filter_void_predictions

        self.add_state("iou", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum")
        self.add_state(
            "iou_tp", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "true_positives", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "any_true_positives",
            default=torch.tensor(0.0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "false_positives", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "false_negatives", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        ignore: Optional[torch.Tensor] = None,
        selected_indices: torch.Tensor = None,
    ):
        """Update this metric.

        Args:
            prediction: Predicted mask of shape (B, C, H, W) or (B, F, C, H, W), where C is the
                number of classes. Assumes class probabilities as inputs.
            target: Ground truth mask of shape (B, K, H, W) or (B, F, K, H, W), where K is the
                number of classes.
            ignore: Ignore mask of shape (B, 1, H, W) or (B, 1, K, H, W)
        """
        if prediction.ndim == 5:
            # Merge frames, height and width to single dimension.
            predictions = prediction.transpose(1, 2).flatten(-3, -1)
            targets = target.transpose(1, 2).flatten(-3, -1)
            if ignore is not None:
                ignore = ignore.to(torch.bool).transpose(1, 2).flatten(-3, -1)
        elif prediction.ndim == 4:
            # Merge height and width to single dimension.
            predictions = prediction.flatten(-2, -1)
            targets = target.flatten(-2, -1)
            if ignore is not None:
                ignore = ignore.to(torch.bool).flatten(-2, -1)
        else:
            raise ValueError(f"Incorrect input shape: f{prediction.shape}")

        if self.use_threshold:
            predictions = predictions > self.threshold
        else:
            indices = torch.argmax(predictions, dim=1)
            predictions = torch.nn.functional.one_hot(indices, num_classes=predictions.shape[1])
            predictions = predictions.transpose(1, 2)

        if self.ignore_background:
            targets = targets[:, 1:]

        targets = targets > 0  # Ensure masks are binary

        if self.ignore_overlaps:
            overlaps = targets.sum(1, keepdim=True) > 1
            if ignore is None:
                ignore = overlaps
            else:
                ignore = ignore | overlaps

        predictions_orig = predictions
        if ignore is not None:
            assert ignore.ndim == 3 and ignore.shape[1] == 1
            predictions[ignore.expand_as(predictions)] = 0
            targets[ignore.expand_as(targets)] = 0

        targets_empty = targets.sum(dim=1) == 0  # B x P

        # Should be either 0 (empty, padding) or 1 (single object).
        assert torch.all(targets.sum(dim=1) < 2), "Issues with target format, mask non-exclusive"
        idx = 0

        for pred, pred_orig, target, target_empty in zip(
            predictions, predictions_orig, targets, targets_empty
        ):
            nonzero_classes = torch.sum(target, dim=-1) > 0
            target = target[nonzero_classes]  # Remove empty (e.g. padded) classes
            if len(target) == 0:
                continue  # Skip elements without any target mask

            if selected_indices is not None:
                indices = selected_indices[idx].clone()
                indices[indices >= 0] = 1
                indices[indices < 0] = 0
                sum_indices = torch.sum(indices)
                pred = pred[:sum_indices]
                idx += 1

            iou_per_class, iou_per_pred = unsupervised_mask_iou(
                pred,
                target,
                matching=self.matching,
                reduction="none",
                matching_threshold=self.discovery_threshold,
            )

            iou = iou_per_class.sum()
            discovered = iou_per_class > self.discovery_threshold
            iou_tp = iou_per_class[discovered].sum()
            tp = discovered.sum()
            any_tp = torch.any(discovered).sum()
            fn = (~discovered).sum()

            undiscovered_pred = iou_per_pred <= self.discovery_threshold
            if self.filter_void_predictions:
                # For the panoptic quality metric, unmatched predictions that contain a large
                # fraction of void or crowd pixels are removed from the set of false positives.
                # Void or crowd pixels are supposed to be removed from the target masks after
                # applying the ignore mask.
                intersec = (pred_orig.to(torch.bool) & target_empty.unsqueeze(0)).sum(-1)
                pred_area = pred_orig.sum(-1)
                matching = (intersec / pred_area) > self.discovery_threshold
                undiscovered_pred = undiscovered_pred & ~matching
            fp = undiscovered_pred.sum()

            if self.average_per_image:
                iou = iou / len(iou_per_class)
                iou_tp = iou_tp / len(discovered)
                tp = tp / len(iou_per_class)
                fn = fn / len(iou_per_class)
                fp = fp / len(undiscovered_pred)
                total = 1
            else:
                total = len(iou_per_class)

            self.iou += iou
            self.iou_tp += iou_tp
            self.true_positives += tp
            self.any_true_positives += any_tp
            self.false_negatives += fn
            self.false_positives += fp
            self.total += total

    def compute(self) -> torch.Tensor:
        if self.total == 0:
            return torch.zeros_like(self.iou)
        elif self.compute_discovery_fraction:
            return self.true_positives / self.total
        elif self.correct_localization:
            return self.any_true_positives / self.total
        elif self.compute_panoptic_quality:
            return self.iou_tp / (
                self.true_positives + 0.5 * self.false_positives + 0.5 * self.false_negatives
            )
        elif self.compute_panoptic_segmentation_quality:
            return self.iou_tp / self.true_positives
        elif self.compute_panoptic_recognition_quality:
            return self.true_positives / (
                self.true_positives + 0.5 * self.false_positives + 0.5 * self.false_negatives
            )
        else:
            return self.iou / self.total


class MaskCorLocMetric(UnsupervisedMaskIoUMetric):
    def __init__(self, **kwargs):
        super().__init__(matching="best_overlap", correct_localization=True, **kwargs)


class AverageBestOverlapMetric(UnsupervisedMaskIoUMetric):
    def __init__(self, **kwargs):
        super().__init__(matching="best_overlap", **kwargs)


class BestOverlapObjectRecoveryMetric(UnsupervisedMaskIoUMetric):
    def __init__(self, **kwargs):
        super().__init__(matching="best_overlap", compute_discovery_fraction=True, **kwargs)


class PanopticQuality(UnsupervisedMaskIoUMetric):
    """Compute panoptic quality as used for panoptic segmentation.

    A note on how to handle special pixels for the metric:

    - Void pixels: are not counted in IoU computation.
      * Void pixels should be added to ignore mask in preprocessing.
    - Crowd pixels: are counted in IoU computation, but are not valid targets for matching.
      * Crowd masks should be removed from target masks in preprocessing.
    """

    def __init__(self, **kwargs):
        super().__init__(
            matching="threshold",
            average_per_image=False,
            compute_panoptic_quality=True,
            discovery_threshold=0.5,
            filter_void_predictions=True,
            **kwargs,
        )


class PanopticSegmentationQuality(UnsupervisedMaskIoUMetric):
    """Compute panoptic segmentation quality as used for panoptic segmentation."""

    def __init__(self, **kwargs):
        super().__init__(
            matching="threshold",
            average_per_image=False,
            compute_panoptic_segmentation_quality=True,
            discovery_threshold=0.5,
            filter_void_predictions=True,
            **kwargs,
        )


class PanopticRecognitionQuality(UnsupervisedMaskIoUMetric):
    """Compute panoptic recognition quality as used for panoptic segmentation."""

    def __init__(self, **kwargs):
        super().__init__(
            matching="threshold",
            average_per_image=False,
            compute_panoptic_recognition_quality=True,
            discovery_threshold=0.5,
            filter_void_predictions=True,
            **kwargs,
        )


def unsupervised_mask_iou(
    pred_mask: torch.Tensor,
    true_mask: torch.Tensor,
    matching: str = "hungarian",
    reduction: str = "mean",
    iou_empty: float = 0.0,
    matching_threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute intersection-over-union (IoU) between masks with unknown class correspondences.

    This metric is also known as Jaccard index. Note that this is a non-batched implementation.

    Args:
        pred_mask: Predicted mask of shape (C, N), where C is the number of predicted classes and
            N is the number of points. Masks are assumed to be binary.
        true_mask: Ground truth mask of shape (K, N), where K is the number of ground truth
            classes and N is the number of points. Masks are assumed to be binary.
        matching: How to match predicted classes to ground truth classes. For "hungarian", computes
            assignment that maximizes total IoU between all classes. For "best_overlap", uses the
            predicted/true class with maximum overlap for each true/predicted class (each
            predicted/true class can be assigned to multiple true/predicted classes). Empty
            true/predicted classes are assigned IoU of zero.
        reduction: If "mean", return IoU averaged over classes. If "none", return per-class IoU.
        iou_empty: IoU for the case when a class does not occur, but was also not predicted.
        matching_threshold: Threshold for matching mode 'threshold'.

    Returns:
        Tuple of tensors of shape (K,) and (C,), containing per-class IoU for all true classes and
        predicted classes respectively. If reduction is `mean`, return the average instead.
    """
    assert pred_mask.ndim == 2
    assert true_mask.ndim == 2
    pred_mask = pred_mask.unsqueeze(1).to(torch.bool)
    true_mask = true_mask.unsqueeze(0).to(torch.bool)

    intersection = torch.sum(pred_mask & true_mask, dim=-1).to(torch.float64)
    union = torch.sum(pred_mask | true_mask, dim=-1).to(torch.float64)
    pairwise_iou = intersection / union
    n_pred_classes, n_true_classes = pairwise_iou.shape

    # Remove NaN from divide-by-zero: class does not occur, and class was not predicted.
    pairwise_iou[union == 0] = iou_empty

    if matching == "hungarian":
        pred_idxs, true_idxs = scipy.optimize.linear_sum_assignment(
            pairwise_iou.cpu(), maximize=True
        )
        pred_idxs_t = torch.as_tensor(pred_idxs, dtype=torch.int64, device=pairwise_iou.device)
        true_idxs_t = torch.as_tensor(true_idxs, dtype=torch.int64, device=pairwise_iou.device)
        pred_idxs_p = pred_idxs_t
        true_idxs_p = true_idxs_t
    elif matching == "best_overlap":
        non_empty_gt = torch.sum(true_mask.squeeze(0), dim=1) > 0
        pred_idxs_t = torch.argmax(pairwise_iou, dim=0)[non_empty_gt]
        true_idxs_t = torch.arange(n_true_classes, device=pairwise_iou.device)[non_empty_gt]

        non_empty_pred = torch.sum(pred_mask.squeeze(1), dim=1) > 0
        true_idxs_p = torch.argmax(pairwise_iou, dim=1)[non_empty_pred]
        pred_idxs_p = torch.arange(n_pred_classes, device=pairwise_iou.device)[non_empty_pred]
    elif matching == "threshold":
        matched = pairwise_iou > 0.5
        assert torch.all(torch.sum(matched, dim=0) <= 1) and torch.all(
            torch.sum(matched, dim=1) <= 1
        )
        pred_iou, pred_idxs_t = torch.max(matched, dim=0)
        pred_idxs_t = pred_idxs_t[pred_iou > matching_threshold]
        true_idxs_t = torch.arange(n_true_classes, device=pairwise_iou.device)
        true_idxs_t = true_idxs_t[pred_iou > matching_threshold]

        true_iou, true_idxs_p = torch.max(matched, dim=1)
        true_idxs_p = true_idxs_p[true_iou > matching_threshold]
        pred_idxs_p = torch.arange(n_pred_classes, device=pairwise_iou.device)
        pred_idxs_p = pred_idxs_p[true_iou > matching_threshold]
    else:
        raise ValueError(f"Unknown matching {matching}")

    matched_iou = pairwise_iou[pred_idxs_t, true_idxs_t]
    iou_true = torch.zeros(n_true_classes, dtype=torch.float64, device=pairwise_iou.device)
    iou_true[true_idxs_t] = matched_iou

    matched_iou = pairwise_iou[pred_idxs_p, true_idxs_p]
    iou_pred = torch.zeros(n_pred_classes, dtype=torch.float64, device=pairwise_iou.device)
    iou_pred[pred_idxs_p] = matched_iou

    if reduction == "mean":
        return iou_true.mean(), iou_pred.mean()
    else:
        return iou_true, iou_pred
