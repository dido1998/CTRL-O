"""Metrics used for diagnosis."""
from typing import Optional, Union

import torch
import torchmetrics


def _flatten2d(tensor: torch.Tensor, num_batch_dimensions: int) -> torch.Tensor:
    """Flatten tensor into two dimensions, with the first `num_batch_dimensions` flattened."""
    # Flatten batch dimensions
    tensor = tensor.flatten(0, min(num_batch_dimensions, tensor.ndim) - 1)
    # Flatten data dimensions
    tensor = tensor.flatten(1, -1) if tensor.ndim > 1 else tensor.unsqueeze(-1)
    return tensor


class TensorStatistic(torchmetrics.Metric):
    """Metric that computes summary statistic of tensors for logging purposes.

    First `num_batch_dimensions` dimension of tensor are assumed to be batch dimensions (default 1).
    The other dimensions are reduced to a scalar by the chosen reduction approach.
    """

    def __init__(
        self,
        reduction: str = "mean",
        num_batch_dimensions: int = 1,
        batch_aggregation: bool = False,
        num_aggregation_samples: Optional[int] = None,
    ):
        super().__init__()
        if reduction not in (
            "sum",
            "mean",
            "max",
            "min",
            "std",
            "norm_normed",
            "batch_norm_normed",
            "batch_variance",
            "batch_covariance_squared_offdiag",
            "matrix_rank",
            "matrix_rank_soft",
        ):
            raise ValueError(f"Unknown reduction {reduction}")
        self.reduction = reduction
        self.num_batch_dimensions = num_batch_dimensions
        self.batch_aggregation = batch_aggregation
        self.num_aggregation_samples = num_aggregation_samples

        if self.batch_aggregation:
            self.add_state("values", default=[], dist_reduce_fx="cat")
        else:
            self.add_state(
                "values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
            )
            self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def aggregate(self, tensor: torch.Tensor):
        if self.reduction == "mean":
            tensor = torch.mean(tensor, dim=-1)
        elif self.reduction == "sum":
            tensor = torch.sum(tensor, dim=-1)
        elif self.reduction == "max":
            tensor = torch.max(tensor, dim=-1).values
        elif self.reduction == "min":
            tensor = torch.min(tensor, dim=-1).values
        elif self.reduction == "std":
            tensor = torch.std(tensor, dim=-1)
        elif self.reduction == "norm_normed":
            tensor = torch.norm(tensor, p=2, dim=-1) / tensor.shape[-1] ** 0.5
        elif self.reduction == "batch_norm_normed":
            tensor = torch.norm(tensor, p=2, dim=0) / tensor.shape[0] ** 0.5
            tensor = tensor.mean().unsqueeze(0)
        elif self.reduction == "batch_variance":
            tensor = torch.var(tensor, dim=0).mean().unsqueeze(0)
        elif self.reduction == "batch_covariance_squared_offdiag":
            cov = torch.cov(tensor.T)  # torch.cov treats dim=-1 as sample dimension, so transpose.
            offdiag = cov[~torch.eye(len(cov), dtype=torch.bool, device=cov.device)]
            tensor = (offdiag**2).mean().unsqueeze(0)
        elif self.reduction == "matrix_rank":
            tensor = torch.linalg.matrix_rank(tensor).unsqueeze(0)
        elif self.reduction == "matrix_rank_soft":
            vals = torch.linalg.svdvals(tensor)  # min(B, D)
            dist = (vals / vals.norm(p=1)) + 1e-7
            tensor = torch.exp(-torch.sum(dist * dist.log())).unsqueeze(0)

        return tensor

    def update(self, tensor: torch.Tensor):
        tensor = _flatten2d(tensor, self.num_batch_dimensions)

        if self.batch_aggregation:
            # We only store up to `num_aggregation_samples` elements until reset() is called.
            n_remaining = max(self.num_aggregation_samples - len(self.values), 0)
            self.values.extend([elem.unsqueeze(0) for elem in tensor[:n_remaining]])
        else:
            tensor = self.aggregate(tensor)
            self.values += tensor.to(torch.float64).sum()
            self.total += len(tensor)

    def compute(self) -> torch.Tensor:
        if self.batch_aggregation:
            values = torchmetrics.utilities.dim_zero_cat(self.values)
            return self.aggregate(values).to(torch.float64).mean()
        else:
            return self.values / self.total


class TwoTensorStatistic(torchmetrics.Metric):
    """Metric that computes summary statistic between two tensors for logging purposes.

    First `num_batch_dimensions` dimension of tensor are assumed to be batch dimensions (default 1).
    """

    def __init__(self, reduction: str, num_batch_dimensions: int = 1):
        super().__init__()
        if reduction not in ("mse", "norm_diff"):
            raise ValueError(f"Unknown statistic {reduction}")
        self.reduction = reduction
        self.num_batch_dimensions = num_batch_dimensions
        self.add_state(
            "values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, tensor1: torch.Tensor, tensor2: torch.Tensor):
        if tensor1.shape != tensor2.shape:
            raise ValueError(
                f"Tensors need to have the same shape, but got {tensor1.shape} and {tensor2.shape}"
            )
        tensor1 = _flatten2d(tensor1, self.num_batch_dimensions)
        tensor2 = _flatten2d(tensor2, self.num_batch_dimensions)

        if self.reduction == "mse":
            tensor = torch.mean((tensor1 - tensor2) ** 2, dim=-1)
        elif self.reduction == "norm_of_diff":
            tensor = (tensor1 - tensor2).norm(dim=-1)

        self.values += tensor.to(torch.float64).sum()
        self.total += len(tensor)

    def compute(self) -> torch.Tensor:
        return self.values / self.total


class SlotMaskOccupancy(torchmetrics.Metric):
    """Metric that computes the fraction of used slots by evaluating slot mask occupancy."""

    def __init__(self, threshold: Union[int, float] = 0.01):
        super().__init__()
        self.min_elems_occupied = None
        self.min_frac_occupied = None
        if isinstance(threshold, int):
            self.min_elems_occupied = threshold
        elif isinstance(threshold, float):
            self.min_frac_occupied = threshold
        else:
            raise ValueError("Invalid type for `threshold`: must be int or float.")

        self.add_state(
            "values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, masks: torch.Tensor):
        masks = torch.atleast_3d(masks).flatten(2, -1)  # Assume shape b x slots x [dims]
        bs, num_slots, num_elems = masks.shape

        masks_hard = torch.floor(masks / masks.max(dim=1, keepdim=True).values)
        num_elems_per_slot = masks_hard.sum(-1)
        if self.min_elems_occupied is not None:
            occupied_slots = num_elems_per_slot >= self.min_elems_occupied
        elif self.min_frac_occupied is not None:
            occupied_slots = (num_elems_per_slot / num_elems) >= self.min_frac_occupied
        num_occupied_slots = occupied_slots.sum(-1)

        self.values += (num_occupied_slots / num_slots).sum()
        self.total += bs

    def compute(self) -> torch.Tensor:
        return self.values / self.total


class CategoricalEntropy(torchmetrics.Metric):
    """Metric that computes entropy of a categorical distribution along a dimension."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.add_state(
            "values", default=torch.tensor(0.0, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, tensor: torch.Tensor):
        tensor = tensor.clamp(min=0, max=1)
        entropy = -torch.nansum(tensor * torch.log(tensor), axis=self.dim)

        self.values += entropy.sum()
        self.total += entropy.numel()

    def compute(self) -> torch.Tensor:
        return self.values / self.total
