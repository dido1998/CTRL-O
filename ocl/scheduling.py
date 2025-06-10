"""Scheduling of learning rate and hyperparameters."""
import abc
import functools
import math
import warnings
from typing import Any, Dict, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, _LRScheduler


def _warmup_fn(step: int, warmup_steps: int) -> float:
    """Learning rate warmup.

    Maps the step to a factor for rescaling the learning rate.
    """
    if warmup_steps:
        return min(1.0, step / warmup_steps)
    else:
        return 1.0


def _exp_decay_after_warmup_fn(
    step: int, decay_rate: float, decay_steps: int, warmup_steps: int
) -> float:
    """Decay function for exponential decay with learning rate warmup.

    Maps the step to a factor for rescaling the learning rate.
    """
    factor = _warmup_fn(step, warmup_steps)
    if step < warmup_steps:
        return factor
    else:
        return factor * (decay_rate ** ((step - warmup_steps) / decay_steps))


def _exp_decay_with_warmup_fn(
    step: int, decay_rate: float, decay_steps: int, warmup_steps: int
) -> float:
    """Decay function for exponential decay with learning rate warmup.

    Maps the step to a factor for rescaling the learning rate.
    """
    factor = _warmup_fn(step, warmup_steps)
    return factor * (decay_rate ** (step / decay_steps))


class _CosineAnnealingWithWarmup(_LRScheduler):
    """Cosine annealing with warmup."""

    def __init__(
        self,
        optimizer,
        T_max: int,
        warmup_steps: int = 0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
        error_on_exceeding_steps: bool = True,
        verbose: bool = False,
    ):
        self.T_max = T_max
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        self.error_on_exceeding_steps = error_on_exceeding_steps
        super().__init__(optimizer, last_epoch, verbose)

    def _linear_lr_warmup(self, base_lr, step_num):
        return base_lr * ((step_num + 0.5) / self.warmup_steps)

    def _cosine_annealing(self, base_lr, step_num):
        fraction_of_steps = (step_num - self.warmup_steps) / (self.T_max - self.warmup_steps - 1)
        return self.eta_min + 1 / 2 * (base_lr - self.eta_min) * (
            1 + math.cos(math.pi * fraction_of_steps)
        )

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                stacklevel=2,
            )
        step_num = self.last_epoch

        if step_num < self.warmup_steps:
            # Warmup.
            return [self._linear_lr_warmup(base_lr, step_num) for base_lr in self.base_lrs]
        elif step_num < self.T_max:
            # Cosine annealing.
            return [self._cosine_annealing(base_lr, step_num) for base_lr in self.base_lrs]
        else:
            if self.error_on_exceeding_steps:
                raise ValueError(
                    "Tried to step {} times. The specified number of total steps is {}".format(
                        step_num + 1, self.T_max
                    )
                )
            else:
                return [self.eta_min for _ in self.base_lrs]


class HPScheduler(torch.nn.Module, metaclass=abc.ABCMeta):
    """Base class for scheduling of scalar hyperparameters based on the number of training steps.

    A separate callback [ocl.callbacks.UpdateHyperparameterScheduling][] calls
    `update_global_step` to update the state of the hyperparameter according
    to the scheduling.

    This class can be used in computations similar to a regular float if operations
    are applied from the left otherwise it needs to be converted using
    `float(instance)` which will return the currently scheduled value of the
    hyperparameter.


    """

    def __init__(self):
        super().__init__()
        self.last_global_step: Optional[int] = None

    def update_global_step(self, global_step: int):
        """Update global step used in `compute_scheduled_value`.

        This should be called by the
        [ocl.callbacks.UpdateHyperparameterScheduling][] callback.

        Args:
            global_step: The current global step.
        """
        self.last_global_step = global_step

    @abc.abstractmethod
    def compute_scheduled_value(self) -> float:
        """Return current value of hyperparameter based on global step.

        Returns:
            The scheduled hyperparameter value.
        """
        pass

    def __float__(self):
        if self.last_global_step is None:
            raise RuntimeError(
                "HPScheduler was not provided with last_global_step. "
                "Make sure UpdateHyperparameterScheduling callback is called."
            )
        return self.compute_scheduled_value()

    def __add__(self, other):
        return float(self) + other

    def __sub__(self, other):
        return float(self) - other

    def __mul__(self, other):
        return float(self) * other

    def __div__(self, other):
        return float(self) / other


class LinearHPScheduler(HPScheduler):
    """Linearly increase value of a hyperparameter."""

    def __init__(
        self, end_value: float, end_step: int, start_value: float = 0.0, start_step: int = 0
    ):
        """Initialize LinearHPScheduler.

        Args:
            end_value: Value after scheduling.
            end_step: `global_step` at which `end_value` should be reached.
            start_value: Value to be used prior to `start_step`
            start_step: Value at which linear scheduling schould start.
        """
        super().__init__()
        if start_step > end_step:
            raise ValueError("`start_step` needs to be smaller equal to `end_step`.")

        self.start_value = start_value
        self.end_value = end_value
        self.start_step = start_step
        self.end_step = end_step

    def compute_scheduled_value(self) -> float:
        step: int = self.last_global_step
        if step < self.start_step:
            return self.start_value
        elif step > self.end_step:
            return self.end_value
        else:
            t = step - self.start_step
            T = self.end_step - self.start_step
            return self.start_value + t * (self.end_value - self.start_value) / T


class StepHPScheduler(HPScheduler):
    """Schedule hyperparameter using discrete step."""

    def __init__(self, end_value: float, switch_step: int, start_value: float = 0.0):
        """Initialize StepHPScheduler.

        Args:
            end_value: Value after `switch_step`.
            switch_step: `global_step` at which to switch from `start_value` to `end_value`
            start_value: Value to be used prior to `switch_step`
        """
        super().__init__()
        self.start_value = start_value
        self.end_value = end_value
        self.switch_step = switch_step

    def compute_scheduled_value(self) -> float:
        if self.last_global_step < self.switch_step:
            return self.start_value
        else:
            return self.end_value


class CosineAnnealingHPScheduler(HPScheduler):
    """Cosine annealing of hyperparameter."""

    def __init__(
        self,
        start_value: float,
        end_value: float,
        start_step: int,
        end_step: int,
        warmup_steps: int = 0,
        warmup_value: Optional[float] = None,
        warmup_type: str = "linear",
    ):
        """Initialize CosineAnnealingHPScheduler.

        Args:
            end_value: Value after scheduling.
            end_step: `global_step` at which `end_value` should be reached.
            start_value: Value to be used prior to `start_step`
            start_step: Value at which cosine scheduling schould start.
            warmup_steps: Number of warmup steps after scheduling started.
            warmup_value: Value to which to warmup to.
            warmup_type: Type of warmup. Either "linear", or "cosine".
        """
        super().__init__()
        assert start_step <= end_step
        self.start_value = start_value
        self.end_value = end_value
        self.start_step = start_step
        self.end_step = end_step
        self.warmup_steps = warmup_steps
        self.warmup_value = warmup_value if warmup_value is not None else start_value
        self.warmup_type = warmup_type
        if warmup_type not in ("linear", "cosine"):
            raise ValueError(f"Unknown warmup_type {warmup_type}")

    def compute_scheduled_value(self) -> float:
        step: int = self.last_global_step
        steps_since_start = step - self.start_step

        if step <= self.start_step:
            value = self.start_value
        elif step <= self.start_step + self.warmup_steps:
            progress = min(1.0, steps_since_start / self.warmup_steps)
            if self.warmup_type == "linear":
                value = (1 - progress) * self.start_value + progress * self.warmup_value
            elif self.warmup_type == "cosine":
                a = 0.5 * (self.start_value - self.warmup_value)
                b = 0.5 * (self.start_value + self.warmup_value)
                value = a * math.cos(math.pi * progress) + b
        elif step >= self.end_step:
            value = self.end_value
        else:
            a = 0.5 * (self.warmup_value - self.end_value)
            b = 0.5 * (self.warmup_value + self.end_value)
            progress = steps_since_start / (self.end_step - self.start_step - self.warmup_steps)
            value = a * math.cos(math.pi * progress) + b

        return value


def exponential_decay_with_optional_warmup(
    optimizer: Optimizer, decay_rate: float = 1.0, decay_steps: int = 10000, warmup_steps: int = 0
) -> Dict[str, Any]:
    """Return pytorch lighting optimizer configuration for exponential decay with optional warmup.

    Exponential decay is applied at each optimization step.  Exponential decay starts
    **while** warmup is still taking place.  This is in line with the typical scheduling
    used to train Transformer models.

    Args:
        optimizer: Pytorch lighting optimizer of which the learning rate should be scheduled.
        decay_rate: Decay rate of exponential decay.
        decay_steps: Number of optimization steps after which learning rate should be decayed
            by decay factor.
        warmup_steps: Number of warmup steps.

    Returns:
        Dict with structure compatible with ptl.  See
            [pytorch lightning documentation](
                https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers)

    """
    decay_fn = functools.partial(
        _exp_decay_with_warmup_fn,
        decay_rate=decay_rate,
        decay_steps=decay_steps,
        warmup_steps=warmup_steps,
    )

    return {"lr_scheduler": {"scheduler": LambdaLR(optimizer, decay_fn), "interval": "step"}}


def exponential_decay_after_optional_warmup(
    optimizer: Optimizer, decay_rate: float = 1.0, decay_steps: int = 10000, warmup_steps: int = 0
) -> Dict[str, Any]:
    """Return pytorch lighting optimizer configuration for exponential decay with optional warmup.

    Exponential decay is applied at each optimization step.  Exponential decay starts
    **after** warmup is took place.

    Args:
        optimizer: Pytorch lighting optimizer of which the learning rate should be scheduled.
        decay_rate: Decay rate of exponential decay.
        decay_steps: Number of optimization steps after which learning rate should be decayed
            by decay factor.
        warmup_steps: Number of warmup steps.

    Returns:
        Dict with structure compatible with ptl.  See
            [pytorch lightning documentation](
                https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers)
    """
    decay_fn = functools.partial(
        _exp_decay_after_warmup_fn,
        decay_rate=decay_rate,
        decay_steps=decay_steps,
        warmup_steps=warmup_steps,
    )

    return {"lr_scheduler": {"scheduler": LambdaLR(optimizer, decay_fn), "interval": "step"}}


def plateau_decay(
    optimizer: Optimizer,
    decay_rate: float = 1.0,
    patience: int = 10,
    monitor_metric: str = "val/lotal_loss",
    mode: str = "min",
) -> Dict[str, Any]:
    """Return pytorch lighting optimizer configuration for plato decay.

    Args:
        optimizer: Pytorch lighting optimizer of which the learning rate should be scheduled.
        decay_rate: Factor by which learning rate should be decayed when plateau is reached.
        patience: Number of epochs to wait for improvement.
        mode: `min` or `max`.

    Returns:
        Dict with structure compatible with ptl.  See
            [pytorch lightning documentation](
                https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers)
    """
    plateau_scheduler = ReduceLROnPlateau(
        optimizer=optimizer, mode=mode, factor=decay_rate, patience=patience
    )
    return {
        "lr_scheduler": {
            "scheduler": plateau_scheduler,
            "interval": "epoch",
            "monitor": monitor_metric,
        }
    }


def cosine_annealing_with_optional_warmup(
    optimizer: Optimizer,
    T_max: int = 100000,
    eta_min: float = 0.0,
    warmup_steps: int = 0,
    error_on_exceeding_steps: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Return pytorch lighting optimizer configuration for cosine annealing with warmup.

    Args:
        optimizer: Pytorch lighting optimizer of which the learning rate should be scheduled.
        T_max: The length of the scheduling in steps.
        eta_min: Minimal fraction of initial learning rate that should be reached when
            scheduling cycle is complete.
        warmup_steps: Number of warmup steps.
        error_on_exceeding_steps: Raise error if more than `T_max` steps are trained.

    Returns:
        Dict with structure compatible with ptl.  See
            [pytorch lightning documentation](
                https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers)
    """
    return {
        "lr_scheduler": {
            "scheduler": _CosineAnnealingWithWarmup(
                optimizer,
                T_max,
                eta_min=eta_min,
                warmup_steps=warmup_steps,
                error_on_exceeding_steps=error_on_exceeding_steps,
            ),
            "interval": "step",
        }
    }
