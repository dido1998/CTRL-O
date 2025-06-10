"""Convenience functions that allow defining optimization via config."""
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torch.optim import Optimizer


class OptimizationWrapper:
    """Optimize (a subset of) the parameters using a optimizer and a LR scheduler."""

    def __init__(
        self,
        optimizer: Optimizer,
        lr_scheduler: Optional[Callable[[Optimizer], Dict[str, Any]]] = None,
        parameter_groups: Optional[
            Union[Callable[[], List[Dict[str, Any]]], List[Dict[str, Any]]]
        ] = None,
    ):
        """Initialize OptimizationWrapper.

        Args:
            optimizer: The oprimizer that should be used to optimize the parameters.
            lr_scheduler: The LR scheduling callable that should be used.  This should
                be a callable that returns a dict for updating the optimizer output in
                pytorch_lightning. See [ocl.scheduling.exponential_decay_with_optional_warmup][]
                for an example of such a callable.
            parameter_groups: Define parameter groups which have different optimizer parameters.
                Each element of the list should at least one of two keys `params` (for defining
                parameters based on their path in the model) or `predicate` (for defining parameters
                using a predicate function which returns true if the parameter should be included).
                For an example on how to use this parameter_groups, see
                `configs/experiment/examples/parameter_groups.yaml`.
        """
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.parameter_group_specs = (
            parameter_groups() if callable(parameter_groups) else parameter_groups
        )
        if self.parameter_group_specs:
            for idx, param_group_spec in enumerate(self.parameter_group_specs):
                if "params" not in param_group_spec:
                    raise ValueError(f'Parameter group {idx + 1} does not contain key "params"')
                param_spec = param_group_spec["params"]
                if isinstance(param_spec, str):
                    param_group_spec["params"] = [param_spec]
                elif isinstance(param_spec, Iterable):
                    param_group_spec["params"] = list(param_spec)
                else:
                    raise ValueError(
                        f'"params" for parameter group {idx + 1} is not of type str or iterable'
                    )

                if "predicate" in param_group_spec:
                    if not callable(param_group_spec["predicate"]):
                        raise ValueError(
                            f'"predicate" for parameter group {idx + 1} is not a callable'
                        )

    def _get_parameter_groups(self, model):
        """Build parameter groups from specification."""
        if not self.parameter_group_specs:
            return model.parameters()
        parameter_groups = []
        for param_group_spec in self.parameter_group_specs:
            param_spec = param_group_spec["params"]
            # Default predicate includes all parameters
            predicate = param_group_spec.get("predicate", lambda name, param: True)

            parameters = []
            for parameter_path in param_spec:
                root = model
                for child in parameter_path.split("."):
                    root = getattr(root, child)
                if root is None:
                    continue
                named_params = (
                    root.named_parameters() if hasattr(root, "named_parameters") else [(child, root)]
                )
                parameters.extend(param for name, param in named_params if predicate(name, param))

            param_group = {
                k: v for k, v in param_group_spec.items() if k not in ("params", "predicate")
            }
            param_group["params"] = parameters
            parameter_groups.append(param_group)

        return parameter_groups

    def __call__(self, model: torch.nn.Module):
        """Called in configure optimizers."""
        params_or_param_groups = self._get_parameter_groups(model)
        optimizer = self.optimizer(params_or_param_groups)
        output = {"optimizer": optimizer}
        if self.lr_scheduler:
            output.update(self.lr_scheduler(optimizer))
        return output


class PredicateNoWeightDecay:
    """Predicate for parameter groups for parameters that should not receive weight decay."""

    @staticmethod
    def predicate(name: str, param: torch.Tensor) -> bool:
        keywords = {"bias", "pos_embed", "cls_token", "reg_token"}
        return any(keyword in name for keyword in keywords) or param.ndim == 1

    def __call__(self, name: str, param: torch.Tensor) -> bool:
        return self.predicate(name, param)


class PredicateWeightDecay:
    """Predicate for parameter groups for parameters that should receive weight decay."""

    def __call__(self, name: str, param: torch.Tensor) -> bool:
        return not PredicateNoWeightDecay.predicate(name, param)


class ParameterGroupCreator:
    """Convenience class allowing simpler weight decay and layerwise learning rate decay config."""

    def __init__(
        self,
        param_groups: Dict[str, Dict[str, Any]],
        verbose: bool = False,
    ):
        self.param_groups = param_groups
        self.verbose = verbose
        for name, group in self.param_groups.items():
            if not isinstance(group, dict):
                raise ValueError(f"param_group {name} needs to contain dictionaries")
            if "params" not in group:
                raise ValueError(f"Need to specify parameters for group {name }under key `params` ")
            if "layerwise_lr_decay" in group and (
                group["layerwise_lr_decay"] < 0 or group["layerwise_lr_decay"] > 1
            ):
                raise ValueError("`layerwise_lr_decay` for group {name} needs to be in [0, 1]")

    def _add_group(
        self, spec: List[Dict[str, Any]], params: Union[str, List[str]], args: Dict[str, Any]
    ):
        if "weight_decay" in args and args["weight_decay"] > 0:
            spec.append(dict(params=params, predicate=PredicateWeightDecay(), **args))
            spec.append(
                dict(
                    params=params,
                    predicate=PredicateNoWeightDecay(),
                    **{k: v if k != "weight_decay" else 0.0 for k, v in args.items()},
                )
            )
        else:
            spec.append(dict(params=params, **args))

    def __call__(self) -> List[Dict[str, Any]]:
        spec = []
        for name, group in self.param_groups.items():
            params = [group["params"]] if isinstance(group["params"], str) else group["params"]
            args = {k: v for k, v in group.items() if k != "params"}
            if "layerwise_lr_decay" in args:
                if "lr" not in group:
                    raise ValueError(f"No learning rate `lr` specified for lr decay group {name}.")
                lr = group["lr"]
                for param in reversed(params):
                    self._add_group(
                        spec, param, {k: v if k != "lr" else lr for k, v in args.items()}
                    )
                    lr *= args["layerwise_lr_decay"]
            else:
                self._add_group(spec, params, args)

        if self.verbose:
            rank_zero_info(spec)

        return spec
