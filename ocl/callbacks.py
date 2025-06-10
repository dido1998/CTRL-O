import functools
import itertools
import os
from collections import defaultdict, deque
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
from torch import nn

from ocl.scheduling import HPScheduler
from ocl.utils.trees import get_tree_element, walk_tree_with_paths


class FreezeParameters(Callback):
    """Freeze parameters of model prior to training."""

    def __init__(self, parameter_groups: List[Dict[str, Any]]):
        """Initialize FreezeParameters callback.

        Args:
            parameter_groups: Parameter groups that should be frozen.
                Uses same syntax as [ocl.optimization.OptimizationWrapper][]
        """
        super().__init__()
        self.parameter_group_specs = parameter_groups
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
                    raise ValueError(f'"predicate" for parameter group {idx + 1} is not a callable')

    def _get_parameters_to_freeze(self, model):
        """Build parameter groups from specification."""
        parameters_to_freeze = []
        for param_group_spec in self.parameter_group_specs:
            for current_params in param_group_spec["params"]:
                param_path = current_params.split(".")
                # Default predicate includes all parameters
                predicate = param_group_spec.get("predicate", lambda name, param: True)
                param = get_tree_element(model, param_path)
                if isinstance(param, nn.Module):
                    parameters_to_freeze.extend(
                        param for name, param in param.named_parameters() if predicate(name, param)
                    )
                elif isinstance(param, nn.Parameter):
                    parameters_to_freeze.append(param)
                else:
                    raise ValueError(
                        "Object at path {'.'.join(param_path)} is neither nn.Module nor nn.Parameter"
                    )
        return parameters_to_freeze

    def on_fit_start(self, trainer, model: nn.Module):
        parameters_to_freeze = self._get_parameters_to_freeze(model)
        for param in parameters_to_freeze:
            param.requires_grad_(False)


class RestoreParameterSubset(Callback):
    """Restore a subset of parameters using a checkpoint form a different model."""

    def __init__(
        self,
        checkpoint_file: str,
        target_path: str,
        source_path: Optional[str] = None,
        filter_fn: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None,
    ):
        """Initialize RestoreParameterSubset callback.

        Args:
            checkpoint_file: File from which the model weights should be loaded.
            target_path: The path in the model where the model weights should be
                restored.  This should follow a dot separated syntax, such a `encoder.layer1`.
            source_path: The path in the checkpoint_file that should be used to restore weights.
                If none provided assumes to be the same as `target_path`.
            filter_fn: Optional function used for filtering/transforming the state dict before
                loading it into the model.
        """
        self.checkpoint_file = checkpoint_file
        self.target_path = target_path
        self.source_path = source_path if source_path else self.target_path
        self.filter_fn = filter_fn

    def on_train_start(self, trainer, model: nn.Module):
        if model.global_step != 0:
            # Don't restore when we are resuming training.
            rank_zero_warn("Not restoring parameter subset as training is being resumed")
            return
        # Get parameters from state dict
        state_dict = torch.load(self.checkpoint_file, map_location=model.device)["state_dict"]
        # Add offset of 1 to remove potential dot.
        offset_keys = len(self.source_path) + 1
        state_dict = {
            key[offset_keys:]: value
            for key, value in state_dict.items()
            if key.startswith(self.source_path)
        }

        if self.filter_fn:
            state_dict = self.filter_fn(state_dict)

        # Get module from model
        model_component: nn.Module = get_tree_element(model, self.target_path.split("."))
        result = model_component.load_state_dict(state_dict)
        if len(result.missing_keys):
            rank_zero_warn(
                f"Mismatch between state dict and model. Missing keys: {result.missing_keys}"
            )
        if len(result.unexpected_keys):
            rank_zero_warn(
                f"Mismatch between state dict and model. Unexpected keys: {result.missing_keys}"
            )
        rank_zero_info(
            f"Restored subset of model parameters at {self.target_path} from {self.checkpoint_file}"
        )


class UpdateHyperparameterScheduling(Callback):
    """Callback to update hyperparameter schedulers found `ocl.scheduling`."""

    def __init__(self):
        self._hyperparameter_schedulers: List[HPScheduler] = []

    def on_fit_start(self, trainer, pl_module):
        schedulers = itertools.chain(
            walk_tree_with_paths(pl_module, instance_check=lambda t: isinstance(t, HPScheduler)),
            # Check if a callback explicitly offers hyperparameters for scheduling
            *[
                walk_tree_with_paths(
                    cb.hyperparameters, instance_check=lambda t: isinstance(t, HPScheduler)
                )
                for cb in trainer.callbacks
                if hasattr(cb, "hyperparameters")
            ],
        )

        self._hyperparameter_schedulers = list(map(lambda a: a[1], schedulers))
        if len(self._hyperparameter_schedulers) == 0:
            rank_zero_warn(
                "UpdateHyperparameterScheduling: "
                "No schedulable hyperparameters where found in model."
            )
        # Set global step to 0 for pretraining evaluation routine.
        self._update_schedulers(0)

    def _update_schedulers(self, step):
        for hparam in self._hyperparameter_schedulers:
            hparam.update_global_step(step)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        del trainer, batch, batch_idx
        global_step = pl_module.global_step
        self._update_schedulers(global_step)


class SetEpochEnvironmentVariable(Callback):
    """Sets environment variable `EPOCH` which is used by [ocl.transforms.SampleSlices][]."""

    def on_train_epoch_start(self, trainer, pl_module):
        os.environ["EPOCH"] = str(pl_module.current_epoch)


class EMAUpdater(Callback):
    """Callback performing EMA updates from one module to another module."""

    def __init__(
        self,
        source_to_target_modules: Dict[str, str],
        momentum: Union[float, HPScheduler],
        requires_grad: bool = False,
        no_grad: bool = True,
    ):
        super().__init__()
        self.source_to_target_modules = source_to_target_modules
        self.momentum = momentum
        self.requires_grad = requires_grad
        self.no_grad = no_grad
        self._target_modules: List[torch.nn.Module] = []
        self._source_modules: List[torch.nn.Module] = []

    @property
    def hyperparameters(self) -> List[Union[float, HPScheduler]]:
        return [self.momentum]

    @staticmethod
    def _init_weights(target_module: torch.nn.Module, source_module: torch.nn.Module):
        source_keys = set(k for k, _ in source_module.named_parameters())
        target_keys = set(k for k, _ in target_module.named_parameters())
        if source_keys != target_keys:
            only_source = sorted(list(source_keys.difference(target_keys)))
            only_target = sorted(list(target_keys.difference(source_keys)))
            raise ValueError(
                "Mismatch in parameters: "
                f"Params only in source:\n {', '.join(only_source)}.\nParams only in target "
                f"{', '.join(only_target)}.\nHave you set the correct source module?"
            )

        with torch.no_grad():
            for param_src, param_dest in zip(source_module.parameters(), target_module.parameters()):
                param_dest.data.copy_(param_src.detach().data)

    @staticmethod
    def _update(target_module: torch.nn.Module, source_module: torch.nn.Module, momentum: float):
        with torch.no_grad():
            for param_src, param_dest in zip(source_module.parameters(), target_module.parameters()):
                param_dest.data.mul_(momentum).add_(param_src.detach().data, alpha=1.0 - momentum)

    def on_train_start(self, trainer, pl_module):
        del trainer

        for source_path, target_path in self.source_to_target_modules.items():
            self._source_modules.append(get_tree_element(pl_module, source_path.split(".")))
            self._target_modules.append(get_tree_element(pl_module, target_path.split(".")))

        if pl_module.global_step == 0:  # Don't initialize on resumed training
            for (source_path, target_path), target_module, source_module in zip(
                self.source_to_target_modules.items(), self._target_modules, self._source_modules
            ):
                try:
                    self._init_weights(target_module, source_module)
                except ValueError as e:
                    raise ValueError(
                        f"Error while initializing module at path {target_path} from module "
                        f"at path {source_path}"
                    ) from e
                rank_zero_info(
                    f"Initialized module at path {target_path} with weights from module at path "
                    f"{source_path}"
                )

        for target_module in self._target_modules:
            target_module.requires_grad_(self.requires_grad)
            if self.no_grad:
                target_module.forward = torch.no_grad(target_module.forward)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        del trainer, pl_module, outputs, batch, batch_idx
        # Convert to float in case momentum is a scheduled parameter
        m = float(self.momentum)

        for target_module, source_module in zip(self._target_modules, self._source_modules):
            self._update(target_module, source_module, m)


class LogFeatureStatistics(Callback):
    """Callback that tracks weight and activation magnitudes.

    This follows the approach described in Karras et al, 2023: "Analyzing and Improving the Training
    Dynamics of Diffusion Models".
    """

    def __init__(
        self,
        track_param_norms: bool = True,
        track_activation_norms: bool = True,
        track_gradient_norms: bool = True,
        log_every: int = 64,
        num_activation_samples: int = 4096,
    ):
        super().__init__()
        self.track_param_norms = track_param_norms
        self.track_activation_norms = track_activation_norms
        self.track_gradient_norms = track_gradient_norms
        self.log_every = log_every
        self.activated_modules = set()
        self.activation_norms = defaultdict(functools.partial(deque, maxlen=num_activation_samples))

    def on_train_start(self, trainer, pl_module):
        if not self.track_activation_norms:
            return

        def hook_fn(module, input, output, name):
            if module.training and isinstance(output, torch.Tensor):
                # Modules can be executed multiple times. Track each activation with a unique name.
                prefix = name
                count = 2
                while name in self.activated_modules:
                    name = f"{prefix}_{count}"
                    count += 1
                self.activated_modules.add(name)  # Track modules activated for this batch.
                norms = self.get_activation_norms(output)
                self.activation_norms[name].extend(norms.cpu())

        for name, module in pl_module.named_modules():
            if self.should_track_module(name, module):
                module.register_forward_hook(functools.partial(hook_fn, name=name))

    def on_after_backward(self, trainer, pl_module):
        self.activated_modules.clear()

        if pl_module.global_step % self.log_every != 0:
            return

        to_log = {}
        if self.track_param_norms:
            norms = {
                f"params/norm/{name}": norm for name, norm in self.get_param_norms(pl_module).items()
            }
            to_log.update(norms)

        if self.track_gradient_norms:
            norms = {
                f"gradients/norm/{name}": norm
                for name, norm in self.get_grad_norms(pl_module).items()
            }
            to_log.update(norms)

        if self.track_activation_norms:
            norms = {
                name: torch.stack(tuple(norms)).mean()
                for name, norms in self.activation_norms.items()
            }
            norms = {f"activations/norm/{name}": norm for name, norm in norms.items()}
            to_log.update(norms)

        pl_module.log_dict(to_log, on_step=True)

    @torch.no_grad()
    def get_activation_norms(self, activation: torch.Tensor) -> torch.Tensor:
        activation = activation.detach()  # Assume first dimension is batch dimension
        if activation.ndim == 2:
            # Add a dummy positional dimension
            activation = activation.unsqueeze(1)
        elif activation.ndim == 4:
            # Assume a spatial map of [bs, dim, height, width] as from convolutions.
            activation = activation.flatten(-2, -1).transpose(-2, -1)
        elif activation.ndim != 3:
            raise ValueError(f"Unsupported number of dimensions: {activation.ndim}")

        # Take norm over positional dimensions
        norms = activation.norm(p=2, dim=1) / activation.shape[-1] ** 0.5
        return norms  # B x D

    @torch.no_grad()
    def get_param_norms(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        norms = {}
        for name, param in model.named_parameters():
            param = param.detach()
            if not self.should_track_param(name, param):
                continue
            if any(s in name for s in ("cls_token", "slots_mu", "slots_logsigma", "pos_embed")):
                param = param.squeeze(0)  # These have a dummy batch dimension, remove it.
            param = param.flatten(1, -1)  # Flatten into [output_features, input_features]
            per_feature_norm = param.norm(p=2, dim=-1) / param.shape[-1] ** 0.5
            norms[name] = per_feature_norm.mean()
        return norms

    @torch.no_grad()
    def get_grad_norms(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        norms = {}
        for name, param in model.named_parameters():
            grad = param.grad
            if not self.should_track_param(name, param) or grad is None:
                continue
            if any(s in name for s in ("cls_token", "slots_mu", "slots_logsigma", "pos_embed")):
                grad = grad.squeeze(0)  # These have a dummy batch dimension, remove it.
            grad = grad.flatten(1, -1)  # Flatten into [output_features, input_features]
            per_feature_norm = grad.norm(p=2, dim=-1) / grad.shape[-1] ** 0.5
            norms[name] = per_feature_norm.mean()
        return norms

    def should_track_module(self, name: str, module: torch.nn.Module) -> bool:
        has_params = any(
            self.should_track_param(name, param)
            for name, param in module.named_parameters(recurse=False)
        )
        return has_params

    def should_track_param(self, name: str, param: torch.nn.Parameter) -> bool:
        if "bias" in name or param.ndim == 1:
            return False
        return True


class VerifyOptimizerGroups(Callback):
    """Callback to check whether all parameters of a model are included in the optimizer."""

    def __init__(self, missing_okay: List[str] = None):
        self.missing_okay = missing_okay if missing_okay else []

    def on_fit_start(self, trainer, pl_module):
        optimizer_params = set()
        for optimizer in trainer.optimizers:
            for param_group in optimizer.param_groups:

                optimizer_params.update(set(param_group["params"]))
        for name, param in pl_module.named_parameters():
            if any(name.startswith(prefix) for prefix in self.missing_okay):
                continue
            if param not in optimizer_params:
                raise ValueError(f"Parameter {name} is not included in the optimizer weights.")
