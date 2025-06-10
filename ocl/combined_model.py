"""Implementation of combined model."""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union

import pytorch_lightning as pl
import torch
from huggingface_hub import hf_hub_download
from torch import nn

from ocl.optimization import OptimizationWrapper
from ocl.utils.routing import Combined, read_path
from ocl.utils.trees import walk_tree_with_paths
from ocl.visualization_types import Visualization
from ocl.visualizations import VisualizationMethod

if TYPE_CHECKING:
    import torchmetrics


class CombinedModel(pl.LightningModule):
    """Core pytorch lightning model used for training, loss compuation and visualization."""

    def __init__(
        self,
        models: Union[Dict[str, Any], nn.Module],
        optimizers: Dict[str, Union[OptimizationWrapper, Callable]],
        losses: Dict[str, Any],
        visualizations: Dict[str, VisualizationMethod],
        training_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        evaluation_metrics: Optional[Dict[str, torchmetrics.Metric]] = None,
        vis_log_frequency: int = 100,
        trainable_models: Optional[list[str]] = None,
        modules_to_load: Optional[Dict[str, str]] = None,
        load_checkpoint_partial: Optional[str] = None,
    ):
        """Initialize combined model.

        Args:
            models: The model to run the forward pass.  If a dict is provieded the
                modules of the dict are wrapped with [ocl.utils.routing.Combined][].
            optimizers: Dictionary of partial optimizer objects or OptimizationWrappers.
            losses: Dictionary of losses. The key is used to track the loss value during
                logging, the sum of all losses is used to optimize the model.
            visualizations: Visualizations for visualizing and monitoring training progress.
            training_metrics: Metrics to evaluate during training.
            evaluation_metrics: Metrics to evaluate during validation and testing.
            vis_log_frequency: Frequency in optimization steps when to run visualizations.
        """
        super().__init__()
        if isinstance(models, Dict):
            models = Combined(**models)
        self.models = models
        self.optimizers = optimizers
        self.losses = torch.nn.ModuleDict(losses)
        self.visualizations = visualizations
        self.vis_log_frequency = vis_log_frequency
        self.return_outputs_on_validation = False

        if training_metrics is None:
            training_metrics = {}
        self.training_metrics = torch.nn.ModuleDict(training_metrics)

        if evaluation_metrics is None:
            evaluation_metrics = {}
        self.evaluation_metrics = torch.nn.ModuleDict(evaluation_metrics)
        self.trainable_models = trainable_models

        if self.trainable_models is not None:
            assert all(
                m in self.models for m in self.trainable_models
            ), "All trainable model names must be in models."
            self.freeze_parameters()

        if load_checkpoint_partial is not None:
            # Load weights from checkpoint for specified modules.
            self.load_weights_from_checkpoint(load_checkpoint_partial, modules_to_load)

    def freeze_parameters(self):
        for model_name, model in self.models.items():
            if model_name not in self.trainable_models:
                for param in model.parameters():
                    param.requires_grad = False

    def _build_optimizer(self, optimizer: Union[OptimizationWrapper, Callable]):
        if isinstance(optimizer, OptimizationWrapper):
            return optimizer(self)
        # Support using a partial of a standard pytorch optimizer.
        return optimizer(self.parameters())

    def configure_optimizers(self):
        return [self._build_optimizer(self.optimizers[name]) for name in sorted(self.optimizers)]

    def forward(self, input_data: dict):
        # Maybe we should use something like a read only dict to prevent existing keys from being
        # overwritten.
        data: Dict[str, Any]
        data = {
            "input": input_data,
            # TODO(hornmax): Figure out if there is a better way to acces multi-gpu operations.
            "model": self,
        }
        return self.models(inputs=data)

    def _compute_losses(self, inputs, phase="train"):
        quantities_to_log = {}
        # We write additional loss outputs directly into the inputs dict, and thus do not need to
        # return them.
        outputs = inputs["losses"] = {}
        for name, loss in self.losses.items():
            out = loss(inputs=inputs)
            if isinstance(out, tuple):
                # Additional outputs that should be logged for later access.
                # Some visualizations require having access to loss quantities, thus we need to save
                # them for later here.
                out, additional_outputs = out
                outputs[name] = additional_outputs
            quantities_to_log[f"{phase}/{name}"] = out

        losses = []
        for loss in quantities_to_log.values():
            losses.append(loss)

        total_loss = torch.stack(losses).sum()

        # Log total loss only if there is more than one task
        if len(losses) > 1:
            quantities_to_log[f"{phase}/loss_total"] = total_loss

        return total_loss, quantities_to_log

    def predict_step(self, batch, batch_idx):
        outputs = self(batch)
        # Remove things not needed in prediction output.
        del outputs["model"]
        return outputs

    def training_step(self, batch, batch_idx):
        batch_size = batch["batch_size"]
        outputs = self(batch)
        total_loss, quantities_to_log = self._compute_losses(outputs)

        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            quantities_to_log.update(self._compute_metrics(outputs, self.training_metrics))
        self.log_dict(
            quantities_to_log, on_step=True, on_epoch=False, batch_size=batch_size, sync_dist=True
        )
        if self.trainer.global_step % self.vis_log_frequency == 0:
            self._log_visualizations(outputs)

        return total_loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch["batch_size"]

        outputs = self(batch)

        total_loss, quantities_to_log = self._compute_losses(outputs, phase="val")

        quantities_to_log.update(
            self._compute_metrics(outputs, self.evaluation_metrics, phase="val")
        )
        self.log_dict(
            quantities_to_log,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
            sync_dist=True,
        )

        if batch_idx == 0:
            self._log_visualizations(outputs, phase="val")

        if self.return_outputs_on_validation:
            return outputs  # Used for saving model outputs during eval
        else:
            return None

    def _compute_metrics(self, outputs, metric_fns, phase="train"):
        metrics = {}
        if len(metric_fns) > 0:
            for metric_name, metric in metric_fns.items():
                if phase == "val":
                    # Call update instead of forward to avoid unnecessary metric compute on batch.
                    metric.update(**outputs)
                else:
                    metric(**outputs)
                metrics[f"{phase}/{metric_name}"] = metric

        return metrics

    def _log_visualizations(self, outputs, phase="train"):
        if self.logger is None:
            return
        logger_experiment = self.logger.experiment
        visualizations = {}
        for name, vis in self.visualizations.items():
            visualizations[name] = vis(inputs=outputs)

        visualization_iterator = walk_tree_with_paths(
            visualizations, path=None, instance_check=lambda t: isinstance(t, Visualization)
        )
        for path, vis in visualization_iterator:
            try:
                str_path = ".".join(path)
                vis.add_to_experiment(
                    experiment=logger_experiment,
                    tag=f"{phase}/{str_path}",
                    global_step=self.trainer.global_step,
                )
            except AttributeError:
                # The logger does not support the right data format.
                pass

    def on_load_checkpoint(self, checkpoint):
        # Reset timer during loading of the checkpoint, as timer is used to track time from the start
        # of the current run.
        if "callbacks" in checkpoint and "Timer" in checkpoint["callbacks"]:
            checkpoint["callbacks"]["Timer"]["time_elapsed"] = {
                "train": 0.0,
                "sanity_check": 0.0,
                "validate": 0.0,
                "test": 0.0,
                "predict": 0.0,
            }

    def load_weights_from_checkpoint(
        self, checkpoint_path: str, module_mapping: Optional[Dict[str, str]] = None
    ):
        """Load weights from a checkpoint into the specified modules."""

        if not os.path.exists(checkpoint_path):
            print(f"{checkpoint_path} does not exists.")
            checkpoint_path = hf_hub_download(
                repo_id=checkpoint_path,
                filename="model.ckpt",
                use_auth_token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
            )
            print(f"Use weights from checkpoint {os.path.abspath(checkpoint_path)}")

        checkpoint_path = Path(checkpoint_path)
        assert checkpoint_path.exists(), f"Checkpoint {checkpoint_path} does not exist."
        checkpoint = torch.load(checkpoint_path)
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        if module_mapping is None:
            module_mapping = {
                key.split(".")[0]: key.split(".")[0]
                for key in checkpoint
                if hasattr(self.models, key.split(".")[0])
            }

        for dest_module, source_module in module_mapping.items():
            try:
                module = read_path(self.models, dest_module)
            except ValueError:
                raise ValueError(f"Module {dest_module} could not be retrieved from model") from None

            state_dict = {}
            for key, weights in checkpoint.items():
                if key.startswith(source_module):
                    if key != source_module:
                        key = key[len(source_module + ".") :]  # Remove prefix
                    state_dict[key] = weights
            if len(state_dict) == 0:
                raise ValueError(
                    f"No weights for module {source_module} found in checkpoint {checkpoint_path}."
                )

            module.load_state_dict(state_dict)
