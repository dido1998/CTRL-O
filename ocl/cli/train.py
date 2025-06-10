"""Train a slot attention type model."""
import dataclasses
from typing import Any, Callable, Dict, List, Optional

import hydra
import hydra_zen
import pytorch_lightning as pl
from omegaconf import SI

import ocl.cli._config  # noqa: F401
from ocl.cli import cli_utils
from ocl.combined_model import CombinedModel

RESULT_FINISHED = 0
RESULT_TIMEOUT = 1
RESULT_STATUS = None

# --8<-- [start:TrainingConfig]
# Convert dict of callbacks in experiment to list for use with PTL.
CALLBACK_INTERPOLATION = SI("${oc.dict.values:experiment.callbacks}")

TrainerConf = hydra_zen.builds(
    pl.Trainer, callbacks=CALLBACK_INTERPOLATION, zen_partial=False, populate_full_signature=True
)


@dataclasses.dataclass
class TrainingConfig:
    """Configuration of a training run.

    For losses, metrics and visualizations it can be of use to utilize the
    [routed][] module as these are simply provided with a dictionary of all
    model inputs and outputs.

    Attributes:
        dataset: The pytorch lightning datamodule that will be used for training
        models: Either a dictionary of [torch.nn.Module][]s which will be interpreted
            as a [Combined][ocl.utils.routing.Combined] model or a [torch.nn.Module][] itself
            that accepts a dictionary as input.
        optimizers: Dictionary of [functools.partial][] wrapped optimizers or
            [OptimizationWrapper][ocl.optimization.OptimizationWrapper] instances
        losses: Dict of callables that return scalar values which will be summed to
            compute a total loss.  Typically should contain [routed][] versions of callables.
        visualizations: Dictionary of [visualizations][ocl.visualizations].  Typically
            should contain [routed][] versions of visualizations.
        trainer: Pytorch lightning trainer
        training_vis_frequency: Number of optimization steps between generation and
            storage of visualizations.
        training_metrics: Dictionary of torchmetrics that should be used to log training progress.
            Typically should contain [routed][] versions of torchmetrics.
        evaluation_metrics: Dictionary of torchmetrics that should be used to log progress on
            evaluation splits of the data.  Typically should contain [routed][] versions of
            Torchmetrics.
        load_checkpoint: Path to checkpoint file that should be loaded prior to starting training.
        seed: Seed used to ensure reproducability.
        load_checkpoint_partial: Path to checkpoint file from which some of the modules
            should be loaded
        modules_to_load: Dictionary of module names and and corresponding checkpoint paths
        trainable_models: List of model names that should be trained.  If None, all models
        experiment: Dictionary with arbitrary additional information.  Useful when building
            configurations as it can be used as central point for a single parameter that might
            influence multiple model components.
    """

    dataset: Any
    models: Any  # When provided with dict wrap in `utils.Combined`, otherwise interpret as model.
    optimizers: Dict[str, Any]
    losses: Dict[str, Any]
    visualizations: Dict[str, Any] = dataclasses.field(default_factory=dict)
    trainer: TrainerConf = dataclasses.field(default_factory=lambda: TrainerConf())
    training_vis_frequency: Optional[int] = None
    training_metrics: Optional[Dict[str, Any]] = None
    evaluation_metrics: Optional[Dict[str, Any]] = None
    load_checkpoint: Optional[str] = None
    load_checkpoint_partial: Optional[str] = None
    modules_to_load: Optional[Dict[str, str]] = None
    trainable_models: Optional[List[str]] = None

    seed: Optional[int] = None
    experiment: Dict[str, Any] = dataclasses.field(default_factory=lambda: {"callbacks": {}})


# --8<-- [end:TrainingConfig]


hydra.core.config_store.ConfigStore.instance().store(
    name="training_config",
    node=TrainingConfig,
)


def _get_runtime_path(config: TrainingConfig) -> Optional[str]:
    try:
        hydra_conf = hydra.core.hydra_config.HydraConfig.get()
    except ValueError:
        # Hydra config not set
        return config.trainer.default_root_dir

    configured_path = hydra_conf.run.dir
    if configured_path == "." and config.trainer.default_root_dir is None:
        # This combination should lead to disabling the output directory
        runtime_path = None
    else:
        runtime_path = hydra_conf.runtime.output_dir

    return runtime_path


def build_and_register_datamodule_from_config(
    config: TrainingConfig,
    **datamodule_kwargs,
) -> pl.LightningDataModule:
    datamodule = hydra_zen.instantiate(config.dataset, _convert_="all", **datamodule_kwargs)
    return datamodule


def build_model_from_config(
    config: TrainingConfig,
    checkpoint_path: Optional[str] = None,
    checkpoint_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> pl.LightningModule:
    models = hydra_zen.instantiate(config.models, _convert_="all")
    optimizers = hydra_zen.instantiate(config.optimizers, _convert_="all")
    losses = hydra_zen.instantiate(config.losses, _convert_="all")
    visualizations = hydra_zen.instantiate(config.visualizations, _convert_="all")

    training_metrics = hydra_zen.instantiate(config.training_metrics)
    evaluation_metrics = hydra_zen.instantiate(config.evaluation_metrics)

    train_vis_freq = config.training_vis_frequency if config.training_vis_frequency else 100

    assert checkpoint_path is None or config.load_checkpoint_partial is None, (
        "Cannot load a checkpoint and load a partial checkpoint at the same time. "
        "Please set only one of `load_checkpoint` or `load_checkpoint_partial`."
    )
    if checkpoint_path is None:
        model = CombinedModel(
            models=models,
            optimizers=optimizers,
            losses=losses,
            visualizations=visualizations,
            training_metrics=training_metrics,
            evaluation_metrics=evaluation_metrics,
            vis_log_frequency=train_vis_freq,
            trainable_models=config.trainable_models,
            modules_to_load=config.modules_to_load,
            load_checkpoint_partial=config.load_checkpoint_partial,
        )
    else:
        if checkpoint_hook is not None:

            def on_load_checkpoint(self, checkpoint):
                checkpoint_hook(checkpoint["state_dict"])

            CombinedModel.on_load_checkpoint = on_load_checkpoint
        model = CombinedModel.load_from_checkpoint(
            checkpoint_path,
            models=models,
            optimizers=optimizers,
            losses=losses,
            visualizations=visualizations,
            training_metrics=training_metrics,
            evaluation_metrics=evaluation_metrics,
            vis_log_frequency=train_vis_freq,
        )
        if checkpoint_hook is not None:
            CombinedModel.on_load_checkpoint = None
    return model


@hydra.main(config_name="training_config", config_path="../../configs/", version_base="1.1")
def train(config: TrainingConfig):
    # Set all relevant random seeds. If `config.seed is None`, the function samples a random value.
    # The function takes care of correctly distributing the seed across nodes in multi-node training,
    # and assigns each dataloader worker a different random seed.
    # IMPORTANTLY, we need to take care not to set a custom `worker_init_fn` function on the
    # dataloaders (or take care of worker seeding ourselves).
    pl.seed_everything(config.seed, workers=True)

    ocl.cli.cli_utils.set_torch_optimizations(enable=True)

    datamodule = build_and_register_datamodule_from_config(config)
    model = build_model_from_config(config)

    callbacks = hydra_zen.instantiate(config.trainer.callbacks, _convert_="all")
    callbacks = callbacks if callbacks else []
    if config.trainer.logger is not False:
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)

    log_dir = _get_runtime_path(config)
    if log_dir:
        checkpointer = pl.callbacks.ModelCheckpoint(
            dirpath=log_dir + "/" + cli_utils.CHECKPOINT_DIR,
            every_n_train_steps=config.experiment.get("checkpoint_every_n_steps", 5000),
        )
        callbacks.append(checkpointer)
    else:
        checkpointer = None

    if "exit_after" in config.experiment:
        timer = pl.callbacks.Timer(duration=config.experiment["exit_after"], interval="step")
        callbacks.append(timer)
    else:
        timer = None

    trainer: pl.Trainer = hydra_zen.instantiate(config.trainer, callbacks=callbacks, _convert_="all")

    if config.load_checkpoint:
        checkpoint_path = hydra.utils.to_absolute_path(config.load_checkpoint)
    else:
        checkpoint_path = None

    trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_path)

    if checkpointer is not None:
        # Explicitly save a final checkpoint after training.
        monitor_candidates = checkpointer._monitor_candidates(trainer)
        checkpointer._save_topk_checkpoint(trainer, monitor_candidates)

    if config.experiment.get("run_eval_after_training"):
        # Run one more evaluation. This is useful because some more training might have happened
        # after the last epoch, but before training was stopped.
        trainer.validate(model=model, datamodule=datamodule)

    # Hydra does not return anything from functions wrapped with hydra.main. As we still want to
    # have access to the result of training from the outside, we use a global (ugh) variable here.
    global RESULT_STATUS
    if timer and timer.time_remaining() <= 0:
        RESULT_STATUS = RESULT_TIMEOUT  # Signal that training was interrupted because of timeout
    else:
        RESULT_STATUS = RESULT_FINISHED

    return RESULT_STATUS


if __name__ == "__main__":
    train()
