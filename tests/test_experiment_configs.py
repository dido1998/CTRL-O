import glob
import os

import hydra
import hydra_zen
import pytest

import ocl.cli.eval
import ocl.cli.train

TRAIN_OVERRIDES = [
    # Minimal test run.  Use same parameters as trainer.fast_dev_run=True but keep logging enabled.
    "trainer.devices=1",
    "trainer.max_steps=1",
    "trainer.limit_train_batches=1",
    "trainer.limit_val_batches=1",
    "trainer.limit_test_batches=1",
    "trainer.val_check_interval=1.0",
    "trainer.num_sanity_val_steps=0",  # We run one val batch at the end of the fake training run.
    "++dataset.batch_size=2",
    "++dataset.num_workers=0",
    "++dataset.shuffle_buffer_size=1",  # Disable shuffling when running tests to speed things up.
]



def is_valid_file(filename):
    # Check if the file name starts with an underscore
    if os.path.basename(filename).startswith("_"):
        return False
    # Check if any part of the path starts with an underscore
    for part in os.path.dirname(filename).split(os.sep):
        if part.startswith("_"):
            return False
    return True

def _remove_filename_components(path):
    """Convert a in the experiment folder to a valid setting for experiment."""
    # Remove file extension.
    path, ext = os.path.splitext(path)
    # Remove `config/experiment` prefix.
    return os.path.join(*path.split(os.path.sep)[2:])


EXPERIMENTS = list(
    map(
        _remove_filename_components,
        filter(
            is_valid_file,
            glob.glob("configs/experiment/**/*.yaml", recursive=True),
        ),
    )
)

PROJECT_EXPERIMENTS = list(
    map(
        _remove_filename_components,
        filter(
            is_valid_file,
            glob.glob("configs/experiment/projects/prompting/**/*.yaml", recursive=True),
        ),
    )
)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore:LightningDeprecationWarning")
@pytest.mark.slow
@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_experiment_config(experiment, tmpdir):
    with hydra.initialize(config_path="../configs", version_base="1.1"):
        # Need to explicitly register TrainingConfig again, because we opened a new Hydra context.
        hydra.core.config_store.ConfigStore.instance().store(
            name="training_config",
            node=ocl.cli.train.TrainingConfig,
        )

        overrides = [f"+experiment={experiment}", f"hydra.run.dir={tmpdir}"]
        config = hydra.compose("training_config", overrides=overrides + TRAIN_OVERRIDES)
        ocl.cli.train.train(config)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore:LightningDeprecationWarning")
@pytest.mark.parametrize("experiment", PROJECT_EXPERIMENTS)
def test_experiment_configs_loadable(experiment):
    """Test checking that configs load and dataset and model can be constructed from them.

    We only check the configs in `experiments/projects/` for now.
    """
    with hydra.initialize(config_path="../configs", version_base="1.1"):
        # Need to explicitly register TrainingConfig again, because we opened a new Hydra context.
        hydra.core.config_store.ConfigStore.instance().store(
            name="training_config",
            node=ocl.cli.train.TrainingConfig,
        )
        overrides = [f"+experiment={experiment}"]
        config = hydra.compose("training_config", overrides=overrides)

        os.environ["DATASET_PREFIX"] = "./outputs"  # Set dummy dataset prefix
        ocl.cli.train.build_and_register_datamodule_from_config(config)
        ocl.cli.train.build_model_from_config(config)
        callbacks = hydra_zen.instantiate(config.trainer.callbacks, _convert_="all")
        hydra_zen.instantiate(config.trainer, callbacks=callbacks, devices="auto", _convert_="all")
