"""Script to compute the size of a dataset.

This is useful when subsampling data using transformations in order to determine the final dataset
size.  The size of the dataset is typically need when running distributed training in order to
ensure that all nodes and gpu training processes are presented with the same number of batches.
"""
import dataclasses
import logging
import os
from typing import Any

import hydra
import hydra_zen
import tqdm


@dataclasses.dataclass
class ComputeSizeConfig:
    """Configuration of a training run."""

    dataset: Any


hydra.core.config_store.ConfigStore.instance().store(
    name="compute_size_config",
    node=ComputeSizeConfig,
)


@hydra.main(config_name="compute_size_config", config_path="../../configs", version_base="1.1")
def compute_size(config: ComputeSizeConfig):
    datamodule = hydra_zen.instantiate(config.dataset, _convert_="all")

    # Compute dataset sizes
    # TODO(hornmax): This is needed for webdataset shuffling, is there a way to make this more
    # elegant and less specific?
    os.environ["WDS_EPOCH"] = str(0)
    train_size = sum(
        1
        for _ in tqdm.tqdm(
            datamodule.train_data_iterator(), desc="Reading train split", unit="samples"
        )
    )
    logging.info("Train split size: %d", train_size)
    val_size = sum(
        1
        for _ in tqdm.tqdm(
            datamodule.val_data_iterator(), desc="Reading validation split", unit="samples"
        )
    )
    logging.info("Validation split size: %d", val_size)
    test_size = sum(
        1
        for _ in tqdm.tqdm(
            datamodule.test_data_iterator(), desc="Reading test split", unit="samples"
        )
    )
    logging.info("Test split size: %d", test_size)


if __name__ == "__main__":
    compute_size()
