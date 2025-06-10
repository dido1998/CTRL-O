import glob
import os
import sys
from typing import Any, Dict, Optional

import hydra
import pandas as pd

from ocl.cli import cli_utils, train

try:
    import cluster
except ImportError as e:
    raise ModuleNotFoundError("MPI cluster dependency not installed") from e

hydra.core.config_store.ConfigStore.instance().store(
    name="training_config",
    node=train.TrainingConfig,
)

# Keys not to be transfered to experiment config
EXCLUDED_KEYS = ("experiment_path", "exit_for_resume_time", "working_dir", "id")


def read_metrics(working_dir: str) -> Optional[pd.Series]:
    path = os.path.join(working_dir, "**", "metrics.csv")
    metrics_csv_files = glob.glob(path, recursive=True)
    if len(metrics_csv_files) == 0:
        return None

    if "version" in metrics_csv_files[0]:
        # Parse version number as int for sorting
        sorter = lambda p: int(p.split("version_")[-1].split("/")[0])
        # Test sorting, if it fails unexpectedly, do not sort
        try:
            sorter(metrics_csv_files[0])
        except Exception:
            sorter = None
    else:
        sorter = None

    metrics_csv_file = sorted(metrics_csv_files, key=sorter)[-1]  # Take last (newest) metric file
    metrics_df = pd.read_csv(metrics_csv_file, sep=",")

    val_keys = sorted(k for k in metrics_df.keys() if k.startswith("val/"))
    val_key = None if len(val_keys) == 0 else val_keys[0]

    # Combine multiple rows at the same step, with mean() ignoring NaN cells.
    metrics_df = metrics_df.groupby("step", as_index=False, sort=False).mean()

    if val_key is None:
        metrics = metrics_df.iloc[-1].squeeze()  # Just return last row
    else:
        metrics_df = metrics_df.loc[metrics_df[val_key].notna()]  # Filter rows without validation
        metrics_df = metrics_df.loc[:, metrics_df.notna().all()]  # Filter columns without data
        metrics = metrics_df.iloc[-1].squeeze()  # Select last row

    return metrics


def _flatten_tree_recursive(in_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a tree structure to a flat Hydra path structure."""
    out_dict = {}
    for key, value in in_dict.items():
        if isinstance(value, dict):
            flattened = _flatten_tree_recursive(value)
            for postfix_key, val in flattened.items():
                out_dict[f"{key}.{postfix_key}"] = val
        else:
            out_dict[key] = value
    return out_dict


assert _flatten_tree_recursive({"a": {"b": "c"}}) == {"a.b": "c"}


def _get_tb_logger_conf(log_path: str) -> str:
    args = {
        "_target_": "pytorch_lightning.loggers.TensorBoardLogger",
        "save_dir": log_path,
        "name": "tb",
        "version": "",
    }
    return _format_dict_conf(args)


def _get_csv_logger_conf(log_path: str) -> str:
    args = {
        "_target_": "pytorch_lightning.loggers.CSVLogger",
        "save_dir": log_path,
        "name": "metrics",
    }
    return _format_dict_conf(args)


def _format_dict_conf(d) -> str:
    return "{" + ",".join(f'{k}:"{v}"' for k, v in d.items()) + "}"


def main(argv):
    is_secondary_ddp_node = "LOCAL_RANK" in os.environ and os.environ["LOCAL_RANK"] != "0"

    if is_secondary_ddp_node:
        del argv[1]
        params = cluster.read_params_from_cmdline(argv, verbose=False, save_params=False)
    else:
        params = cluster.read_params_from_cmdline(list(argv))

    config_overrides = _flatten_tree_recursive(
        {key: value for key, value in params.items() if key not in EXCLUDED_KEYS}
    )

    if "experiment_path" not in params:
        raise ValueError("Need to specify main experiment config via `experiment_path`.")

    log_dir = params["working_dir"]
    overrides = [
        f"+experiment={params['experiment_path']}",
        "+experiment.run_eval_after_training=true",
        f"hydra.run.dir={log_dir}",
        "hydra.output_subdir=config",
        f"trainer.default_root_dir={log_dir}",
        "trainer.logger=["
        + _get_tb_logger_conf(log_dir)
        + ","
        + _get_csv_logger_conf(log_dir)
        + "]",
    ]
    if "exit_for_resume_time" in params:
        overrides.append(f"+experiment.exit_after={params['exit_for_resume_time']}")

    for key, value in config_overrides.items():
        # Special null syntax: cluster utils does not allow a null value
        if value == "_null_":
            value = "null"

        # Special slash syntax: cluster utils does not allow slashes in parameter names That's
        # why we instead use a literal "-slash-" and replace it here.
        if "-slash-" in key:
            key = key.replace("-slash-", "/")

        # Special prefix syntax: cluster utils does not allow + and ~ in parameter names. That's
        # why we instead use a literal "plus-" and replace it here.
        if key.startswith("plus-"):
            key = key[len("plus-") :]
            overrides.append(f"+{key}={value}")
        elif key.startswith("plusplus-"):
            key = key[len("plusplus-") :]
            overrides.append(f"++{key}={value}")
        elif key.startswith("tilde-"):
            key = key[len("tilde-") :]
            overrides.append(f"~{key}={value}")
        else:
            overrides.append(f"{key}={value}")

    checkpoint_path = cli_utils.find_checkpoint(log_dir)
    if checkpoint_path is not None:
        print(f"Using existing checkpoint {checkpoint_path}")
        overrides.append(f'load_checkpoint="{checkpoint_path}"')

    # Useful for debug:
    # with hydra.initialize(version_base="1.1", config_path="../../configs/"):
    #    cfg = hydra.compose(config_name="training_config", overrides=overrides)
    #    result = train.train(cfg)
    sys.argv = [sys.argv[0]] + overrides
    train.train()
    result = train.RESULT_STATUS

    if not is_secondary_ddp_node:
        metrics = read_metrics(log_dir)

        if result == train.RESULT_TIMEOUT:
            if metrics is not None:
                cluster.announce_early_results(metrics.to_dict())
            print("Exit for resume")
            cluster.exit_for_resume()
        else:
            if metrics is None:
                print("Warning: metrics.csv not found")
            else:
                cluster.save_metrics_params(metrics.to_dict(), params)


if __name__ == "__main__":
    main(sys.argv)
