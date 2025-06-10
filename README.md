# CTRL-O: Language-Controllable Object-Centric Visual Representation Learning

*CVPR 2025* | **[Project Website](https://ctrl-o-paper.github.io/)**

> Object-centric representation learning aims to decompose visual scenes into fixed-size vectors called "slots" or "object files", where each slot captures a distinct object. CTRL-O introduces language-based control, enabling directed object extractions and multimodal applications, and achieves strong results on downstream tasks such as text-to-image generation and visual question answering.

![CTRL-O Demo](images/vg_demo.png)

Our code is based on the [Object Centric Learning Framwork](https://github.com/amazon-science/object-centric-learning-framework)

## Object Centric Learning Framework (OCLF)

[![Linting and Testing Status](https://github.com/amazon-science/object-centric-learning-framework/actions/workflows/lint_and_test.yaml/badge.svg?branch=main)](https://github.com/amazon-science/object-centric-learning-framework/actions/workflows/lint_and_test.yaml)
[![Docs site](https://img.shields.io/badge/docs-GitHub_Pages-blue)](https://amazon-science.github.io/object-centric-learning-framework/)


## What is OCLF?
OCLF (Object Centric Learning framework) is a framework designed to ease running
experiments for object centric learning research, yet is not limited to this
use case.  At its heart lies the idea that while code is not typically
composable many experiments in machine learning very similar with minor changes
and only represent minor changes.

One such example is multi-task training where a model might be trained to solve
multiple tasks at the same time.  Different ablations of said model would then
contain different model components but largely remain the same.

OCLF allows for such ablations without creating duplicate code by defining
models and experiments in configuration files and allowing their composition in
configuration space via [hydra](https://hydra.cc/).


## Quickstart - Development setup
Installing OCLF requires at least python3.8. Installation can be done using
[poetry](https://python-poetry.org/docs/#installation).  After installing
`poetry`, check out the repo and setup a development environment:

```bash
git clone https://github.com/amazon-science/object-centric-learning-framework.git
cd object-centric-learning-framework
poetry install
```

This installs the `ocl` package and the cli scripts used for running
experiments in a poetry managed virtual environment.

Next we need to prepare a dataset.  For this follow the steps below
to install the dependencies needed for dataset conversion and creation.


## Datasets

We provide pre-curated datasets for training CTRL-O.

1. VG + COCO: https://huggingface.co/adidolkar123/visual_genome_coco
2. VG: https://huggingface.co/adidolkar123/visual_genome/tree/main

To download these datasets use:

```
huggingface-cli download <dataset_name> --local-dir scripts/datasets/outputs/ --local-dir-use-symlinks False
```

We also provide scripts to create your own datasets:

For the coco dataset
```bash
cd scripts/datasets
poetry install
bash download_scripts/download_coco_data.sh
bash download_and_convert.sh COCO
```

This should create a webdataset in the path `scripts/datasets/outputs/coco`.

To run the experiments, the dataset needs to be exposed to OCLF

```bash
cd ../..   # Go back to root folder
export DATASET_PREFIX=scripts/datasets/outputs  # Expose dataset path
```

## Training

The main model from the paper is trained on VG+COCO data. To launch an experiment for this training run you can use:

```bash
poetry run ocl_train +experiment=projects/prompting/vg/prompt_vg_small14_dinov2_mapping_lang_point_pred_sep
```

This run should achieve a binding hits of ~60%.

The output of the training run should be stored at `outputs/projects/prompting/vg/prompt_vg_small14_dinov2_mapping_lang_point_pred_sep/<timestamp>`.

For a more detailed guide on how to install, setup, and use OCLF check out
the Tutorial in the docs.

## Inference and Visualization

We also provide inference and visualization scripts for the pretrained model in `ocl/cli/inference.py`

Before running the script, make sure to update the paths to the pretrained model checkpoint [here](language_conditioned_oclf/ocl/cli/inference.py#L32) and the images you want to use for inference [here](language_conditioned_oclf/ocl/cli/inference.py#L203).

```bash
poetry run python ocl/cli/inference.py
```

## Pretrained models

We provide a pretrained CTRL-O model on Hugging Face. You can download it using the following command:

```bash
huggingface-cli download rabiulawal/ctrlo --local-dir pretrained_models/ctrlo --local-dir-use-symlinks False
```

This will download the model checkpoint and configuration file into the `pretrained_models/ctrlo` directory. After downloading, please update the paths in `ocl/cli/inference.py` to point to the downloaded files.

## Citation

If you CTRL-O in your work please cite the bibtex entry below

```bibtex
@inproceedings{didolkar2025ctrlo,
    title={CTRL-O: Language-Controllable Object-Centric Visual Representation Learning},
    author={Didolkar, Aniket Rajiv and Zadaianchuk, Andrii and Awal, Rabiul and Seitzer, Maximilian and Gavves, Efstratios and Agrawal, Aishwarya},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2025}
}
```



## License
This project is licensed under the Apache-2.0 License.
