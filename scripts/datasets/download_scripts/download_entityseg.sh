#!/bin/bash

DATA_PATH=${DATA_PATH:=./outputs}

poetry run python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('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')"


mkdir -p $DATA_PATH/entityseg

poetry run python download_scripts/hf_download_entityseg.py

mkdir -p $DATA_PATH/entityseg/images

mv $DATA_PATH/entityseg/downloads/extracted/9b35d0f25ad5ae6d8485f4d01265bf34fd4c75173bf28b81283d46d4c142582e/images/* $DATA_PATH/entityseg/images/
mv $DATA_PATH/entityseg/downloads/extracted/76a99ac878d8549a0c080f69c1418146899b1d2633c27da99c743fb36d5548b9/entity_03_10049/* $DATA_PATH/entityseg/images/
mv $DATA_PATH/entityseg/downloads/extracted/7315bbbcbd1ffcd3d85952d1dafad6de43b7a8c43fa854c65d13c8af40becce4e/images_merge/* $DATA_PATH/entityseg/images/
mv $DATA_PATH/entityseg/downloads/extracted/7315bbbcbd1ffcd3d85952d1dafad6de43b7a8c43fa854c65d13c8af40becce4e/images_merge/* $DATA_PATH/entityseg/images/
mv $DATA_PATH/entityseg/downloads/extracted/7192c39b55751664fcc0134d8f93af696f7aeafbbe80dc0943e918ede249f77a/entity_01_11580/* $DATA_PATH/entityseg/images/
mv $DATA_PATH/entityseg/downloads/extracted/c2ad91676bc914bcefd1987c665d1eeafb172d93ed2870b22638bd0327786387/images_03_10049/* $DATA_PATH/entityseg/images/
mv $DATA_PATH/entityseg/downloads/extracted/fc9f3747d1967a9e6b50970986f5dac3a6cd9e4f57673f0d31b3e1061e5fa938/entity_02_11598/* $DATA_PATH/entityseg/images/

wget https://github.com/adobe-research/EntitySeg-Dataset/releases/download/v1.0/entityseg_insseg_train.json -P $DATA_PATH/entityseg
wget https://github.com/adobe-research/EntitySeg-Dataset/releases/download/v1.0/entityseg_insseg_val.json -P $DATA_PATH/entityseg
wget https://github.com/adobe-research/EntitySeg-Dataset/releases/download/v1.0/entityseg_panseg_train.json -P $DATA_PATH/entityseg
wget https://github.com/adobe-research/EntitySeg-Dataset/releases/download/v1.0/entityseg_panseg_val.json -P $DATA_PATH/entityseg
