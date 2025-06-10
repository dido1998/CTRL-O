#!/bin/bash

DATA_PATH=${DATA_PATH:=./outputs}
OUTPUT_PATH=${OUTPUT_PATH:=./data}

COCO_SEED=23894734

case $1 in
  COCO)
    echo "Downloading COCO2017 and COCO20k data to $DATA_PATH/coco"
    # ./download_scripts/download_coco_data.sh

    echo "Converting COCO2017 to webdataset stored at $OUTPUT_PATH/coco2017"
    mkdir -p $OUTPUT_PATH/coco2017/train
    poetry run python conversion_scripts/convert_coco_resize.py $DATA_PATH/coco/train2017 $OUTPUT_PATH/coco2017/train --instance $DATA_PATH/coco/annotations/instances_train2017.json  --caption $DATA_PATH/coco/annotations/captions_train2017.json --category_embedding_path $DATA_PATH/coco/category_name_to_llama3_emb.pkl --seed $COCO_SEED
    mkdir -p $OUTPUT_PATH/coco2017/val
    poetry run python conversion_scripts/convert_coco_resize.py $DATA_PATH/coco/val2017 $OUTPUT_PATH/coco2017/val --instance $DATA_PATH/coco/annotations/instances_val2017.json --caption $DATA_PATH/coco/annotations/captions_val2017.json --category_embedding_path $DATA_PATH/coco/category_name_to_llama3_emb.pkl --seed $COCO_SEED --split val
    mkdir -p $OUTPUT_PATH/coco2017/test
    #poetry run python conversion_scripts/convert_coco.py $DATA_PATH/coco/test2017 $OUTPUT_PATH/coco2017/test --test $DATA_PATH/coco/annotations/image_info_test2017.json --category_embedding_path $DATA_PATH/coco/category_name_to_llama3_emb.pkl --seed $COCO_SEED
    mkdir -p $OUTPUT_PATH/coco2017/unlabeled
    #poetry run python conversion_scripts/convert_coco.py $DATA_PATH/coco/unlabeled2017 $OUTPUT_PATH/coco2017/unlabeled --test $DATA_PATH/coco/annotations/image_info_unlabeled2017.json --category_embedding_path $DATA_PATH/coco/category_name_to_llama3_emb.pkl --seed $COCO_SEED

    echo "Converting COCO20k to webdataset stored at $OUTPUT_PATH/coco2014/20k"
    mkdir -p $OUTPUT_PATH/coco2014/20k
    #poetry run python conversion_scripts/convert_coco.py $DATA_PATH/coco/train2014 $OUTPUT_PATH/coco2014/20k --instance $DATA_PATH/coco/annotations/instances_train2014.json --caption $DATA_PATH/coco/annotations/captions_train2014.json --seed $COCO_SEED --category_embedding_path $DATA_PATH/coco/category_name_to_llama3_emb.pkl --subset_list misc/coco20k_files.txt
    ;;

  COCO_panoptic)
    echo "Converting COCO2017 panoptic to webdataset stored at $OUTPUT_PATH/coco2017/val_panoptic"
    mkdir -p $OUTPUT_PATH/coco2017/val_panoptic
    poetry run python conversion_scripts/convert_coco.py $DATA_PATH/coco/val2017 $OUTPUT_PATH/coco2017/val_panoptic --seed $COCO_SEED --panoptic $DATA_PATH/coco/annotations/panoptic_val2017.json
    ;;

  clevr+cater)
    echo "Downloading clevr and cater data to $DATA_PATH/multi-object-datasets"
    mkdir -p $OUTPUT_PATH/multi-object-datasets/clevr_with_masks
    gsutil -m rsync -r gs://multi-object-datasets/clevr_with_masks $DATA_PATH/multi-object-datasets/clevr_with_masks
    mkdir -p $OUTPUT_PATH/multi-object-datasets/cater_with_masks
    gsutil -m rsync -r gs://multi-object-datasets/cater_with_masks $DATA_PATH/multi-object-datasets/cater_with_masks
    export CUDA_VISIBLE_DEVICES=""
    export TF_FORCE_GPU_ALLOW_GROWTH=True
    SEED=837452923

    echo "Converting clevr to webdataset stored at $DATA_PATH/clevr_with_masks"
    python conversion_scripts/convert_tfrecords.py clevr_with_masks $DATA_PATH/multi-object-datasets/clevr_with_masks/clevr_with_masks_train.tfrecords $OUTPUT_PATH/clevr_with_masks --split_names train val test --split_ratios 0.8 0.1 0.1 --n_instances 100000 --seed $SEED
    python conversion_scripts/convert_tfrecords.py clevr_with_masks $DATA_PATH/multi-object-datasets/clevr_with_masks/clevr_with_masks_train.tfrecords $OUTPUT_PATH/clevr_with_masks --split_names train val test --split_ratios 0.7 0.15 0.15 --n_instances 100000 --seed $SEED

    echo "Converting cater to webdataset stored at $DATA_PATH/cater_with_masks"
    python conversion_scripts/convert_tfrecords.py cater_with_masks "$DATA_PATH/multi-object-datasets/cater_with_masks/cater_with_masks_train.tfrecords-*-of-*" $OUTPUT_PATH/cater_with_masks --split_names train val --split_ratios 0.9 0.1 --n_instances 39364 --seed $SEED
    python conversion_scripts/convert_tfrecords.py cater_with_masks "$DATA_PATH/multi-object-datasets/cater_with_masks/cater_with_masks_test.tfrecords-*-of-*" $OUTPUT_PATH/cater_with_masks/test --n_instances 17100 --seed $SEED
    ;;


  clevrer)
    echo "Downloading clevrer data"
    ./download_scripts/download_clevrer_data.sh

    echo "Converting clevrer to webdataset stored at $DATA_PATH/clevrer"
    python conversion_scripts/convert_clevrer.py --video_dir="$DATA_PATH/clevrer/video_train/" --annotation_dir="$DATA_PATH/clevrer/annotation_train/" --output_dir="$OUTPUT_PATH/clevrer/train/"
    python conversion_scripts/convert_clevrer.py --video_dir="$DATA_PATH/clevrer/video_validation/" --annotation_dir="$DATA_PATH/clevrer/annotation_validation/" --output_dir="$OUTPUT_PATH/clevrer/validation/"
    ;;


  voc2007)
    echo "Creating voc2007 webdataset in $OUTPUT_PATH/voc2007"
    # Ensure downloaded data is stored in data folder.
    export TFDS_DATA_DIR=$DATA_PATH/tensorflow_datasets
    mkdir -p $OUTPUT_PATH/voc2007/train
    poetry run python conversion_scripts/convert_tfds.py extended_voc/2007-segmentation train $OUTPUT_PATH/voc2007/train
    mkdir -p $OUTPUT_PATH/voc2007/val
    poetry run python conversion_scripts/convert_tfds.py extended_voc/2007-segmentation validation $OUTPUT_PATH/voc2007/val
    mkdir -p $OUTPUT_PATH/voc2007/test
    poetry run python conversion_scripts/convert_tfds.py extended_voc/2007-segmentation test $OUTPUT_PATH/voc2007/test
    ;;


  voc2012)
    # Augmented pascal voc dataset with segmentations and additional instances.
    echo "Creating voc2012 webdataset in $OUTPUT_PATH/voc2012"
    # Ensure downloaded data is stored in data folder.
    export TFDS_DATA_DIR=$DATA_PATH/tensorflow_datasets
    mkdir -p $OUTPUT_PATH/voc2012/trainaug
    python conversion_scripts/convert_tfds.py extended_voc/2012-segmentation train+sbd_train+sbd_validation $OUTPUT_PATH/voc2012/trainaug
    # Regular pascal voc splits.
    mkdir -p $OUTPUT_PATH/voc2012/train
    python conversion_scripts/convert_tfds.py extended_voc/2012-segmentation train $OUTPUT_PATH/voc2012/train
    mkdir -p $OUTPUT_PATH/voc2012/val
    python conversion_scripts/convert_tfds.py extended_voc/2012-segmentation validation $OUTPUT_PATH/voc2012/val
    ;;


  movi_c)
    echo "Creating movi_c webdataset in $OUTPUT_PATH/movi_c"
    mkdir -p $OUTPUT_PATH/movi_c
    mkdir -p $OUTPUT_PATH/movi_c/train
    python conversion_scripts/convert_tfds.py movi_c/128x128:1.0.0 train $OUTPUT_PATH/movi_c/train --dataset_path gs://kubric-public/tfds
    mkdir -p $OUTPUT_PATH/movi_c/val
    python conversion_scripts/convert_tfds.py movi_c/128x128:1.0.0 validation $OUTPUT_PATH/movi_c/val --dataset_path gs://kubric-public/tfds
    mkdir -p $OUTPUT_PATH/movi_c/test
    python conversion_scripts/convert_tfds.py movi_c/128x128:1.0.0 test $OUTPUT_PATH/movi_c/test --dataset_path gs://kubric-public/tfds
    ;;


  ycb)
    echo "Creating YCB dataset in $OUTPUT_PATH/ycb"
    mkdir -p $OUTPUT_PATH/ycb
    mkdir -p $OUTPUT_PATH/ycb/train
    poetry run python conversion_scripts/convert_scannet_or_ycb.py --split train --data_path $DATA_PATH/ycb/ --output_path $OUTPUT_PATH/ycb/train
    mkdir -p $OUTPUT_PATH/ycb/val
    poetry run python conversion_scripts/convert_scannet_or_ycb.py --split val --data_path $DATA_PATH/ycb/ --output_path $OUTPUT_PATH/ycb/val
  ;;

  scannet)
    echo "Creating ScanNet dataset in $OUTPUT_PATH/scannet"
    mkdir -p $OUTPUT_PATH/scannet
    mkdir -p $OUTPUT_PATH/scannet/train
    poetry run python conversion_scripts/convert_scannet_or_ycb.py --split train --data_path $DATA_PATH/scannet/ --output_path $OUTPUT_PATH/scannet/train
    mkdir -p $OUTPUT_PATH/scannet/val
    poetry run python conversion_scripts/convert_scannet_or_ycb.py --split val --data_path $DATA_PATH/scannet/ --output_path $OUTPUT_PATH/scannet/val
  ;;

  entityseg)
    echo "Creating EntitySeg dataset in $OUTPUT_PATH/entityseg"
    mkdir -p $OUTPUT_PATH/entityseg
    mkdir -p $OUTPUT_PATH/entityseg/train
    poetry run python conversion_scripts/convert_entity_seg.py $DATA_PATH/entityseg/images/ $OUTPUT_PATH/entityseg/${split}/ --instance $DATA_PATH/entityseg/entityseg_insseg_${split}.json --panoptic $DATA_PATH/entityseg/entityseg_panseg_train.json
    mkdir -p $OUTPUT_PATH/entityseg/val
    poetry run python conversion_scripts/convert_entity_seg.py $DATA_PATH/entityseg/images/ $OUTPUT_PATH/entityseg/${split}/ --instance $DATA_PATH/entityseg/entityseg_insseg_${split}.json --panoptic $DATA_PATH/entityseg/entityseg_panseg_val.json
  ;;

  movi_e)
    echo "Creating movi_e webdataset in $OUTPUT_PATH/movi_e"
    mkdir -p $OUTPUT_PATH/movi_e
    mkdir -p $OUTPUT_PATH/movi_e/train
    poetry run python conversion_scripts/convert_tfds.py movi_e/128x128:1.0.0 train $OUTPUT_PATH/movi_e/train --dataset_path gs://kubric-public/tfds
    mkdir -p $OUTPUT_PATH/movi_e/val
    poetry run python conversion_scripts/convert_tfds.py movi_e/128x128:1.0.0 validation $OUTPUT_PATH/movi_e/val --dataset_path gs://kubric-public/tfds
    mkdir -p $OUTPUT_PATH/movi_e/test
    poetry run python conversion_scripts/convert_tfds.py movi_e/128x128:1.0.0 test $OUTPUT_PATH/movi_e/test --dataset_path gs://kubric-public/tfds
    ;;

  clevrtex)
    echo "Downloading ClevrTex data to $DATA_PATH/clevrtex"
    ./download_scripts/download_clevrtex_data.sh

    echo "Creating ClevrTex webdataset in $OUTPUT_PATH/clevrtex"
    mkdir -p $OUTPUT_PATH/clevrtex

    python conversion_scripts/convert_clevrtex --data-dir $DATA_PATH/clevrtex --output-dir $OUTPUT_PATH/clevrtex --variant full --split train
    python conversion_scripts/convert_clevrtex --data-dir $DATA_PATH/clevrtex --output-dir $OUTPUT_PATH/clevrtex --variant full --split val
    python conversion_scripts/convert_clevrtex --data-dir $DATA_PATH/clevrtex --output-dir $OUTPUT_PATH/clevrtex --variant full --split test
    python conversion_scripts/convert_clevrtex --data-dir $DATA_PATH/clevrtex --output-dir $OUTPUT_PATH/clevrtex --variant outd
    python conversion_scripts/convert_clevrtex --data-dir $DATA_PATH/clevrtex --output-dir $OUTPUT_PATH/clevrtex --variant camo --split train
    python conversion_scripts/convert_clevrtex --data-dir $DATA_PATH/clevrtex --output-dir $OUTPUT_PATH/clevrtex --variant camo --split val
    python conversion_scripts/convert_clevrtex --data-dir $DATA_PATH/clevrtex --output-dir $OUTPUT_PATH/clevrtex --variant camo --split test
    ;;

  clevrtex_variants)
    echo "Creating ClevrTex variants webdataset in $OUTPUT_PATH/clevrtex"
    python conversion_scripts/convert_clevrtex --data-dir $DATA_PATH/clevrtex --output-dir $OUTPUT_PATH/clevrtex --variant pbg --split train
    python conversion_scripts/convert_clevrtex --data-dir $DATA_PATH/clevrtex --output-dir $OUTPUT_PATH/clevrtex --variant pbg --split val
    python conversion_scripts/convert_clevrtex --data-dir $DATA_PATH/clevrtex --output-dir $OUTPUT_PATH/clevrtex --variant pbg --split test
    python conversion_scripts/convert_clevrtex --data-dir $DATA_PATH/clevrtex --output-dir $OUTPUT_PATH/clevrtex --variant vbg --split train
    python conversion_scripts/convert_clevrtex --data-dir $DATA_PATH/clevrtex --output-dir $OUTPUT_PATH/clevrtex --variant vbg --split val
    python conversion_scripts/convert_clevrtex --data-dir $DATA_PATH/clevrtex --output-dir $OUTPUT_PATH/clevrtex --variant vbg --split test
    python conversion_scripts/convert_clevrtex --data-dir $DATA_PATH/clevrtex --output-dir $OUTPUT_PATH/clevrtex --variant grassbg --split train
    python conversion_scripts/convert_clevrtex --data-dir $DATA_PATH/clevrtex --output-dir $OUTPUT_PATH/clevrtex --variant grassbg --split val
    python conversion_scripts/convert_clevrtex --data-dir $DATA_PATH/clevrtex --output-dir $OUTPUT_PATH/clevrtex --variant grassbg --split test
    ;;

  grit)
    echo "Creating Grit dataset in $OUTPUT_PATH/grit"
    mkdir -p $OUTPUT_PATH/grit_shards
    # will create both train and test shards, with 20% of data in test.
    poetry run python conversion_scripts/convert_grit_resize.py $DATA_PATH/grit $OUTPUT_PATH/grit_shards
    ;;

  *)
    echo "Unknown dataset $1"
    echo "Only COCO, COCO_panoptic, clevr+cater, clevrer, voc2007, voc2012, movi_c, movi_e, and grit are supported."
    ;;
esac
