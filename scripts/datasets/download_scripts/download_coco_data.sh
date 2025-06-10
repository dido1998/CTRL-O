#!/bin/bash
# bsdtar can be installed on Ubuntu with package libarchive-tools

DATA_PATH=${DATA_PATH:=./outputs}

mkdir -p $DATA_PATH/coco
# COCO 2017
# Google storage hosted files seem to be down, use http instead.
# # Data
wget http://images.cocodataset.org/zips/train2017.zip
wget  http://images.cocodataset.org/zips/val2017.zip
wget  http://images.cocodataset.org/zips/test2017.zip
wget  http://images.cocodataset.org/zips/unlabeled2017.zip

unzip train2017.zip -d $DATA_PATH/coco
unzip val2017.zip -d $DATA_PATH/coco
unzip test2017.zip -d $DATA_PATH/coco
unzip unlabeled2017.zip -d $DATA_PATH/coco

# Annotations
wget  http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget  http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
wget  http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
# Test annotations
wget  http://images.cocodataset.org/annotations/image_info_test2017.zip
# Unlabeled annotations
wget  http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip

unzip annotations_trainval2017.zip -d $DATA_PATH/coco
unzip stuff_annotations_trainval2017.zip -d $DATA_PATH/coco
unzip panoptic_annotations_trainval2017.zip -d $DATA_PATH/coco
unzip image_info_test2017.zip -d $DATA_PATH/coco
unzip image_info_unlabeled2017.zip -d $DATA_PATH/coco

# COCO 2014, only train for generating 20k dataset
wget  http://images.cocodataset.org/zips/train2014.zip
wget  http://images.cocodataset.org/annotations/annotations_trainval2014.zip

unzip train2014.zip -d $DATA_PATH/coco
unzip annotations_trainval2014.zip -d $DATA_PATH/coco
