#!/bin/bash

DATA_PATH=${DATA_PATH:=./outputs}

wget https://www.dropbox.com/sh/u1p1d6hysjxqauy/AABGSNf8HOIfUyNZbZ7kQM_aa/YCB/PNG?dl=0\&preview=YCB_train.zip

mv 'PNG?dl=0&preview=YCB_train.zip'  YCB_train.zip

mkdir -p $DATA_PATH/ycb

unzip YCB_train.zip -d $DATA_PATH/ycb

unzip $DATA_PATH/ycb/YCB_train.zip -d $DATA_PATH/ycb
unzip $DATA_PATH/ycb/YCB_test.zip -d $DATA_PATH/ycb

rm -rf YCB_train.zip
