#!/bin/bash

DATA_PATH=${DATA_PATH:=./outputs}

wget https://www.dropbox.com/sh/u1p1d6hysjxqauy/AABNwBWI9qn_K_AuUkXjGibIa/ScanNet/PNG?dl=0\&preview=ScanNet_train.zip

mv 'PNG?dl=0&preview=ScanNet_train.zip'  ScanNet_train.zip

mkdir -p $DATA_PATH/scannet

unzip ScanNet_train.zip -d $DATA_PATH/scannet

unzip $DATA_PATH/scannet/ScanNet_train.zip -d $DATA_PATH/scannet
unzip $DATA_PATH/scannet/ScanNet_test.zip -d $DATA_PATH/scannet

rm -rf ScanNet_train.zip
