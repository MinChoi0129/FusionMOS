#!/bin/bash

DatasetPath=/home/ssd_4tb/minjae/KITTI/dataset
PredictionsPath=./log/Valid/predictions/
DataConfig=./config/labels/semantic-kitti-mos.raw.yaml

python3 utils/evaluate_mos.py -d $DatasetPath \
                              -p $PredictionsPath \
                              --dc $DataConfig
