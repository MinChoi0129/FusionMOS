#!/bin/bash

DatasetPath=/home/workspace/KITTI/dataset
PredictionsPath=/home/workspace/work/FusionMOS/log/Valid/predictions
DataConfig=./config/labels/semantic-kitti-mos.raw.yaml

python3 utils/evaluate_mos.py -d $DatasetPath \
                              -p $PredictionsPath \
                              -s valid \
                              --dc $DataConfig
