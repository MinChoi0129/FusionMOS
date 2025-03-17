#!/bin/bash

DatasetPath=/home/ssd_4tb/minjae/KITTI/dataset
PredictionsPath=/home/work/MF-MOS/log/Valid/predictions
DataConfig=./config/labels/semantic-kitti-mos.raw.yaml

python3 utils/evaluate_mos.py -d $DatasetPath \
                              -p $PredictionsPath \
                              -s valid \
                              --dc $DataConfig
