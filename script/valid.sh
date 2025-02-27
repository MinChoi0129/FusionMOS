#!/bin/bash

DatasetPath=/home/work_docker/KITTI/dataset
ModelPath=/home/work_docker/MF-MOS/log/TrainWithSIEM/2025-2-20-08:58
SavePath=./log/Valid/predictions/
SPLIT=valid # valid or test

# If you want to use SIEM, set pointrefine on
export CUDA_VISIBLE_DEVICES=0 && python3 infer.py -d $DatasetPath \
                                                  -m $ModelPath \
                                                  -l $SavePath \
                                                  -s $SPLIT \
                                                  --movable  \
                                                  --pointrefine 
