#!/bin/bash

DatasetPath=/home/workspace/KITTI/dataset
ModelPath=/home/workspace/work/FusionMOS/log/Train/2025-3-20-07:25
SavePath=./log/Valid/predictions/
SPLIT=valid # valid or test

# If you want to use SIEM, set pointrefine on
export CUDA_VISIBLE_DEVICES=1 && python3 infer.py -d $DatasetPath \
                                                  -m $ModelPath \
                                                  -l $SavePath \
                                                  -s $SPLIT 
                                                  # --movable  
                                                #   --pointrefine 
