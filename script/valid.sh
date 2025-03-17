#!/bin/bash

DatasetPath=/home/ssd_4tb/minjae/KITTI/dataset
ModelPath=/home/work/MF-MOS/log/Train/2025-3-10-06:21
SavePath=./log/Valid/predictions/
SPLIT=valid # valid or test

# If you want to use SIEM, set pointrefine on
export CUDA_VISIBLE_DEVICES=0 && python3 infer.py -d $DatasetPath \
                                                  -m $ModelPath \
                                                  -l $SavePath \
                                                  -s $SPLIT 
                                                  # --movable  
                                                #   --pointrefine 
