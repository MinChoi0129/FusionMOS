#!/bin/bash

DatasetPath=/home/work_docker/KITTI/dataset
ArchConfig=./train_yaml/mos_pointrefine_stage.yml
DataConfig=./config/labels/semantic-kitti-mos.raw.yaml
LogPath=./log/TrainWithSIEM
FirstStageModelPath=/home/work_docker/MF-MOS/log/Train/2025-2-18-05:09

export SETUPTOOLS_USE_DISTUTILS=stdlib
export CUDA_VISIBLE_DEVICES=0 && python train_2stage.py -d $DatasetPath \
                                                        -ac $ArchConfig \
                                                        -dc $DataConfig \
                                                        -l $LogPath \
                                                        -p $FirstStageModelPath