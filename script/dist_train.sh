#!/bin/bash

# DatasetPath=/home/workspace/KITTI/dataset
# ArchConfig=./train_yaml/ddp_mos_coarse_stage.yml
# DataConfig=./config/labels/semantic-kitti-mos.raw.yaml
# LogPath=./log/Train

# # 환경 변수 설정
# export SETUPTOOLS_USE_DISTUTILS=stdlib
# export OMP_NUM_THREADS=1
# export CUDA_VISIBLE_DEVICES="0"  # 단일 GPU 설정

# # 단일 GPU 학습 실행
# python ./train.py --dataset "$DatasetPath" \
#                   --arch_cfg "$ArchConfig" \
#                   --data_cfg "$DataConfig" \
#                   --log "$LogPath" 
#                 #   --pretrained "/home/work/MF-MOS/log/Train/2025-3-10-06:21"

DatasetPath=/home/workspace/KITTI/dataset
ArchConfig=./train_yaml/ddp_mos_coarse_stage.yml
DataConfig=./config/labels/semantic-kitti-mos.raw.yaml
LogPath=./log/Train

# 환경 변수 설정
export SETUPTOOLS_USE_DISTUTILS=stdlib
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="1,2,3"

# 분산 학습 실행 (torchrun 사용)
torchrun --nproc_per_node=3 \
         --master_port=29500 \
         -- \
         ./train.py --dataset "$DatasetPath" \
                    --arch_cfg "$ArchConfig" \
                    --data_cfg "$DataConfig" \
                    --log "$LogPath" 
                    # --pretrained "/home/workspace/work/FusionMOS/log/Train/2025-3-19-04:07"
