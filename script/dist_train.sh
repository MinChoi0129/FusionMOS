#!/bin/bash

DatasetPath=/home/ssd_4tb/minjae/KITTI/dataset
ArchConfig=./train_yaml/ddp_mos_coarse_stage.yml
DataConfig=./config/labels/semantic-kitti-mos.raw.yaml
LogPath=./log/Train

# 환경 변수 설정
export SETUPTOOLS_USE_DISTUTILS=stdlib
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="0"  # 단일 GPU 설정

# 단일 GPU 학습 실행
python ./train.py --dataset "$DatasetPath" \
                  --arch_cfg "$ArchConfig" \
                  --data_cfg "$DataConfig" \
                  --log "$LogPath"
                  # --pretrained "/home/work_docker/MF-MOS/log/Train/2025-2-18-05:09"


# #!/bin/bash

# DatasetPath=/home/ssd_4tb/minjae/KITTI/dataset
# ArchConfig=./train_yaml/ddp_mos_coarse_stage.yml
# DataConfig=./config/labels/semantic-kitti-mos.raw.yaml
# LogPath=./log/Train

# # 환경 변수 설정
# export SETUPTOOLS_USE_DISTUTILS=stdlib
# export OMP_NUM_THREADS=8
# export CUDA_VISIBLE_DEVICES="0,1,2,3"

# # 분산 학습 실행 (torchrun 사용)
# torchrun --nproc_per_node=4 \
#          --master_port=29500 \
#          -- \
#          ./train.py --dataset "$DatasetPath" \
#                     --arch_cfg "$ArchConfig" \
#                     --data_cfg "$DataConfig" \
#                     --log "$LogPath"
#                     # --pretrained "/home/work_docker/MF-MOS/log/Train/2025-2-18-05:09"
