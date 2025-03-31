#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import sys

sys.path.append("/home/workspace/work/FusionMOS")

import __init__ as booger
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *

from modules.BaseBlocks import MetaKernel, ResContextBlock, ResBlock, UpBlock


#########################################
# AdvancedAlignFeat 모듈 (항상 conv 적용)
#########################################
class AdvancedAlignFeat(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        """
        in_channels: 입력 extra feature의 채널 수
        out_channels: main branch의 해당 단계 채널에 맞출 목표 채널 수
        reduction: Squeeze-and-Excite에서 채널 축소 비율
        """
        super(AdvancedAlignFeat, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(out_channels, out_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, main_feat, side_feat):
        # main_feat: (B, C_main, H, W) - 기준 해상도(H,W)만 사용
        bs, _, H, W = main_feat.shape
        # side_feat를 항상 1x1 conv로 목표 채널(out_channels)로 변환
        side_feat = self.conv(side_feat)  # -> (B, out_channels, H_side, W_side)
        side_feat = self.bn(side_feat)
        side_feat = self.relu(side_feat)
        # 해상도 보간: main_feat의 H, W에 맞춤
        if side_feat.size(2) != H or side_feat.size(3) != W:
            side_feat = F.interpolate(
                side_feat, size=(H, W), mode="bilinear", align_corners=False
            )
        # Squeeze-and-Excite 적용
        attn = F.adaptive_avg_pool2d(side_feat, (1, 1))
        attn = self.relu(self.fc1(attn))
        attn = self.sigmoid(self.fc2(attn))
        side_feat = side_feat * attn
        return side_feat


#########################################
# MFMOS UNet 본체
#########################################
class MFMOS(nn.Module):
    """
    기존 MFMOS 구조에 추가 BEV extra feature들을
    Range branch에 AdvancedAlignFeat와 MGA로 결합합니다.
    """

    def __init__(self, nclasses, movable_nclasses, params, num_batch=None):
        super(MFMOS, self).__init__()
        self.nclasses = nclasses
        self.range_channel = 5

        # 1) 초기 Context
        self.downCntx = ResContextBlock(self.range_channel, 32)  # (B,32,64,2048)
        self.downCntx2 = ResContextBlock(32, 32)
        self.downCntx3 = ResContextBlock(32, 32)

        # MetaKernel
        self.metaConv = MetaKernel(
            num_batch=(
                int(params["train"]["batch_size"]) if num_batch is None else num_batch
            ),
            feat_height=params["dataset"]["sensor"]["img_prop"]["height"],
            feat_width=params["dataset"]["sensor"]["img_prop"]["width"],
            coord_channels=self.range_channel,
        )

        # 2) Encoder
        self.resBlock1 = ResBlock(
            32, 64, 0.2, pooling=True, drop_out=False, kernel_size=(2, 4)
        )  # (B,64,32,512)
        self.resBlock2 = ResBlock(
            64, 128, 0.2, pooling=True, kernel_size=(2, 4)
        )  # (B,128,16,128)
        self.resBlock3 = ResBlock(
            128, 256, 0.2, pooling=True, kernel_size=(2, 4)
        )  # (B,256,8,32)
        self.resBlock4 = ResBlock(
            256, 256, 0.2, pooling=True, kernel_size=(2, 4)
        )  # (B,256,4,8)
        self.resBlock5 = ResBlock(
            256, 256, 0.2, pooling=False, kernel_size=(2, 4)
        )  # (B,256,4,8)

        # 3) Decoder
        self.upBlock1 = UpBlock(256, 128, 0.2)
        self.upBlock2 = UpBlock(128, 128, 0.2)
        self.upBlock3 = UpBlock(128, 64, 0.2)
        self.upBlock4 = UpBlock(
            64, 32, 0.2, drop_out=False
        )  # 최종 출력: (B,32,64,2048)

        # 4) Residual branch 처리
        self.RI_downCntx = ResContextBlock(params["train"]["n_input_scans"], 32)
        self.RI_resBlock1 = ResBlock(
            32, 64, 0.2, pooling=True, drop_out=False, kernel_size=(2, 4)
        )
        self.RI_resBlock2 = ResBlock(64, 128, 0.2, pooling=True, kernel_size=(2, 4))
        self.RI_resBlock3 = ResBlock(128, 256, 0.2, pooling=True, kernel_size=(2, 4))
        self.RI_resBlock4 = ResBlock(256, 256, 0.2, pooling=True, kernel_size=(2, 4))
        self.RI_resBlock5 = ResBlock(256, 512, 0.2, pooling=False, kernel_size=(2, 4))

        # 최종 예측 Head
        self.logits3 = nn.Conv2d(32, nclasses, kernel_size=1)
        self.movable_logits = nn.Conv2d(32, movable_nclasses, kernel_size=1)

        # Range UpBlock (Movable branch 전용)
        self.range_upBlock1 = UpBlock(256, 128, 0.2)
        self.range_upBlock2 = UpBlock(128, 64, 0.2)
        self.range_upBlock3 = UpBlock(64, 32, 0.2)

        # Attention용 Conv1x1들 (MGA에서 사용)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_spatial_32 = nn.Conv2d(32, 1, 1, bias=True)  # 입력 채널 32
        self.conv_channel_32 = nn.Conv2d(32, 32, 1, bias=True)
        self.conv_spatial_64 = nn.Conv2d(64, 1, 1, bias=True)  # 입력 채널 64
        self.conv_channel_64 = nn.Conv2d(64, 64, 1, bias=True)
        self.conv_spatial_128 = nn.Conv2d(128, 1, 1, bias=True)  # 입력 채널 128
        self.conv_channel_128 = nn.Conv2d(128, 128, 1, bias=True)
        self.conv_spatial_256 = nn.Conv2d(256, 1, 1, bias=True)  # 입력 채널 256
        self.conv_channel_256 = nn.Conv2d(256, 256, 1, bias=True)

        # Advanced AlignFeat 모듈
        # f1: 입력 (B,64,313,313) → 출력 (B,32,64,2048)로 맞춤
        self.align_f1 = AdvancedAlignFeat(in_channels=64, out_channels=32)
        # f2: 입력 (B,128,157,157)
        # 중반 단계: main feature down1c: (B,128,16,128) → align_f2a: (B,128,16,128)
        self.align_f2a = AdvancedAlignFeat(in_channels=128, out_channels=128)
        # 후반 단계: main feature down2c: (B,256,8,32) → align_f2b: (B,128,8,32) 후에 conv로 256채널로 확장
        self.align_f2b = AdvancedAlignFeat(in_channels=128, out_channels=256)
        # f3: 입력 (B,256,79,79) → align_f3: (B,256,64,2048) → 후에 Decoder와 결합하여 (B,32,64,2048)
        self.align_f3 = AdvancedAlignFeat(in_channels=256, out_channels=32)

    def MGA(self, main_feat, side_feat, conv_channel, conv_spatial):
        # main_feat, side_feat: (B, C, H, W)
        sp_map = torch.sigmoid(conv_spatial(side_feat))  # -> (B, 1, H, W)
        sp_out = main_feat * sp_map  # (B, C, H, W)
        ch_vec = self.avg_pool(sp_out)  # (B, C, 1, 1)
        ch_vec = conv_channel(ch_vec)  # (B, C, 1, 1)
        ch_vec = torch.softmax(ch_vec, dim=1) * ch_vec.shape[1]
        return sp_out * ch_vec + main_feat  # (B, C, H, W)

    def forward(self, x, f1, f2, f3):
        """
        x  : (B, 13, 64, 2048)         # Range (5ch) + Residual (8ch)
        f1 : (B, 64, 313, 313)          # BEV extra feature (초반)
        f2 : (B, 128, 157, 157)         # BEV extra feature (중반)
        f3 : (B, 256, 79, 79)           # BEV extra feature (후반)
        """
        ###################################################################
        # 1) Range/Residual 분리
        ###################################################################
        current_range_image = x[:, : self.range_channel, :, :]  # (B,5,64,2048)
        residual_images = x[:, self.range_channel :, :, :]  # (B,8,64,2048)

        ###################################################################
        # 2) Pure Range Branch 처리
        ###################################################################
        range_feat = self.downCntx(current_range_image)  # (B,32,64,2048)
        range_feat = self.metaConv(
            range_feat,
            current_range_image,
            range_feat.size(1),
            current_range_image.size(1),
            3,
        )
        range_feat = self.downCntx2(range_feat)  # (B,32,64,2048)
        range_feat = self.downCntx3(range_feat)  # (B,32,64,2048)

        # f1 정렬: f1 (B,64,313,313) → align_f1 → (B,32,64,2048)
        f1_aligned = self.align_f1(range_feat, f1)
        range_feat = self.MGA(
            main_feat=range_feat,
            side_feat=f1_aligned,
            conv_channel=self.conv_channel_32,
            conv_spatial=self.conv_spatial_32,
        )  # (B,32,64,2048)

        ###################################################################
        # 3) Residual Branch 처리
        ###################################################################
        res_feat = self.RI_downCntx(residual_images)  # (B,32,64,2048)

        ###################################################################
        # 4) Range와 Residual Branch 융합 (초기 결합)
        ###################################################################
        combined_feat = self.MGA(
            main_feat=range_feat,
            side_feat=res_feat,
            conv_channel=self.conv_channel_32,
            conv_spatial=self.conv_spatial_32,
        )  # (B,32,64,2048)

        ###################################################################
        # 5) 초반 (Early) 단계: ResBlock1
        ###################################################################
        down0c, down0b = self.resBlock1(combined_feat)  # (B,64,32,512)
        down0c = self.MGA(
            main_feat=down0c,
            side_feat=down0c,
            conv_channel=self.conv_channel_64,
            conv_spatial=self.conv_spatial_64,
        )  # (B,64,32,512)

        ###################################################################
        # 6) 중반 (Mid) 단계: ResBlock2 및 ResBlock3
        ###################################################################
        down1c, down1b = self.resBlock2(down0c)  # (B,128,16,128)
        # f2 정렬: f2 (B,128,157,157) → align_f2a → (B,128,16,128)
        f2_aligned = self.align_f2a(down1c, f2)
        down1c = self.MGA(
            main_feat=down1c,
            side_feat=f2_aligned,
            conv_channel=self.conv_channel_128,
            conv_spatial=self.conv_spatial_128,
        )  # (B,128,16,128)
        down1c = self.MGA(
            main_feat=down1c,
            side_feat=down1c,
            conv_channel=self.conv_channel_128,
            conv_spatial=self.conv_spatial_128,
        )  # (B,128,16,128)

        down2c, down2b = self.resBlock3(down1c)  # (B,256,8,32)
        # f2 재융합: f2 (B,128,157,157) → align_f2b → (B,256,8,32)
        f2_aligned_2 = self.align_f2b(down2c, f2)
        down2c = self.MGA(
            main_feat=down2c,
            side_feat=f2_aligned_2,
            conv_channel=self.conv_channel_256,
            conv_spatial=self.conv_spatial_256,
        )  # (B,256,8,32)
        down2c = self.MGA(
            main_feat=down2c,
            side_feat=down2c,
            conv_channel=self.conv_channel_256,
            conv_spatial=self.conv_spatial_256,
        )  # (B,256,8,32)

        ###################################################################
        # 7) 후반 (Late) 단계: ResBlock4 및 ResBlock5
        ###################################################################
        down3c, down3b = self.resBlock4(down2c)  # (B,256,4,8)
        down3c = self.MGA(
            main_feat=down3c,
            side_feat=down3c,
            conv_channel=self.conv_channel_256,
            conv_spatial=self.conv_spatial_256,
        )  # (B,256,4,8)
        down5c = self.resBlock5(down3c)  # (B,256,4,8)

        ###################################################################
        # 8) Decoder: 업샘플링 진행
        ###################################################################
        up4e = self.upBlock1(down5c, down3b)  # (B,128,8,32)
        up3e = self.upBlock2(up4e, down2b)  # (B,128,16,128)
        up2e = self.upBlock3(up3e, down1b)  # (B,64,32,512)
        up1e = self.upBlock4(up2e, down0b)  # (B,32,64,2048)

        # f3 정렬: f3 (B,256,79,79) → align_f3 → (B,32,64,2048)
        f3_aligned = self.align_f3(up1e, f3)
        up1e = self.MGA(
            main_feat=up1e,
            side_feat=f3_aligned,
            conv_channel=self.conv_channel_32,
            conv_spatial=self.conv_spatial_32,
        )  # (B,32,64,2048)

        ###################################################################
        # 9) 최종 예측 Head: Moving 예측
        ###################################################################
        logits = self.logits3(up1e)  # (B, nclasses, 64, 2048)
        logits = F.softmax(logits, dim=1)

        ###################################################################
        # 10) Movable 예측: Range UpBlock 사용
        ###################################################################
        range_up4e = self.range_upBlock1(down3b, down2b)
        range_up3e = self.range_upBlock2(range_up4e, down1b)
        range_up2e = self.range_upBlock3(range_up3e, down0b)
        movable_logits = self.movable_logits(
            range_up2e
        )  # (B, movable_nclasses, 64, 2048)
        movable_logits = F.softmax(movable_logits, dim=1)

        ###################################################################
        # 11) Return: 두 출력을 모두 반환
        ###################################################################
        return logits, movable_logits


# 실행 예시 (입력 shape 수정)
if __name__ == "__main__":
    bs = 4
    x = torch.randn(bs, 13, 64, 2048)  # Range + Residual
    f1 = torch.randn(bs, 64, 313, 313)  # f1: (B,64,313,313)
    f2 = torch.randn(bs, 128, 157, 157)  # f2: (B,128,157,157)
    f3 = torch.randn(bs, 256, 79, 79)  # f3: (B,256,79,79)

    params_dummy = {
        "train": {"batch_size": bs, "n_input_scans": 8},
        "dataset": {"sensor": {"img_prop": {"height": 64, "width": 2048}}},
    }
    model = MFMOS(nclasses=3, movable_nclasses=3, params=params_dummy, num_batch=bs)
    out1, out2 = model(x, f1, f2, f3)
    print("out1.shape:", out1.shape)
    print("out2.shape:", out2.shape)
