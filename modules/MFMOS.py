# MFMOS.py
import __init__ as booger
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

sys.path.append("/home/work/MF-MOS/")

from utils.utils import *
from modules.BaseBlocks import (
    MetaKernel,
    ResContextBlock,
    ResBlock,
    UpBlock,
    ResBlockBEV,
    UpBlockBEV,
    FlattenCrossAttentionBlock,
)


class MFMOS(nn.Module):
    def __init__(
        self,
        nclasses,
        movable_nclasses,
        params,
        num_batch=None,
        point_refine=None,
    ):
        super(MFMOS, self).__init__()
        self.nclasses = nclasses
        self.bs = num_batch if num_batch is not None else params["train"]["batch_size"]
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.n_input_scans = params["train"]["n_input_scans"]
        print("self.n_input_scans :", self.n_input_scans)

        # 채널
        self.range_channel = 5
        self.bev_channel = 5

        # 해상도
        self.range_height = 64
        self.range_width = 2048
        self.bev_height = 768
        self.bev_width = 768

        # ---------------------------
        # Range Branch 설정
        # ---------------------------
        self.downCntx = ResContextBlock(self.range_channel, 32)
        self.downCntx2 = ResContextBlock(32, 32)
        self.downCntx3 = ResContextBlock(32, 32)
        self.metaConv = MetaKernel(
            num_batch=self.bs,
            feat_height=self.range_height,
            feat_width=self.range_width,
            coord_channels=self.range_channel,
        )

        self.resBlock1 = ResBlock(
            32, 64, 0.2, pooling=True, drop_out=False, kernel_size=(2, 4)
        )
        self.resBlock2 = ResBlock(64, 128, 0.2, pooling=True, kernel_size=(2, 4))
        self.resBlock3 = ResBlock(128, 256, 0.2, pooling=True, kernel_size=(2, 4))
        self.resBlock4 = ResBlock(256, 256, 0.2, pooling=True, kernel_size=(2, 4))
        self.resBlock5 = ResBlock(256, 256, 0.2, pooling=False, kernel_size=(2, 4))

        self.upBlock1 = UpBlock(256, 128, 0.2)
        self.upBlock2 = UpBlock(128, 128, 0.2)
        self.upBlock3 = UpBlock(128, 64, 0.2)
        self.upBlock4 = UpBlock(64, 32, 0.2, drop_out=False)

        self.range_upBlock1 = UpBlock(256, 128, 0.2)
        self.range_upBlock2 = UpBlock(128, 64, 0.2)
        self.range_upBlock3 = UpBlock(64, 32, 0.2)

        self.RI_downCntx = ResContextBlock(self.n_input_scans, 32)
        self.RI_resBlock1 = ResBlock(
            32, 64, 0.2, pooling=True, drop_out=False, kernel_size=(2, 4)
        )
        self.RI_resBlock2 = ResBlock(64, 128, 0.2, pooling=True, kernel_size=(2, 4))
        self.RI_resBlock3 = ResBlock(128, 256, 0.2, pooling=True, kernel_size=(2, 4))
        self.RI_resBlock4 = ResBlock(256, 256, 0.2, pooling=True, kernel_size=(2, 4))
        self.RI_resBlock5 = ResBlock(256, 512, 0.2, pooling=False, kernel_size=(2, 4))

        self.logits3 = nn.Conv2d(32, nclasses, kernel_size=(1, 1))
        self.movable_logits = nn.Conv2d(32, movable_nclasses, kernel_size=(1, 1))

        # ---------------------------
        # BEV Branch 설정
        # ---------------------------
        self.bev_downCntx = ResContextBlock(self.bev_channel, 32)
        self.bev_downCntx2 = ResContextBlock(32, 32)
        self.bev_downCntx3 = ResContextBlock(32, 32)
        self.bev_metaConv = MetaKernel(
            num_batch=self.bs,
            feat_height=self.bev_height,
            feat_width=self.bev_width,
            coord_channels=self.bev_channel,
        )

        self.bev_resBlock1 = ResBlockBEV(
            32, 64, 0.2, pooling=True, drop_out=False, kernel_size=(2, 2)
        )
        self.bev_resBlock2 = ResBlockBEV(64, 128, 0.2, pooling=True, kernel_size=(2, 2))
        self.bev_resBlock3 = ResBlockBEV(
            128, 256, 0.2, pooling=True, kernel_size=(2, 2)
        )
        self.bev_resBlock4 = ResBlockBEV(
            256, 256, 0.2, pooling=True, kernel_size=(2, 2)
        )
        self.bev_resBlock5 = ResBlockBEV(
            256, 256, 0.2, pooling=False, kernel_size=(2, 2)
        )

        self.bev_upBlock1 = UpBlockBEV(256, 128, 0.2)
        self.bev_upBlock2 = UpBlockBEV(128, 128, 0.2)
        self.bev_upBlock3 = UpBlockBEV(128, 64, 0.2)
        self.bev_upBlock4 = UpBlockBEV(64, 32, 0.2, drop_out=False)

        self.birdeyeview_upBlock1 = UpBlockBEV(256, 128, 0.2)
        self.birdeyeview_upBlock2 = UpBlockBEV(128, 64, 0.2)
        self.birdeyeview_upBlock3 = UpBlockBEV(64, 32, 0.2)

        self.bev_RI_downCntx = ResContextBlock(self.n_input_scans, 32)
        self.bev_RI_resBlock1 = ResBlockBEV(
            32, 64, 0.2, pooling=True, drop_out=False, kernel_size=(2, 2)
        )
        self.bev_RI_resBlock2 = ResBlockBEV(
            64, 128, 0.2, pooling=True, kernel_size=(2, 2)
        )
        self.bev_RI_resBlock3 = ResBlockBEV(
            128, 256, 0.2, pooling=True, kernel_size=(2, 2)
        )
        self.bev_RI_resBlock4 = ResBlockBEV(
            256, 256, 0.2, pooling=True, kernel_size=(2, 2)
        )
        self.bev_RI_resBlock5 = ResBlockBEV(
            256, 512, 0.2, pooling=False, kernel_size=(2, 2)
        )

        self.bev_logits3 = nn.Conv2d(32, nclasses, kernel_size=(1, 1))
        self.bev_movable_logits = nn.Conv2d(32, movable_nclasses, kernel_size=(1, 1))

        # ---------------------------
        # Attention (MGA) 설정 (for Range)
        # ---------------------------
        self.conv1x1_conv1_channel_wise = nn.Conv2d(32, 32, 1, bias=True)
        self.conv1x1_conv1_spatial = nn.Conv2d(32, 1, 1, bias=True)
        self.conv1x1_layer0_channel_wise = nn.Conv2d(64, 64, 1, bias=True)
        self.conv1x1_layer0_spatial = nn.Conv2d(64, 1, 1, bias=True)
        self.conv1x1_layer1_channel_wise = nn.Conv2d(128, 128, 1, bias=True)
        self.conv1x1_layer1_spatial = nn.Conv2d(128, 1, 1, bias=True)
        self.conv1x1_layer2_channel_wise = nn.Conv2d(256, 256, 1, bias=True)
        self.conv1x1_layer2_spatial = nn.Conv2d(256, 1, 1, bias=True)
        self.conv1x1_layer3_channel_wise = nn.Conv2d(256, 256, 1, bias=True)
        self.conv1x1_layer3_spatial = nn.Conv2d(256, 1, 1, bias=True)

        # ---------------------------
        # Attention (MGA) 설정 (for BEV)
        # ---------------------------
        self.bev_conv1x1_conv1_channel_wise = nn.Conv2d(32, 32, 1, bias=True)
        self.bev_conv1x1_conv1_spatial = nn.Conv2d(32, 1, 1, bias=True)
        self.bev_conv1x1_layer0_channel_wise = nn.Conv2d(64, 64, 1, bias=True)
        self.bev_conv1x1_layer0_spatial = nn.Conv2d(64, 1, 1, bias=True)
        self.bev_conv1x1_layer1_channel_wise = nn.Conv2d(128, 128, 1, bias=True)
        self.bev_conv1x1_layer1_spatial = nn.Conv2d(128, 1, 1, bias=True)
        self.bev_conv1x1_layer2_channel_wise = nn.Conv2d(256, 256, 1, bias=True)
        self.bev_conv1x1_layer2_spatial = nn.Conv2d(256, 1, 1, bias=True)
        self.bev_conv1x1_layer3_channel_wise = nn.Conv2d(256, 256, 1, bias=True)
        self.bev_conv1x1_layer3_spatial = nn.Conv2d(256, 1, 1, bias=True)

        ###################################################
        # [수정] 두 UNet이 마지막 다운샘플 후 Feature 공유 (Cross Attention)
        ###################################################
        self.flatten_CAB = FlattenCrossAttentionBlock(channels=256)

    def encoder_attention_module_MGA_tmc(
        self, img_feat, flow_feat, conv1x1_channel_wise, conv1x1_spatial
    ):
        # spatial attention
        flow_feat_map = conv1x1_spatial(flow_feat)
        flow_feat_map = nn.Sigmoid()(flow_feat_map)
        spatial_attentioned_img_feat = flow_feat_map * img_feat

        # channel attention
        feat_vec = self.avg_pool(spatial_attentioned_img_feat)
        feat_vec = conv1x1_channel_wise(feat_vec)
        feat_vec = nn.Softmax(dim=1)(feat_vec) * feat_vec.shape[1]
        channel_attentioned_img_feat = spatial_attentioned_img_feat * feat_vec

        final_feat = channel_attentioned_img_feat + img_feat
        return final_feat

    def forward(self, x_range, x_bev):
        # ---------------------------
        # Range Branch
        # ---------------------------
        current_range_image = x_range[:, : self.range_channel, :, :]  # [B, 5, 64, 2048]
        residual_images = x_range[:, self.range_channel :, :, :]  # [B, 8, 64, 2048]

        downCntx = self.downCntx(current_range_image)
        downCntx = self.metaConv(
            data=downCntx,
            coord_data=current_range_image,
            data_channels=downCntx.size()[1],
            coord_channels=current_range_image.size()[1],
            kernel_size=3,
        )
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)

        RI_downCntx = self.RI_downCntx(residual_images)
        Range_down0c, Range_down0b = self.RI_resBlock1(downCntx)
        Range_down1c, Range_down1b = self.RI_resBlock2(Range_down0c)
        Range_down2c, Range_down2b = self.RI_resBlock3(Range_down1c)
        Range_down3c, Range_down3b = self.RI_resBlock4(Range_down2c)

        downCntx = self.encoder_attention_module_MGA_tmc(
            RI_downCntx,
            downCntx,
            self.conv1x1_conv1_channel_wise,
            self.conv1x1_conv1_spatial,
        )

        down0c, down0b = self.resBlock1(downCntx)
        down0c = self.encoder_attention_module_MGA_tmc(
            down0c,
            Range_down0c,
            self.conv1x1_layer0_channel_wise,
            self.conv1x1_layer0_spatial,
        )

        down1c, down1b = self.resBlock2(down0c)
        down1c = self.encoder_attention_module_MGA_tmc(
            down1c,
            Range_down1c,
            self.conv1x1_layer1_channel_wise,
            self.conv1x1_layer1_spatial,
        )

        down2c, down2b = self.resBlock3(down1c)
        down2c = self.encoder_attention_module_MGA_tmc(
            down2c,
            Range_down2c,
            self.conv1x1_layer2_channel_wise,
            self.conv1x1_layer2_spatial,
        )

        down3c, down3b = self.resBlock4(down2c)
        down3c = self.encoder_attention_module_MGA_tmc(
            down3c,
            Range_down3c,
            self.conv1x1_layer3_channel_wise,
            self.conv1x1_layer3_spatial,
        )

        # 최종 바닥 (4, 8)
        down5c = self.resBlock5(down3c)

        # ---------------------------
        # BEV Branch
        # ---------------------------
        current_bev_image = x_bev[:, : self.bev_channel, :, :]  # [B, 5, 768, 768]
        bev_residual_images = x_bev[:, self.bev_channel :, :, :]  # [B, 8, 768, 768]

        bev_downCntx = self.bev_downCntx(current_bev_image)
        bev_downCntx = self.bev_metaConv(
            data=bev_downCntx,
            coord_data=current_bev_image,
            data_channels=bev_downCntx.size(1),
            coord_channels=current_bev_image.size(1),
            kernel_size=3,
        )
        bev_downCntx = self.bev_downCntx2(bev_downCntx)
        bev_downCntx = self.bev_downCntx3(bev_downCntx)

        bev_RI_downCntx = self.bev_RI_downCntx(bev_residual_images)
        BirdEyeView_down0c, BirdEyeView_down0b = self.bev_RI_resBlock1(bev_downCntx)
        BirdEyeView_down1c, BirdEyeView_down1b = self.bev_RI_resBlock2(
            BirdEyeView_down0c
        )
        BirdEyeView_down2c, BirdEyeView_down2b = self.bev_RI_resBlock3(
            BirdEyeView_down1c
        )
        BirdEyeView_down3c, BirdEyeView_down3b = self.bev_RI_resBlock4(
            BirdEyeView_down2c
        )

        bev_downCntx = self.encoder_attention_module_MGA_tmc(
            bev_RI_downCntx,
            bev_downCntx,
            self.bev_conv1x1_conv1_channel_wise,
            self.bev_conv1x1_conv1_spatial,
        )

        bev_down0c, bev_down0b = self.bev_resBlock1(bev_downCntx)
        bev_down0c = self.encoder_attention_module_MGA_tmc(
            bev_down0c,
            BirdEyeView_down0c,
            self.bev_conv1x1_layer0_channel_wise,
            self.bev_conv1x1_layer0_spatial,
        )

        bev_down1c, bev_down1b = self.bev_resBlock2(bev_down0c)
        bev_down1c = self.encoder_attention_module_MGA_tmc(
            bev_down1c,
            BirdEyeView_down1c,
            self.bev_conv1x1_layer1_channel_wise,
            self.bev_conv1x1_layer1_spatial,
        )

        bev_down2c, bev_down2b = self.bev_resBlock3(bev_down1c)
        bev_down2c = self.encoder_attention_module_MGA_tmc(
            bev_down2c,
            BirdEyeView_down2c,
            self.bev_conv1x1_layer2_channel_wise,
            self.bev_conv1x1_layer2_spatial,
        )

        bev_down3c, bev_down3b = self.bev_resBlock4(bev_down2c)
        bev_down3c = self.encoder_attention_module_MGA_tmc(
            bev_down3c,
            BirdEyeView_down3c,
            self.bev_conv1x1_layer3_channel_wise,
            self.bev_conv1x1_layer3_spatial,
        )

        # 최종 바닥 (48, 48)
        bev_down5c = self.bev_resBlock5(bev_down3c)

        # ---------------------------
        # 정보 공유 부분
        # ---------------------------
        down5c, bev_down5c = self.flatten_CAB(down5c, bev_down5c)

        # ---------------------------
        # Range Branch Decoder
        # ---------------------------
        up4e = self.upBlock1(down5c, down3b)
        up3e = self.upBlock2(up4e, down2b)
        up2e = self.upBlock3(up3e, down1b)
        up1e = self.upBlock4(up2e, down0b)

        logits = self.logits3(up1e)
        range_probs = F.softmax(logits, dim=1)

        range_up4e = self.range_upBlock1(Range_down3b, Range_down2b)
        range_up3e = self.range_upBlock2(range_up4e, Range_down1b)
        range_up2e = self.range_upBlock3(range_up3e, Range_down0b)

        movable_logits = self.movable_logits(range_up2e)
        range_movable_probs = F.softmax(movable_logits, dim=1)

        # ---------------------------
        # BEV Branch Decoder
        # ---------------------------
        bev_up4e = self.bev_upBlock1(bev_down5c, bev_down3b)
        bev_up3e = self.bev_upBlock2(bev_up4e, bev_down2b)
        bev_up2e = self.bev_upBlock3(bev_up3e, bev_down1b)
        bev_up1e = self.bev_upBlock4(bev_up2e, bev_down0b)

        bev_logits = self.bev_logits3(bev_up1e)
        bev_probs = F.softmax(bev_logits, dim=1)

        bev_up4e = self.birdeyeview_upBlock1(BirdEyeView_down3b, BirdEyeView_down2b)
        bev_up3e = self.birdeyeview_upBlock2(bev_up4e, BirdEyeView_down1b)
        bev_up2e = self.birdeyeview_upBlock3(bev_up3e, BirdEyeView_down0b)

        bev_movable_logits = self.bev_movable_logits(bev_up2e)
        bev_movable_probs = F.softmax(bev_movable_logits, dim=1)

        # 최종 출력: Range와 BEV 각각의 Moving, Movable 로짓
        # [B, 3, 64, 2048], [B, 3, 64, 2048], [B, 3, 768, 768], [B, 3, 768, 768]
        return range_probs, range_movable_probs, bev_probs, bev_movable_probs


if __name__ == "__main__":

    params = {
        "train": {
            "n_input_scans": 8,  # residual 스캔 개수
            "batch_size": 1,
        },
    }
    nclasses = 3
    movable_nclasses = 3

    model = MFMOS(nclasses, movable_nclasses, params)
    if torch.cuda.is_available():
        model = model.cuda()

    # 더미 입력 생성
    x_range = torch.rand(
        params["train"]["batch_size"], 5 + params["train"]["n_input_scans"], 64, 2048
    )
    x_bev = torch.rand(
        params["train"]["batch_size"], 5 + params["train"]["n_input_scans"], 768, 768
    )
    if torch.cuda.is_available():
        x_range = x_range.cuda()
        x_bev = x_bev.cuda()

    # 추론
    range_logits, range_movable_logits, bev_logits, bev_movable_logits = model(
        x_range, x_bev
    )

    # 출력 형태 확인
    print("Range logits shape:", range_logits.shape)
    print("Range movable logits shape:", range_movable_logits.shape)
    print("BEV logits shape:", bev_logits.shape)
    print("BEV movable logits shape:", bev_movable_logits.shape)
