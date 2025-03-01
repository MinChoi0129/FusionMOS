# MFMOS.py
import __init__ as booger
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *

# BaseBlocks.py의 클래스들을 임포트.
# ResContextBlock, ResBlock, UpBlock, MetaKernel 등이 들어있다고 가정.
# 다만, BEV 전용으로 pixelshuffle factor나 pooling kernel_size 등을 (2,2)로 쓰기 위해
# 아래처럼 별도 UpBlockBEV를 정의하거나, 기존 UpBlock에 파라미터를 추가할 수 있음.

from modules.BaseBlocks import (
    MetaKernel,
    ResContextBlock,
    ResBlock,
    UpBlock,  # Range 쪽에서 (2,4) pixelshuffle
)


class UpBlockBEV(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True):
        super(UpBlockBEV, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)

        # (중요) (2,2) pixelshuffle -> in_filters//4 만큼 채널이 감소
        self.conv1 = nn.Conv2d(
            in_filters // 4 + 2 * out_filters,  # 여기서 // 8 -> // 4 로 수정
            out_filters,
            (3, 3),
            padding=1,
        )
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), dilation=2, padding=2)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (2, 2), dilation=2, padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters * 3, out_filters, kernel_size=(1, 1))
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

    def pixelshuffle2x(self, x):
        """(2,2) pixelshuffle로 2배 업샘플"""
        B, iC, iH, iW = x.shape
        pH, pW = 2, 2
        oC = iC // (pH * pW)  # iC//4
        oH, oW = iH * pH, iW * pW

        y = x.reshape(B, oC, pH, pW, iH, iW)  # pixelshuffle
        y = y.permute(0, 1, 4, 2, 5, 3)
        y = y.reshape(B, oC, oH, oW)
        return y

    def forward(self, x, skip):
        # x: [B, in_filters, H, W]
        # skip: [B, out_filters*2, H, W] (ResBlockBEV 등에서 반환)
        upA = self.pixelshuffle2x(x)  # -> [B, in_filters//4, 2H, 2W]
        if self.drop_out:
            upA = self.dropout1(upA)

        upB = torch.cat((upA, skip), dim=1)  # 채널 합: in_filters//4 + skip_ch
        if self.drop_out:
            upB = self.dropout2(upB)

        # conv1 입력 채널수가 in_filters//4 + skip_ch와 동일해야 함
        upE = self.conv1(upB)
        upE = self.act1(upE)
        upE1 = self.bn1(upE)

        upE = self.conv2(upE1)
        upE = self.act2(upE)
        upE2 = self.bn2(upE)

        upE = self.conv3(upE2)
        upE = self.act3(upE)
        upE3 = self.bn3(upE)

        concat = torch.cat((upE1, upE2, upE3), dim=1)
        upE = self.conv4(concat)
        upE = self.act4(upE)
        upE = self.bn4(upE)
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE


class ResBlockBEV(nn.Module):
    """
    BEV용 ResBlock
    (2,2) Pooling으로 360->180->90->45
    """

    def __init__(
        self,
        in_filters,
        out_filters,
        dropout_rate,
        kernel_size=(2, 2),
        stride=1,
        pooling=True,
        drop_out=True,
    ):
        super(ResBlockBEV, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out

        self.conv1 = nn.Conv2d(
            in_filters, out_filters, kernel_size=(1, 1), stride=stride
        )
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(
            out_filters, out_filters, kernel_size=(3, 3), dilation=2, padding=2
        )
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(
            out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1
        )
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters * 3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)  # (2,2)로 반토막
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        resA = self.conv4(resA2)
        resA = self.act4(resA)
        resA3 = self.bn3(resA)

        concat = torch.cat((resA1, resA2, resA3), dim=1)
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = self.bn4(resA)
        resA = shortcut + resA

        if self.pooling:
            resB = self.dropout(resA) if self.drop_out else resA
            resB = self.pool(resB)
            return resB, resA
        else:
            resB = self.dropout(resA) if self.drop_out else resA
            return resB


class MFMOS(nn.Module):
    def __init__(
        self,
        nclasses,
        movable_nclasses,
        params,
        num_batch=None,
        point_refine=None,
        bev_in_ch=13,  # BEV 입력 채널 수
    ):
        super(MFMOS, self).__init__()
        self.nclasses = nclasses
        self.use_attention = "MGA"
        self.point_refine = point_refine

        # ---------------------------
        # Range Branch 설정
        # ---------------------------
        self.range_channel = 5
        print("Channel of range image input = ", self.range_channel)
        print("Number of residual images input = ", params["train"]["n_input_scans"])

        self.downCntx = ResContextBlock(self.range_channel, 32)
        self.downCntx2 = ResContextBlock(32, 32)
        self.downCntx3 = ResContextBlock(32, 32)
        print("params['train']['batch_size']", params["train"]["batch_size"])
        self.metaConv = MetaKernel(
            num_batch=(
                int(params["train"]["batch_size"]) if num_batch is None else num_batch
            ),
            feat_height=params["dataset"]["sensor"]["img_prop"]["height"],
            feat_width=params["dataset"]["sensor"]["img_prop"]["width"],
            coord_channels=self.range_channel,
        )
        print(
            'params["dataset"]["sensor"]["img_prop"]["height"] :',
            params["dataset"]["sensor"]["img_prop"]["height"],
        )

        # Range Branch Encoder
        self.resBlock1 = ResBlock(
            32, 64, 0.2, pooling=True, drop_out=False, kernel_size=(2, 4)
        )
        self.resBlock2 = ResBlock(64, 128, 0.2, pooling=True, kernel_size=(2, 4))
        self.resBlock3 = ResBlock(128, 256, 0.2, pooling=True, kernel_size=(2, 4))
        self.resBlock4 = ResBlock(256, 256, 0.2, pooling=True, kernel_size=(2, 4))
        self.resBlock5 = ResBlock(256, 256, 0.2, pooling=False, kernel_size=(2, 4))

        # Range Branch Decoder
        self.upBlock1 = UpBlock(256, 128, 0.2)
        self.upBlock2 = UpBlock(128, 128, 0.2)
        self.upBlock3 = UpBlock(128, 64, 0.2)
        self.upBlock4 = UpBlock(64, 32, 0.2, drop_out=False)

        # Residual Image(Temporal) Branch
        self.RI_downCntx = ResContextBlock(params["train"]["n_input_scans"], 32)

        self.RI_resBlock1 = ResBlock(
            32, 64, 0.2, pooling=True, drop_out=False, kernel_size=(2, 4)
        )
        self.RI_resBlock2 = ResBlock(64, 128, 0.2, pooling=True, kernel_size=(2, 4))
        self.RI_resBlock3 = ResBlock(128, 256, 0.2, pooling=True, kernel_size=(2, 4))
        self.RI_resBlock4 = ResBlock(256, 256, 0.2, pooling=True, kernel_size=(2, 4))
        self.RI_resBlock5 = ResBlock(256, 512, 0.2, pooling=False, kernel_size=(2, 4))

        # Range Branch 최종 로짓
        self.logits3 = nn.Conv2d(32, nclasses, kernel_size=(1, 1))
        self.movable_logits = nn.Conv2d(32, movable_nclasses, kernel_size=(1, 1))

        # ---------------------------
        # BEV Branch 설정 (새로 추가)
        #  - BEV는 (2,2) 풀링으로 3번만 다운샘플(360->180->90->45)
        #  - UpBlockBEV로 3번 업샘플
        # ---------------------------
        self.bev_downCntx = ResContextBlock(bev_in_ch, 32)
        self.bev_downCntx2 = ResContextBlock(32, 32)
        self.bev_downCntx3 = ResContextBlock(32, 32)

        self.bev_resBlock1 = ResBlockBEV(
            32, 64, 0.2, pooling=True, drop_out=False, kernel_size=(2, 2)
        )
        self.bev_resBlock2 = ResBlockBEV(64, 128, 0.2, pooling=True, kernel_size=(2, 2))
        self.bev_resBlock3 = ResBlockBEV(
            128, 256, 0.2, pooling=True, kernel_size=(2, 2)
        )

        self.bev_upBlock1 = UpBlockBEV(256, 128, 0.2)
        self.bev_upBlock2 = UpBlockBEV(128, 64, 0.2)
        self.bev_upBlock3 = UpBlockBEV(64, 32, 0.2, drop_out=False)

        self.bev_logits3 = nn.Conv2d(32, nclasses, kernel_size=(1, 1))
        self.movable_bev_logits = nn.Conv2d(32, movable_nclasses, kernel_size=(1, 1))

        # ---------------------------
        # Attention (MGA) 설정
        # ---------------------------
        if self.use_attention == "MGA":
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

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

            self.conv1x1_layer4_channel_wise = nn.Conv2d(256, 256, 1, bias=True)
            self.conv1x1_layer4_spatial = nn.Conv2d(256, 1, 1, bias=True)

            self.conv1x1_bev_channel_wise = nn.Conv2d(32, 32, 1, bias=True)
            self.conv1x1_bev_spatial = nn.Conv2d(32, 1, 1, bias=True)
            self.conv1x1_bev_layer1_channel_wise = nn.Conv2d(64, 64, 1, bias=True)
            self.conv1x1_bev_layer1_spatial = nn.Conv2d(64, 1, 1, bias=True)
            self.conv1x1_bev_layer2_channel_wise = nn.Conv2d(128, 128, 1, bias=True)
            self.conv1x1_bev_layer2_spatial = nn.Conv2d(128, 1, 1, bias=True)
        else:
            pass

    def encoder_attention_module_MGA_tmc(
        self, img_feat, flow_feat, conv1x1_channel_wise, conv1x1_spatial
    ):
        """
        간단한 MGA 예시: (spatial + channel attention)
        """
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
        """
        x_range: [B, 5 + n_input_scans, H, W] 형태로 Range + Residual 합쳐서 들어온다고 가정
                 (예: 5채널(range) + 8채널(residual) = 13채널)
        x_bev:   [B, bev_in_ch(=13), 360, 360] 형태로 BEV 입력
        """
        # ---------------------------
        # Range + Residual Branch
        # ---------------------------
        current_range_image = x_range[:, : self.range_channel, :, :]  # 5채널
        residual_images = x_range[
            :, self.range_channel :, :, :
        ]  # 나머지 (n_input_scans)채널

        # Residual Image Branch
        RI_downCntx = self.RI_downCntx(residual_images)
        RI_down0c, RI_down0b = self.RI_resBlock1(RI_downCntx)
        RI_down1c, RI_down1b = self.RI_resBlock2(RI_down0c)
        RI_down2c, RI_down2b = self.RI_resBlock3(RI_down1c)
        RI_down3c, RI_down3b = self.RI_resBlock4(RI_down2c)
        RI_down4c = self.RI_resBlock5(RI_down3c)

        # Range Image Branch
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

        # Attention 결합
        if self.use_attention == "MGA":
            downCntx = self.encoder_attention_module_MGA_tmc(
                RI_downCntx,
                downCntx,
                self.conv1x1_conv1_channel_wise,
                self.conv1x1_conv1_spatial,
            )
        else:
            downCntx += RI_downCntx

        down0c, down0b = self.resBlock1(downCntx)
        if self.use_attention == "MGA":
            down0c = self.encoder_attention_module_MGA_tmc(
                down0c,
                RI_down0c,
                self.conv1x1_layer0_channel_wise,
                self.conv1x1_layer0_spatial,
            )
        else:
            down0c += RI_down0c

        down1c, down1b = self.resBlock2(down0c)
        if self.use_attention == "MGA":
            down1c = self.encoder_attention_module_MGA_tmc(
                down1c,
                RI_down1c,
                self.conv1x1_layer1_channel_wise,
                self.conv1x1_layer1_spatial,
            )
        else:
            down1c += RI_down1c

        down2c, down2b = self.resBlock3(down1c)
        if self.use_attention == "MGA":
            down2c = self.encoder_attention_module_MGA_tmc(
                down2c,
                RI_down2c,
                self.conv1x1_layer2_channel_wise,
                self.conv1x1_layer2_spatial,
            )
        else:
            down2c += RI_down2c

        down3c, down3b = self.resBlock4(down2c)
        if self.use_attention == "MGA":
            down3c = self.encoder_attention_module_MGA_tmc(
                down3c,
                RI_down3c,
                self.conv1x1_layer3_channel_wise,
                self.conv1x1_layer3_spatial,
            )
        else:
            down3c += RI_down3c

        down5c = self.resBlock5(down3c)

        # Decoder(Range)
        up4e = self.upBlock1(down5c, down3b)
        up3e = self.upBlock2(up4e, down2b)
        up2e = self.upBlock3(up3e, down1b)
        up1e = self.upBlock4(up2e, down0b)

        logits_ = self.logits3(up1e)
        logits = F.softmax(logits_, dim=1)

        movable_logits_ = self.movable_logits(up1e)
        movable_logits = F.softmax(movable_logits_, dim=1)

        # ---------------------------
        # BEV Branch (새로 추가)
        # ---------------------------
        # [B, 13, 360, 360]를 가정
        bev_down0 = self.bev_downCntx(x_bev)
        bev_down0 = self.bev_downCntx2(bev_down0)
        bev_down0 = self.bev_downCntx3(bev_down0)
        if self.use_attention == "MGA":
            bev_down0 = self.encoder_attention_module_MGA_tmc(
                bev_down0,
                bev_down0,
                self.conv1x1_bev_channel_wise,
                self.conv1x1_bev_spatial,
            )

        bev_down1c, bev_down1b = self.bev_resBlock1(bev_down0)
        if self.use_attention == "MGA":
            bev_down1c = self.encoder_attention_module_MGA_tmc(
                bev_down1c,
                bev_down1c,
                self.conv1x1_bev_layer1_channel_wise,
                self.conv1x1_bev_layer1_spatial,
            )

        bev_down2c, bev_down2b = self.bev_resBlock2(bev_down1c)
        if self.use_attention == "MGA":
            bev_down2c = self.encoder_attention_module_MGA_tmc(
                bev_down2c,
                bev_down2c,
                self.conv1x1_bev_layer2_channel_wise,
                self.conv1x1_bev_layer2_spatial,
            )

        bev_down3c, bev_down3b = self.bev_resBlock3(bev_down2c)
        # (원하는 경우 여기서도 추가 attention 적용 가능)

        bev_up2e = self.bev_upBlock1(bev_down3c, bev_down3b)
        bev_up1e = self.bev_upBlock2(bev_up2e, bev_down2b)
        bev_up0e = self.bev_upBlock3(bev_up1e, bev_down1b)

        bev_logits_ = self.bev_logits3(bev_up0e)
        bev_logits = F.softmax(bev_logits_, dim=1)

        movable_bev_logits_ = self.movable_bev_logits(bev_up0e)
        movable_bev_logits = F.softmax(movable_bev_logits_, dim=1)

        # 4가지 출력 (quad branch)
        return logits, movable_logits, bev_logits, movable_bev_logits
