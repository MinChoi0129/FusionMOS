# !/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import sys
sys.path.append("/home/workspace/work/FusionMOS")


import __init__ as booger
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *


from modules.BaseBlocks import MetaKernel, ResContextBlock, ResBlock, UpBlock

##############################################################################
# (Optional) shape/채널 맞춤 유틸 함수
##############################################################################
def align_feat(main_feat, side_feat):
    """
    main_feat.shape: (bs, C_main, H_main, W_main)
    side_feat.shape: (bs, C_side, H_side, W_side)

    - 채널이 다르면 side_feat에 1×1 Conv 수행
    - spatial 크기가 다르면 F.interpolate
    - 반환: side_feat를 main_feat와 동일 (채널/해상도)로 맞춘 텐서
    """
    bs, main_c, main_h, main_w = main_feat.shape
    bs2, side_c, side_h, side_w = side_feat.shape

    # 1) 채널 맞추기 (1x1 conv) - 예시
    if side_c != main_c:
        conv = nn.Conv2d(side_c, main_c, kernel_size=1, bias=False).to(side_feat.device)
        bn   = nn.BatchNorm2d(main_c).to(side_feat.device)
        with torch.no_grad():
            side_feat = bn(conv(side_feat))

    # 2) 해상도 보간
    if (side_h != main_h) or (side_w != main_w):
        side_feat = F.interpolate(side_feat, size=(main_h, main_w),
                                  mode='bilinear', align_corners=False)
    return side_feat

##############################################################################
# (1) MFMOS UNet 본체
##############################################################################
class MFMOS(nn.Module):
    """
    기존 MFMOS 구조에, forward(x, f1, f2, f3)로 추가 BEV-like feature들을
    초/중/후반 각 레이어에서 Attention으로 결합.
    """
    def __init__(self, nclasses, movable_nclasses, params, num_batch=None):
        super(MFMOS, self).__init__()
        self.nclasses = nclasses
        self.range_channel = 5

        # 1) 초기 Context
        self.downCntx  = ResContextBlock(self.range_channel, 32)
        self.downCntx2 = ResContextBlock(32, 32)
        self.downCntx3 = ResContextBlock(32, 32)

        # MetaKernel
        self.metaConv = MetaKernel(
            num_batch=int(params["train"]["batch_size"]) if num_batch is None else num_batch,
            feat_height=params["dataset"]["sensor"]["img_prop"]["height"],
            feat_width=params["dataset"]["sensor"]["img_prop"]["width"],
            coord_channels=self.range_channel,
        )

        # 2) Encoder
        self.resBlock1 = ResBlock(32, 64, 0.2, pooling=True,  drop_out=False, kernel_size=(2,4))
        self.resBlock2 = ResBlock(64,128, 0.2, pooling=True,  kernel_size=(2,4))
        self.resBlock3 = ResBlock(128,256,0.2, pooling=True,  kernel_size=(2,4))
        self.resBlock4 = ResBlock(256,256,0.2, pooling=True,  kernel_size=(2,4))
        self.resBlock5 = ResBlock(256,256,0.2, pooling=False, kernel_size=(2,4))

        # 3) Decoder
        self.upBlock1 = UpBlock(256,128,0.2)
        self.upBlock2 = UpBlock(128,128,0.2)
        self.upBlock3 = UpBlock(128,64, 0.2)
        self.upBlock4 = UpBlock(64,  32, 0.2, drop_out=False)

        # 4) Residual branch
        self.RI_downCntx = ResContextBlock(params["train"]["n_input_scans"], 32)
        self.RI_resBlock1= ResBlock(32,64,  0.2, pooling=True, drop_out=False, kernel_size=(2,4))
        self.RI_resBlock2= ResBlock(64,128, 0.2, pooling=True, kernel_size=(2,4))
        self.RI_resBlock3= ResBlock(128,256,0.2, pooling=True, kernel_size=(2,4))
        self.RI_resBlock4= ResBlock(256,256,0.2, pooling=True, kernel_size=(2,4))
        self.RI_resBlock5= ResBlock(256,512,0.2, pooling=False, kernel_size=(2,4))

        # 최종 예측 Head
        self.logits3        = nn.Conv2d(32, nclasses,        kernel_size=1)
        self.movable_logits = nn.Conv2d(32, movable_nclasses,kernel_size=1)

        # Range UpBlock
        self.range_upBlock1 = UpBlock(256, 128, 0.2)
        self.range_upBlock2 = UpBlock(128, 64, 0.2)
        self.range_upBlock3 = UpBlock(64, 32, 0.2)

        # Attention용 Conv1x1들
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        # 32채널
        self.conv_spatial_32  = nn.Conv2d(32, 1, 1, bias=True)
        self.conv_channel_32  = nn.Conv2d(32, 32,1, bias=True)
        # 64채널
        self.conv_spatial_64  = nn.Conv2d(64, 1, 1, bias=True)
        self.conv_channel_64  = nn.Conv2d(64, 64,1, bias=True)
        # 128채널
        self.conv_spatial_128 = nn.Conv2d(128,1, 1, bias=True)
        self.conv_channel_128 = nn.Conv2d(128,128,1,bias=True)
        # 256채널
        self.conv_spatial_256 = nn.Conv2d(256,1, 1, bias=True)
        self.conv_channel_256 = nn.Conv2d(256,256,1,bias=True)

    def MGA(self, main_feat, side_feat, conv_channel, conv_spatial):
        """
        Motion Guided Attention (MGA) 간소화 버전:
         - Spatial Attention
         - Channel Attention
        """
        # (1) Spatial
        sp_map = torch.sigmoid(conv_spatial(side_feat))
        sp_out = main_feat * sp_map

        # (2) Channel
        ch_vec = self.avg_pool(sp_out)
        ch_vec = conv_channel(ch_vec)
        ch_vec = torch.softmax(ch_vec, dim=1)*ch_vec.shape[1]

        return sp_out*ch_vec + main_feat

    def forward(self, x, f1, f2, f3):
        """
        x : (bs, 13, 64, 2048)  # range(5ch) + residual(~8ch)
        f1: (bs, 64,  32, 512)  # 초반(early) 결합용 Feature
        f2: (bs, 256, 8,  32)   # 중반(mid) 결합용 Feature
        f3: (bs, 32,  64, 2048) # 후반(late) 결합용 Feature
        """

        ###################################################################
        # 1) 간단한 shape/채널 맞춤 함수 (forward 내부에서만 사용)
        ###################################################################
        def align_feat(main_feat, side_feat):
            """
            main_feat: (bs, C_main, H_main, W_main)
            side_feat: (bs, C_side, H_side, W_side)

            1) side_feat 채널 != main_feat 채널이면 1x1 Conv 후 BN
            2) 공간 해상도 다르면 F.interpolate로 (H_main, W_main) 맞춤
            """
            b, c_main, h_main, w_main = main_feat.shape
            _, c_side, h_side, w_side = side_feat.shape

            # 1) 채널 맞춤
            if c_side != c_main:
                conv_1x1 = nn.Conv2d(c_side, c_main, kernel_size=1, bias=False).to(side_feat.device)
                bn_1x1   = nn.BatchNorm2d(c_main).to(side_feat.device)
                with torch.no_grad():
                    side_feat = bn_1x1(conv_1x1(side_feat))

            # 2) 공간 맞춤
            if (h_side != h_main) or (w_side != w_main):
                side_feat = F.interpolate(side_feat, size=(h_main, w_main),
                                        mode='bilinear', align_corners=False)
            return side_feat

        ###################################################################
        # 2) Range/Residual 분리
        ###################################################################
        current_range_image = x[:, : self.range_channel, :, :]  # (bs,5,64,2048)
        residual_images     = x[:, self.range_channel:, :, :]   # (bs,8,64,2048)

        ###################################################################
        # 3) Residual branch 인코더
        ###################################################################
        RI_downCntx = self.RI_downCntx(residual_images)               # (bs,32,64,2048)
        Range_down0c, Range_down0b = self.RI_resBlock1(RI_downCntx)   # (bs,64,32,512), (bs,64,64,2048)
        Range_down1c, Range_down1b = self.RI_resBlock2(Range_down0c)  # (bs,128,16,128), (bs,128,32,512)
        Range_down2c, Range_down2b = self.RI_resBlock3(Range_down1c)  # (bs,256,8, 32),  (bs,256,16,128)
        Range_down3c, Range_down3b = self.RI_resBlock4(Range_down2c)  # (bs,256,4, 8),   (bs,256,8, 32)

        ###################################################################
        # 4) Range branch 초기: downCntx
        ###################################################################
        downCntx = self.downCntx(current_range_image)  # (bs,32,64,2048)
        downCntx = self.metaConv(downCntx, current_range_image,
                                downCntx.size(1), current_range_image.size(1), 3)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)            # (bs,32,64,2048)

        ###################################################################
        # 5) 초반(early) 결합 (down0c / f1)
        ###################################################################
        # residual branch와 Attention (32채널)
        downCntx = self.MGA(
            main_feat=downCntx,
            side_feat=RI_downCntx,
            conv_channel=self.conv_channel_32,  # 32채널용
            conv_spatial=self.conv_spatial_32
        )
        # ResBlock1 -> down0c
        down0c, down0b = self.resBlock1(downCntx)      # (bs,64,32,512)

        # f1 결합 (64채널)
        f1_aligned = align_feat(down0c, f1)
        down0c = self.MGA(
            main_feat=down0c,
            side_feat=f1_aligned,
            conv_channel=self.conv_channel_64,
            conv_spatial=self.conv_spatial_64
        )
        # Range_down0c와도 Attention
        down0c = self.MGA(
            main_feat=down0c,
            side_feat=Range_down0c,
            conv_channel=self.conv_channel_64,
            conv_spatial=self.conv_spatial_64
        )

        ###################################################################
        # 6) 중반(mid) 결합 (down2c / f2)
        ###################################################################
        down1c, down1b = self.resBlock2(down0c)  # (bs,128,16,128)
        down1c = self.MGA(
            main_feat=down1c,
            side_feat=Range_down1c,
            conv_channel=self.conv_channel_128,
            conv_spatial=self.conv_spatial_128
        )

        down2c, down2b = self.resBlock3(down1c)  # (bs,256,8,32)
        f2_aligned = align_feat(down2c, f2)
        down2c = self.MGA(
            main_feat=down2c,
            side_feat=f2_aligned,
            conv_channel=self.conv_channel_256,
            conv_spatial=self.conv_spatial_256
        )
        down2c = self.MGA(
            main_feat=down2c,
            side_feat=Range_down2c,
            conv_channel=self.conv_channel_256,
            conv_spatial=self.conv_spatial_256
        )

        ###################################################################
        # 7) 후반(late) 결합 (up1e / f3)
        ###################################################################
        down3c, down3b = self.resBlock4(down2c)  # (bs,256,4,8)
        down3c = self.MGA(
            main_feat=down3c,
            side_feat=Range_down3c,
            conv_channel=self.conv_channel_256,
            conv_spatial=self.conv_spatial_256
        )
        down5c = self.resBlock5(down3c)          # (bs,256,4,8)

        # Decoder
        up4e = self.upBlock1(down5c, down3b)  # (bs,128,8,32)
        up3e = self.upBlock2(up4e,   down2b)  # (bs,128,16,128)
        up2e = self.upBlock3(up3e,   down1b)  # (bs,64, 32,512)
        up1e = self.upBlock4(up2e,   down0b)  # (bs,32, 64,2048)

        # f3 결합 (32채널)
        f3_aligned = align_feat(up1e, f3)
        up1e = self.MGA(
            main_feat=up1e,
            side_feat=f3_aligned,
            conv_channel=self.conv_channel_32,
            conv_spatial=self.conv_spatial_32
        )

        ###################################################################
        # 8) 최종 출력 (MOS + movable)
        ###################################################################
        logits = self.logits3(up1e)                 # (bs, nclasses, 64, 2048)
        logits = F.softmax(logits, dim=1)

        range_up4e = self.range_upBlock1(Range_down3b, Range_down2b)
        range_up3e = self.range_upBlock2(range_up4e, Range_down1b)
        range_up2e = self.range_upBlock3(range_up3e, Range_down0b)
        movable_logits = self.movable_logits(range_up2e)  # (bs, movable_nclasses, 64, 2048)
        movable_logits = F.softmax(movable_logits, dim=1)

        return logits, movable_logits

##############################################################################
# (2) 실행 예시
##############################################################################
if __name__ == "__main__":
    # 임의 데이터
    bs = 4
    x  = torch.randn(bs, 13,64,2048)      # (range + residual)
    f1 = torch.randn(bs, 64, 313, 313)       # early shape
    f2 = torch.randn(bs, 128, 157, 157)        # mid shape
    f3 = torch.randn(bs, 256, 79, 79)      # late shape

    # dummy params
    params_dummy = {
        "train": {"batch_size": bs, "n_input_scans": 8},
        "dataset": {"sensor": {"img_prop": {"height":64, "width":2048}}}
    }
    model = MFMOS(nclasses=3, movable_nclasses=3, params=params_dummy, num_batch=bs)
    out1, out2 = model(x, f1, f2, f3)
    print("out1.shape:", out1.shape)
    print("out2.shape:", out2.shape)