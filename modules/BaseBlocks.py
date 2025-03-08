#!/usr/bin/env python3
# BaseBlocks.py
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


################### 공통 ###################
class FlattenCrossAttentionBlock(nn.Module):
    """
    Range(down5c)와 BEV(bev_down5c)가 서로 다른 (H×W)라도
    그대로 flatten하여 교차 어텐션을 수행하는 블록.
    - (B, C, H1, W1) vs. (B, C, H2, W2)
    - Q_A, K_A, V_A & Q_B, K_B, V_B 계산
    - A_out = A + softmax(Q_A x K_B^T) x V_B
      B_out = B + softmax(Q_B x K_A^T) x V_A
    - 이후 (B, C, H1, W1) & (B, C, H2, W2) 형태로 복원
      (즉, 해상도 H1×W1, H2×W2가 전혀 바뀌지 않음)
    """

    def __init__(self, channels):
        super(FlattenCrossAttentionBlock, self).__init__()
        self.channels = channels
        # Range를 위한 Q, K, V
        self.qA = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.kA = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.vA = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        # BEV를 위한 Q, K, V
        self.qB = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.kB = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.vB = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        self.scale = channels**0.5  # 채널 차원의 정규화 스케일

    def forward(self, featA, featB):
        """
        featA: [B, C, H1, W1], featB: [B, C, H2, W2]
        1) Q,K,V 얻기
        2) Flatten -> (B, C, N1), (B, C, N2)
        3) Cross Attention
        4) 원래 (H1, W1), (H2, W2)로 복원 + 합성
        """
        B, C, H1, W1 = featA.shape
        _, _, H2, W2 = featB.shape

        # 1) Q, K, V
        qA = self.qA(featA)  # [B, C, H1, W1]
        kA = self.kA(featA)
        vA = self.vA(featA)
        qB = self.qB(featB)
        kB = self.kB(featB)
        vB = self.vB(featB)

        # 2) Flatten (B, C, H*W) -> (B, C, N)
        #    연산 편의를 위해 (B, C, N) -> (B, N, C) 로도 변환
        def flatten_feat(x):
            # x: [B, C, H, W]
            B_, C_, H_, W_ = x.shape
            return x.view(B_, C_, -1)  # [B_, C_, H_*W_]

        qA_ = flatten_feat(qA)  # [B, C, N1]
        kA_ = flatten_feat(kA)
        vA_ = flatten_feat(vA)
        qB_ = flatten_feat(qB)
        kB_ = flatten_feat(kB)
        vB_ = flatten_feat(vB)

        # transpose(1,2): (B, C, N) -> (B, N, C)
        qA_t = qA_.transpose(1, 2)  # [B, N1, C]
        qB_t = qB_.transpose(1, 2)  # [B, N2, C]
        kA_t = kA_.transpose(1, 2)  # [B, N1, C]
        kB_t = kB_.transpose(1, 2)  # [B, N2, C]

        # 3) Cross Attention
        #    A_out = A + softmax(qA x kB^T / scale) x vB
        #    B_out = B + softmax(qB x kA^T / scale) x vA
        #
        #    qA_t: [B, N1, C], kB_t: [B, N2, C] -> qA_t × kB_t^T => [B, N1, N2]
        #    => softmax -> [B, N1, N2]
        #    => matmul vB_t: [B, N2, C] -> [B, N1, C]
        attnA = torch.bmm(qA_t, kB_t.transpose(1, 2))  # [B, N1, N2]
        attnA = attnA / self.scale
        attnA = F.softmax(attnA, dim=-1)  # [B, N1, N2]
        outA = torch.bmm(attnA, vB_.transpose(1, 2))  # [B, N1, C]
        outA = outA.transpose(1, 2).contiguous()  # [B, C, N1]

        # B_out
        attnB = torch.bmm(qB_t, kA_t.transpose(1, 2))  # [B, N2, N1]
        attnB = attnB / self.scale
        attnB = F.softmax(attnB, dim=-1)  # [B, N2, N1]
        outB = torch.bmm(attnB, vA_.transpose(1, 2))  # [B, N2, C]
        outB = outB.transpose(1, 2).contiguous()  # [B, C, N2]

        # 4) 원 해상도로 복원 + Residual로 합침
        A_out = featA + outA.view(B, C, H1, W1)
        B_out = featB + outB.view(B, C, H2, W2)
        return A_out, B_out


class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3, 3), dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        output = shortcut + resA2
        return output


class MetaKernel(nn.Module):
    def __init__(
        self,
        num_batch,
        feat_height,
        feat_width,
        coord_channels,
        fp16=False,
        num_frame=1,
    ):
        super(MetaKernel, self).__init__()
        self.num_batch = num_batch
        self.H = feat_height
        self.W = feat_width
        self.fp16 = fp16
        self.num_frame = num_frame
        self.channel_list = [16, 32]
        self.use_norm = False
        self.coord_channels = coord_channels

        # TODO: Best to put this part of the definition in __init__ function ?
        self.conv1 = nn.Conv2d(
            self.coord_channels,
            self.channel_list[0],
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=1,
            bias=True,
        )
        if self.use_norm:
            self.bn1 = nn.BatchNorm2d(self.channel_list[0])
        # self.act1 = nn.ReLU()
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(
            self.channel_list[0],
            self.channel_list[1],
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=1,
            bias=True,
        )
        self.conv1x1 = nn.Conv2d(
            in_channels=288, out_channels=32, kernel_size=1, stride=1, padding=0
        )

    def update_num_batch(self, num_batch):
        self.num_batch = num_batch

    @staticmethod
    def sampler_im2col(data, name=None, kernel=1, stride=1, pad=None, dilate=1):
        """please refer to mx.symbol.im2col"""
        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(dilate, int):
            dilate = (dilate, dilate)
        if pad is None:
            assert (
                kernel[0] % 2 == 1
            ), "Specify pad for an even kernel size in function sampler_im2col"
            pad = ((kernel[0] - 1) * dilate[0] + 1) // 2
        if isinstance(pad, int):
            pad = (pad, pad)

        # https://mxnet.apache.org/versions/1.8.0/api/python/docs/api/symbol/symbol.html#mxnet.symbol.im2col
        # output = mx.symbol.im2col(
        #     name=name + "sampler",
        #     data=data,
        #     kernel=kernel,
        #     stride=stride,
        #     dilate=dilate,
        #     pad=pad)

        # https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
        # torch._C._nn.im2col(input, _pair(kernel_size), _pair(dilation), _pair(padding), _pair(stride))
        output = F.unfold(
            data, kernel_size=kernel, dilation=dilate, stride=stride, padding=pad
        )

        return output

    def sample_data(self, data, kernel_size):
        """
        data sample
        :param data: num_batch, num_channel_in, H, W
        :param kernel_size: int default=3
        :return: sample_output: num_batch, num_channel_in * kernel_size * kernel_size, H, W
        """
        sample_output = self.sampler_im2col(
            data=data, kernel=kernel_size, stride=1, pad=1, dilate=1
        )
        return sample_output

    def sample_coord(self, coord, kernel_size):
        """
        coord sample
        :param coord: num_batch, num_channel_in, H, W
        :param kernel_size: int default=3
        :return: coord_sample_data: num_batch, num_channel_in * kernel_size * kernel_size, H, W
        """
        coord_sample_data = self.sampler_im2col(
            data=coord, kernel=kernel_size, stride=1, pad=1, dilate=1
        )

        return coord_sample_data

    def relative_coord(self, sample_coord, center_coord, num_channel_in, kernel_size):
        """
        :param sample_coord: num_batch, num_channel_in * kernel_size * kernel_size, H, W
        :param center_coord: num_batch, num_channel_in, H, W
        :param num_channel_in: int
        :param kernel_size: int
        :return: rel_coord: num_batch, num_channel_in, kernel_size * kernel_size, H, W
        """
        sample_reshape = torch.reshape(
            sample_coord,
            shape=(
                self.num_batch,
                num_channel_in,
                kernel_size * kernel_size,
                self.H,
                self.W,
            ),
        )

        center_coord_expand = torch.unsqueeze(center_coord, dim=2)  # expand_dims
        # ic(center_coord_expand.size())

        rel_coord = torch.subtract(sample_reshape, center_coord_expand)
        # ic(rel_coord.size())

        return rel_coord

    def mlp(self, data, in_channels, channel_list=None, b_mul=1, use_norm=False):
        """
        :param data: num_batch, num_channel_in * kernel_size * kernel_size, H, W
        :param in_channels: int
        :param norm: normalizer
        :param channel_list: List[int]
        :param b_mul: int default=1
        :param use_norm: bool default=False
        :return: mlp_output_reshape: num_batch, out_channels, kernel_size * kernel_size, H, W
        """
        assert isinstance(channel_list, list)
        # NOTE: Currently only supports two-layer Conv structure
        assert len(channel_list) == 2

        x = torch.reshape(data, shape=(self.num_batch * b_mul, in_channels, -1, self.W))

        y = self.conv1(x)
        # ic(y.size())

        if use_norm:
            y = self.bn1(y)
        y = self.act1(y)
        y = self.conv2(y)
        # ic(y.size())

        mlp_output_reshape = torch.reshape(
            y, shape=(self.num_batch * b_mul, channel_list[-1], -1, self.H, self.W)
        )
        # ic(mlp_output_reshape.size())
        return mlp_output_reshape

    def forward(self, data, coord_data, data_channels, coord_channels, kernel_size=3):
        """
        # Without data mlp;
        # MLP: fc + norm + relu + fc;
        # Using normalized coordinates
        :param data: num_batch, num_channel_in, H, W
        :param coord_data: num_batch, 3, H, W
        :param data_channels: num_channel_in
        :param coord_channels: 3
        :param norm: normalizer
        :param conv1_filter: int
        :param kernel_size: int default=3
        :param kwargs:
        :return: conv1: num_batch, conv1_filter, H, W
        """

        coord_sample_data = self.sample_coord(
            coord_data, kernel_size
        )  # (1, 45, 131072)
        # ic(coord_sample_data.size())
        # ic(coord_data.size())

        rel_coord = self.relative_coord(  # (1, 5, 9, 64, 2048)
            coord_sample_data, coord_data, coord_channels, kernel_size
        )
        # ic(rel_coord.size())

        weights = self.mlp(  # (1, 32, 9, 64, 2048)
            rel_coord, in_channels=coord_channels, channel_list=self.channel_list
        )
        # ic(weights.size())

        data_sample = self.sample_data(data, kernel_size)  # (1, 288, 131072)
        # ic(data_sample.size())

        data_sample_reshape = torch.reshape(  # (1, 32, 9, 64, 2048)
            data_sample,
            shape=(
                self.num_batch,
                data_channels,
                kernel_size * kernel_size,
                self.H,
                self.W,
            ),
        )
        # ic(data_sample_reshape.size())

        output = data_sample_reshape * weights  # (1, 32, 9, 64, 2048)
        # ic(output.size())

        output_reshape = torch.reshape(  # (1, 288, 64, 2048)
            output, shape=(self.num_batch, -1, self.H, self.W)
        )
        # ic(output_reshape.size())

        output = self.conv1x1(output_reshape)  # (1, 32, 64, 2048)
        return output


################### For Range View ###################


@torch.jit.script
def pixelshuffle(x: torch.Tensor, factor_hw: Tuple[int, int]):
    pH = factor_hw[0]
    pW = factor_hw[1]
    y = x
    B, iC, iH, iW = y.shape
    oC, oH, oW = iC // (pH * pW), iH * pH, iW * pW
    y = y.reshape(B, oC, pH, pW, iH, iW)
    y = y.permute(0, 1, 4, 2, 5, 3)  # B, oC, iH, pH, iW, pW
    y = y.reshape(B, oC, oH, oW)
    return y


class ResBlock(nn.Module):
    def __init__(
        self,
        in_filters,
        out_filters,
        dropout_rate,
        kernel_size=(3, 3),
        stride=1,
        pooling=True,
        drop_out=True,
        use_softpool=False,
    ):
        super(ResBlock, self).__init__()
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
            if use_softpool:
                self.pool = SoftPool2d(kernel_size=(2, 2), stride=(2, 2))
            else:
                # self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
                self.pool = nn.AvgPool2d(kernel_size=kernel_size)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        shortcut = self.conv1(x)  # (3, 64, 64, 2048)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)  # (3, 64, 64, 2048)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        resA = self.conv4(resA2)  # (3, 64, 64, 2048)
        resA = self.act4(resA)
        resA3 = self.bn3(resA)

        concat = torch.cat((resA1, resA2, resA3), dim=1)  # (3, 192, 64, 2048)
        resA = self.conv5(concat)  # (3, 64, 64, 2048)
        resA = self.act5(resA)
        resA = self.bn4(resA)
        resA = shortcut + resA  # (3, 64, 64, 2048)

        if self.pooling:
            resB = self.dropout(resA) if self.drop_out else resA
            resB = self.pool(resB)

            # H, W = resB.shape[2:]
            # resB = resB.flatten(2).transpose(1, 2)
            # return resB, H, W
            return resB, resA
        else:
            resB = self.dropout(resA) if self.drop_out else resA
            return resB


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)

        # self.conv1 = nn.Conv2d(in_filters // 4 + 2*out_filters, out_filters, (3, 3), padding=1)
        self.conv1 = nn.Conv2d(
            in_filters // 8 + 2 * out_filters, out_filters, (3, 3), padding=1
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

    def forward(self, x, skip):
        # upA = nn.PixelShuffle(2)(x)
        upA = pixelshuffle(x, (2, 4))
        if self.drop_out:
            upA = self.dropout1(upA)

        upB = torch.cat((upA, skip), dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)

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


################### For Bird Eye View ###################


def pixelshuffle2x(x: torch.Tensor):
    """
    (2,2) pixelshuffle: 입력 x의 공간 크기를 2배씩 확장.
    x: [B, C, H, W] → 출력: [B, C//4, 2H, 2W]
    """
    B, C, H, W = x.shape
    p = 2
    oC = C // (p * p)
    oH, oW = H * p, W * p
    y = x.reshape(B, oC, p, p, H, W)
    y = y.permute(0, 1, 4, 2, 5, 3).reshape(B, oC, oH, oW)
    return y


class UpBlockBEV(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True):
        super(UpBlockBEV, self).__init__()
        self.drop_out = drop_out

        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)

        # BEV branch: pixelshuffle factor (2,2) → 채널 감소: in_filters//4
        # skip 채널은 보통 2*out_filters (동일하게 유지)
        self.conv1 = nn.Conv2d(
            in_filters // 4 + 2 * out_filters, out_filters, kernel_size=3, padding=1
        )
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(
            out_filters, out_filters, kernel_size=3, dilation=2, padding=2
        )
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(
            out_filters, out_filters, kernel_size=2, dilation=2, padding=1
        )
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters * 3, out_filters, kernel_size=1)
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

    def forward(self, x, skip):
        upA = pixelshuffle2x(x)  # [B, in_filters//4, 2H, 2W]
        if self.drop_out:
            upA = self.dropout1(upA)
        # 만약 skip의 공간 크기가 다르다면 보간하여 맞춰줌
        if upA.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(
                skip, size=upA.shape[2:], mode="bilinear", align_corners=False
            )
        upB = torch.cat((upA, skip), dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)
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
    BEV branch용 ResBlock.
    다운샘플링 시 kernel_size=(2,2)를 사용하여, 해상도를 균일하게 절반씩 줄임.
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

        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=1, stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(
            out_filters, out_filters, kernel_size=3, dilation=2, padding=2
        )
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(
            out_filters, out_filters, kernel_size=2, dilation=2, padding=1
        )
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters * 3, out_filters, kernel_size=1)
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)
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
