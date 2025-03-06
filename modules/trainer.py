#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
수정된 trainer.py 파일 (새로운 MFMOS.py 모델에 호환)
Range 관련 변수는 range_ 접두사, BEV 관련 변수는 bev_ 접두사를 사용합니다.
손실(loss)은 range_moving, range_movable, bev_moving, bev_movable 네 손실의 합으로 정의하며,
각 branch별로 별도의 AverageMeter 및 IoU evaluator를 구성하여 metric들을 산출합니다.
"""

# GPU 개수에 따라 sampler, dist.barrier() 주석하거나 주석해제 해야함.

import datetime
import os
import time
import imp
import cv2
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import __init__ as booger

import torch.optim as optim
from tensorboardX import SummaryWriter as Logger
from common.sync_batchnorm.batchnorm import convert_model
from common.warmupLR import warmupLR

from modules.MFMOS import MFMOS
from modules.loss.Lovasz_Softmax import Lovasz_softmax_PointCloud
from modules.tools import (
    AverageMeter,
    iouEval,
    save_checkpoint,
    save_to_txtlog,
    make_log_img,
)

from torch import distributed as dist
from torch.cuda.amp import autocast, GradScaler


def unproject(points, preds, proj_x, proj_y):
    device = preds.device
    batch_indices_list = []
    xs_list = []
    ys_list = []

    # 각 배치의 유효 점들을 리스트로 모읍니다.
    for b in range(len(points)):
        n_points = points[b].size(0)
        batch_indices_list.append(
            torch.full((n_points,), b, dtype=torch.long, device=device)
        )
        xs_list.append(proj_x[b][:n_points])
        ys_list.append(proj_y[b][:n_points])

    # 배치 인덱스, x, y 좌표를 모두 concatenate하여 1차원 텐서로 만듭니다.
    batch_indices = torch.cat(batch_indices_list, dim=0)  # (total_points,)
    xs = torch.cat(xs_list, dim=0)  # (total_points,)
    ys = torch.cat(ys_list, dim=0)  # (total_points,)

    # 유효하지 않은 좌표(-1) 마스크 생성
    valid_mask = (xs != -1) & (ys != -1)
    # 유효 좌표는 clamp하여 유효 범위 내로 제한합니다.
    xs_clamped = xs.clamp(0, preds.size(3) - 1)
    ys_clamped = ys.clamp(0, preds.size(2) - 1)

    # advanced indexing으로 각 배치의 해당 좌표 픽셀 값을 추출합니다.
    # preds의 shape: (B, C, H, W) → 결과 gathered: (total_points, C)
    gathered = preds[batch_indices, :, ys_clamped, xs_clamped]

    # 유효하지 않은 좌표는 0으로 처리합니다.
    gathered[~valid_mask] = 0

    # 최종 shape: (1, C, total_points)
    return gathered.T.unsqueeze(0)


def loss_and_to_labels(probs, log_probs, labels, each_criterion, ls_func):
    """배치 전체에 대해서 평균 loss가 나옴"""

    if probs.numel() == 0:
        raise ValueError("No unprojected predictions found")

    jacc = ls_func(probs, labels)
    wce = each_criterion(log_probs.double(), labels).float()
    loss = wce + jacc
    # print("jacc, wce, loss:", jacc.item(), wce.item(), loss.item())

    argmax = probs.argmax(dim=1)
    return loss, argmax


class Trainer:
    def __init__(
        self, ARCH, DATA, datadir, logdir, path=None, point_refine=False, local_rank=0
    ):
        # 기본 변수 설정
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.logdir = logdir
        self.path = path
        self.epoch = 0
        self.point_refine = point_refine
        self.local_rank = local_rank

        # 시간 및 성능 측정 AverageMeter (batch, data 등)
        self.batch_time_t = AverageMeter()
        self.data_time_t = AverageMeter()
        self.batch_time_e = AverageMeter()

        # Tensorboard logger
        self.tb_logger = Logger(os.path.join(self.logdir, "tb"))
        self.info = {
            "train_update": 0,
            "train_loss": 0,
            "train_range_acc": 0,
            "train_range_iou": 0,
            "train_bev_acc": 0,
            "train_bev_iou": 0,
            "valid_loss": 0,
            "valid_range_acc": 0,
            "valid_range_iou": 0,
            "valid_bev_acc": 0,
            "valid_bev_iou": 0,
            "best_train_range_iou": 0,
            "best_train_bev_iou": 0,
            "best_val_range_iou": 0,
            "best_val_bev_iou": 0,
        }

        # parser 및 데이터 로더 설정 (원본 parser 코드 사용)
        parserModule = imp.load_source(
            "parserModule",
            f"{booger.TRAIN_PATH}/common/dataset/{self.DATA['name']}/parser.py",
        )
        self.parser = parserModule.Parser(
            root=self.datadir,
            train_sequences=self.DATA["split"]["train"],
            valid_sequences=self.DATA["split"]["valid"],
            test_sequences=None,
            split="train",
            labels=self.DATA["labels"],
            residual_aug=self.ARCH["train"]["residual_aug"],
            valid_residual_delta_t=self.ARCH["train"]["valid_residual_delta_t"],
            color_map=self.DATA["color_map"],
            learning_map=self.DATA["moving_learning_map"],
            movable_learning_map=self.DATA["movable_learning_map"],
            learning_map_inv=self.DATA["moving_learning_map_inv"],
            movable_learning_map_inv=self.DATA["movable_learning_map_inv"],
            sensor=self.ARCH["dataset"]["sensor"],
            max_points=self.ARCH["dataset"]["max_points"],
            batch_size=self.ARCH["train"]["batch_size"],
            workers=self.ARCH["train"]["workers"],
            gt=True,
            shuffle_train=True,
        )

        self.set_loss_weight()

        # 모델 생성 (torch.no_grad() 내에서 초기화)
        with torch.no_grad():
            self.model = MFMOS(
                nclasses=self.parser.get_n_classes(),
                movable_nclasses=self.parser.get_n_classes(movable=True),
                params=self.ARCH,
            )

        self.set_gpu_cuda()
        self.set_loss_function(point_refine)
        self.set_optim_scheduler()

        # 모델 로드 (checkpoint)
        if self.path is not None:
            self.load_pretrained_model()

    def set_loss_weight(self):
        """
        각 클래스별 가중치 계산
        """
        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
        movable_content = torch.zeros(
            self.parser.get_n_classes(movable=True), dtype=torch.float
        )
        for cl, freq in self.DATA["content"].items():
            x_cl = self.parser.to_xentropy(cl)  # 실제 클래스를 xentropy class로 매핑
            content[x_cl] += freq

            movable_x_cl = self.parser.to_xentropy(cl, movable=True)
            movable_content[movable_x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)
        self.movable_loss_w = 1 / (movable_content + epsilon_w)
        for x_cl, w in enumerate(self.loss_w):
            if self.DATA["learning_ignore"][x_cl]:
                self.loss_w[x_cl] = 0
        for x_cl, w in enumerate(self.movable_loss_w):
            if self.DATA["learning_ignore"][x_cl]:
                self.movable_loss_w[x_cl] = 0
        print("Loss weights from content: ", self.loss_w.data)
        print("Movable Loss weights from content: ", self.movable_loss_w.data)

    def set_loss_function(self, point_refine):
        """
        손실함수 정의 (필요 시 Lovasz Softmax 사용)
        """
        self.criterion = nn.NLLLoss(ignore_index=0, weight=self.loss_w.double()).cuda()
        self.movable_criterion = nn.NLLLoss(
            ignore_index=0, weight=self.movable_loss_w.double()
        ).cuda()
        self.ls = Lovasz_softmax_PointCloud().to(self.device)
        # if not point_refine:
        #     self.ls = Lovasz_softmax(ignore=0).cuda()
        #     self.movable_ls = Lovasz_softmax(ignore=0).cuda()
        # else:
        #     self.ls = Lovasz_softmax_PointCloud(ignore=0).to(self.device)

    def set_gpu_cuda(self):
        """
        GPU 및 CUDA 설정
        """
        self.gpu = False
        self.multi_gpu = False
        self.n_gpus = 0
        self.model_single = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Training on device: ", self.device)

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            self.gpu = True
            self.n_gpus = 1
            self.model.cuda()

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = convert_model(self.model).cuda()
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
                broadcast_buffers=False,
            )
            self.model_single = self.model.module
            self.multi_gpu = True
            self.n_gpus = torch.cuda.device_count()

    def set_optim_scheduler(self):
        """
        Optimizer 및 Scheduler 설정
        """
        self.optimizer = optim.SGD(
            [{"params": self.model.parameters()}],
            lr=self.ARCH["train"]["lr"],
            momentum=self.ARCH["train"]["momentum"],
            weight_decay=self.ARCH["train"]["w_decay"],
        )
        steps_per_epoch = self.parser.get_train_size()
        up_steps = int(self.ARCH["train"]["wup_epochs"] * steps_per_epoch)
        final_decay = self.ARCH["train"]["lr_decay"] ** (1 / steps_per_epoch)
        self.scheduler = warmupLR(
            optimizer=self.optimizer,
            lr=self.ARCH["train"]["lr"],
            warmup_steps=up_steps,
            momentum=self.ARCH["train"]["momentum"],
            decay=final_decay,
        )

    def load_pretrained_model(self):
        """
        체크포인트에서 모델 로드
        """
        torch.nn.Module.dump_patches = True
        checkpoint = "MFMOS" if not self.point_refine else "MFMOS_valid_best"
        w_dict = torch.load(
            os.path.join(self.path, checkpoint),
            map_location=lambda storage, loc: storage,
        )
        if not self.point_refine:
            self.model.load_state_dict(w_dict["state_dict"], strict=True)
        else:
            self.model.load_state_dict(
                {k.replace("module.", ""): v for k, v in w_dict["state_dict"].items()}
            )
        self.optimizer.load_state_dict(w_dict["optimizer"])
        self.epoch = w_dict["epoch"] + 1
        self.scheduler.load_state_dict(w_dict["scheduler"])
        print("Loaded pretrained model from", checkpoint)

    def calculate_estimate(self, epoch, iter):
        estimate = int(
            (self.data_time_t.avg + self.batch_time_t.avg)
            * (
                self.parser.get_train_size() * self.ARCH["train"]["max_epochs"]
                - (iter + 1 + epoch * self.parser.get_train_size())
            )
        ) + int(
            self.batch_time_e.avg
            * self.parser.get_valid_size()
            * (self.ARCH["train"]["max_epochs"] - epoch)
        )
        return str(datetime.timedelta(seconds=estimate))

    @staticmethod
    def save_to_tensorboard(
        logdir,
        logger,
        info,
        epoch,
        w_summary=False,
        model=None,
        img_summary=False,
        imgs=[],
    ):
        for tag, value in info.items():
            if "valid_classes" in tag:
                continue
            logger.add_scalar(tag, value, epoch)
        if w_summary and model:
            for tag, value in model.named_parameters():
                tag = tag.replace(".", "/")
                logger.add_histogram(tag, value.data.cpu().numpy(), epoch)
                if value.grad is not None:
                    logger.add_histogram(
                        tag + "/grad", value.grad.data.cpu().numpy(), epoch
                    )
        if img_summary and len(imgs) > 0:
            directory = os.path.join(logdir, "predictions")
            if not os.path.isdir(directory):
                os.makedirs(directory)
            for i, img in enumerate(imgs):
                name = os.path.join(directory, f"{i}.png")
                cv2.imwrite(name, img)

    def init_evaluator(self):
        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
                print("Ignoring class", i, "in IoU evaluation (range)")
        self.range_evaluator = iouEval(
            self.parser.get_n_classes(), self.local_rank, self.ignore_class
        )
        self.range_movable_evaluator = iouEval(
            self.parser.get_n_classes(movable=True), self.local_rank, self.ignore_class
        )

        # BEV evaluator (범위에 상관없이 별도로 구성)
        self.bev_evaluator = iouEval(
            self.parser.get_n_classes(), self.local_rank, self.ignore_class
        )
        self.bev_movable_evaluator = iouEval(
            self.parser.get_n_classes(movable=True), self.local_rank, self.ignore_class
        )

        self.final_evaluator = iouEval(
            self.parser.get_n_classes(), self.local_rank, self.ignore_class
        )

    def train(self):
        self.init_evaluator()
        for epoch in range(self.epoch, self.ARCH["train"]["max_epochs"]):
            # self.parser.train_sampler.set_epoch(epoch)
            range_acc, range_iou, bev_acc, bev_iou, loss_total, update_mean = (
                self.train_epoch(
                    train_loader=self.parser.get_train_set(),
                    model=self.model,
                    all_criterion=(self.criterion, self.movable_criterion),
                    optimizer=self.optimizer,
                    epoch=epoch,
                    all_evaluator=(
                        self.range_evaluator,
                        self.range_movable_evaluator,
                        self.bev_evaluator,
                        self.bev_movable_evaluator,
                        self.final_evaluator,
                    ),
                    scheduler=self.scheduler,
                    report=self.ARCH["train"]["report_batch"],
                )
            )
            if self.local_rank == 0:
                self.update_training_info(
                    epoch,
                    range_acc,
                    range_iou,
                    bev_acc,
                    bev_iou,
                    loss_total,
                    update_mean,
                )
            if epoch % self.ARCH["train"]["report_epoch"] == 0 and self.local_rank == 0:
                (
                    range_acc_val,
                    range_iou_val,
                    bev_acc_val,
                    bev_iou_val,
                    loss_val,
                    rand_img,
                ) = self.validate(
                    val_loader=self.parser.get_valid_set(),
                    model=self.model,
                    all_criterion=(self.criterion, self.movable_criterion),
                    all_evaluator=(
                        self.range_evaluator,
                        self.range_movable_evaluator,
                        self.bev_evaluator,
                        self.bev_movable_evaluator,
                        self.final_evaluator,
                    ),
                    class_func=self.parser.get_xentropy_class_string,
                    color_fn=self.parser.to_color,
                    save_scans=self.ARCH["train"]["save_scans"],
                )
                self.update_validation_info(
                    epoch,
                    range_acc_val,
                    range_iou_val,
                    bev_acc_val,
                    bev_iou_val,
                    loss_val,
                )
            if self.local_rank == 0:
                Trainer.save_to_tensorboard(
                    logdir=self.logdir,
                    logger=self.tb_logger,
                    info=self.info,
                    epoch=epoch,
                    w_summary=self.ARCH["train"]["save_summary"],
                    model=self.model_single,
                    img_summary=self.ARCH["train"]["save_scans"],
                    imgs=rand_img,
                )
            # dist.barrier()
        print("Finished Training")
        return

    def train_epoch(
        self,
        train_loader,
        model,
        all_criterion,
        optimizer,
        epoch,
        all_evaluator,
        scheduler,
        report=10,
    ):
        # AverageMeter 초기화
        range_loss_meter = AverageMeter()
        bev_loss_meter = AverageMeter()
        final_loss_meter = AverageMeter()
        range_acc_meter = AverageMeter()
        range_iou_meter = AverageMeter()
        bev_acc_meter = AverageMeter()
        bev_iou_meter = AverageMeter()
        final_acc_meter = AverageMeter()
        final_iou_meter = AverageMeter()
        update_ratio_meter = AverageMeter()

        (
            range_evaluator,
            range_movable_evaluator,
            bev_evaluator,
            bev_movable_evaluator,
            final_evaluator,
        ) = all_evaluator
        criterion, movable_criterion = all_criterion

        model.train()
        end = time.time()

        scaler = torch.cuda.amp.GradScaler()
        for i, (
            # (
            #     points,
            #     (GTs_moving, GTs_movable),
            # ),  # (B, n_points, 3), (B, n_points) 원본 점 및 원본 라벨
            # (proj_full, bev_full),  # range: (B, 13, H, W), bev: (B, 13, H_bev, W_bev)
            # (proj_labels, proj_movable_labels),  # range: (B, H, W), (B, H, W)
            # (
            #     bev_labels,
            #     bev_movable_labels,
            # ),  # bev: (B, H_bev, W_bev), (B, H_bev, W_bev)
            # (path_seq, path_name, unproj_n_points),  # e.g., '08', '000123.npy', 122319
            # (proj_x, proj_y),  # (B, 150000), (B, 150000)
            # (bev_proj_x, bev_proj_y),  # (B, 150000), (B, 150000)
            # (proj_xyz, bev_xyz),  # (B, H, W, 3), (B, H_bev, W_bev, 3)
            # (proj_range, bev_range),  # (B, H, W), (B, H_bev, W_bev)
            # (unproj_range, bev_unproj_range),  # (B, 150000), (B, 150000)
            (points, (GTs_moving, GTs_movable)),
            (proj_full, bev_full),
            (proj_labels, proj_movable_labels),
            (bev_labels, bev_movable_labels),
            (proj_x, proj_y),
            (bev_proj_x, bev_proj_y),
        ) in enumerate(train_loader):
            self.data_time_t.update(time.time() - end)
            if self.gpu:
                points = points.cuda(non_blocking=True)
                proj_full = proj_full.cuda(non_blocking=True)
                bev_full = bev_full.cuda(non_blocking=True)
                proj_labels = proj_labels.cuda(non_blocking=True).long()
                proj_movable_labels = proj_movable_labels.cuda(non_blocking=True).long()
                bev_labels = bev_labels.cuda(non_blocking=True).long()
                bev_movable_labels = bev_movable_labels.cuda(non_blocking=True).long()
                proj_x = proj_x.cuda(non_blocking=True)
                proj_y = proj_y.cuda(non_blocking=True)
                bev_proj_x = bev_proj_x.cuda(non_blocking=True)
                bev_proj_y = bev_proj_y.cuda(non_blocking=True)
                # proj_xyz = proj_xyz.cuda(non_blocking=True)
                # bev_xyz = bev_xyz.cuda(non_blocking=True)
                # proj_range = proj_range.cuda(non_blocking=True)
                # bev_range = bev_range.cuda(non_blocking=True)
                # unproj_range = unproj_range.cuda(non_blocking=True)
                # bev_unproj_range = bev_unproj_range.cuda(non_blocking=True)
                GTs_moving = GTs_moving.cuda(non_blocking=True)
                GTs_movable = GTs_movable.cuda(non_blocking=True)

            r_moving, r_movable, b_moving, b_movable = model(proj_full, bev_full)

            r_moving_probs = unproject(points, r_moving, proj_x, proj_y)
            r_movable_probs = unproject(points, r_movable, proj_x, proj_y)
            b_moving_probs = unproject(points, b_moving, bev_proj_x, bev_proj_y)
            b_movable_probs = unproject(points, b_movable, bev_proj_x, bev_proj_y)
            final_probs = (r_moving_probs * b_moving_probs) / 2.0

            log_r_moving_probs = torch.log(r_moving_probs.clamp(min=1e-8))
            log_r_movable_probs = torch.log(r_movable_probs.clamp(min=1e-8))
            log_b_moving_probs = torch.log(b_moving_probs.clamp(min=1e-8))
            log_b_movable_probs = torch.log(b_movable_probs.clamp(min=1e-8))
            log_final_probs = torch.log(final_probs.clamp(min=1e-8))

            r_moving_losses, r_moving_preds = loss_and_to_labels(
                r_moving_probs,
                log_r_moving_probs,
                GTs_moving,
                criterion,
                self.ls,
            )

            r_movable_losses, r_movable_preds = loss_and_to_labels(
                r_movable_probs,
                log_r_movable_probs,
                GTs_movable,
                movable_criterion,
                self.ls,
            )

            b_moving_losses, b_moving_preds = loss_and_to_labels(
                b_moving_probs,
                log_b_moving_probs,
                GTs_moving,
                criterion,
                self.ls,
            )

            b_movable_losses, b_movable_preds = loss_and_to_labels(
                b_movable_probs,
                log_b_movable_probs,
                GTs_movable,
                movable_criterion,
                self.ls,
            )

            final_losses, final_preds = loss_and_to_labels(
                final_probs,
                log_final_probs,
                GTs_moving,
                criterion,
                self.ls,
            )

            range_losses = r_moving_losses + r_movable_losses
            bev_losses = b_moving_losses + b_movable_losses
            loss_m = range_losses + bev_losses + final_losses

            # optimizer.zero_grad()
            # loss_m.backward()
            # optimizer.step()
            optimizer.zero_grad()
            loss = loss_m.mean()  # 평균 손실 계산
            # 자동 혼합 정밀도 컨텍스트 없이, 스케일러를 이용해 역전파 수행
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                range_evaluator.reset()
                range_evaluator.addBatch(r_moving_preds, GTs_moving)
                range_acc = range_evaluator.getacc()
                range_jaccard, _ = range_evaluator.getIoU()

                range_movable_evaluator.reset()
                range_movable_evaluator.addBatch(r_movable_preds, GTs_movable)
                range_movable_acc = range_movable_evaluator.getacc()
                range_movable_jaccard, _ = range_movable_evaluator.getIoU()

                # BEV 평가: b_moving과 b_movable 결과 평가 후 평균 처리
                bev_evaluator.reset()
                bev_evaluator.addBatch(b_moving_preds, GTs_moving)
                bev_acc = bev_evaluator.getacc()
                bev_jaccard, _ = bev_evaluator.getIoU()

                bev_movable_evaluator.reset()
                bev_movable_evaluator.addBatch(b_movable_preds, GTs_movable)
                bev_movable_acc = bev_movable_evaluator.getacc()
                bev_movable_jaccard, _ = bev_movable_evaluator.getIoU()

                # Final 평가: final_preds와 GTs_moving 비교
                final_evaluator.reset()
                final_evaluator.addBatch(final_preds, GTs_moving)
                final_acc = final_evaluator.getacc()
                final_jaccard, _ = final_evaluator.getIoU()

            # 배치별 손실 및 평가 지표 미터 업데이트 (배치 크기: points.size(0))
            range_loss_meter.update(range_losses.item(), points.size(0))
            bev_loss_meter.update(bev_losses.item(), points.size(0))
            final_loss_meter.update(final_losses.item(), points.size(0))

            # moving과 movable 결과를 평균 내어 평가 (Range와 BEV 각각)
            avg_range_acc = (range_acc + range_movable_acc) / 2.0
            avg_range_iou = (range_jaccard + range_movable_jaccard) / 2.0
            avg_bev_acc = (bev_acc + bev_movable_acc) / 2.0
            avg_bev_iou = (bev_jaccard + bev_movable_jaccard) / 2.0

            range_acc_meter.update(avg_range_acc, points.size(0))
            range_iou_meter.update(avg_range_iou, points.size(0))
            bev_acc_meter.update(avg_bev_acc, points.size(0))
            bev_iou_meter.update(avg_bev_iou, points.size(0))
            final_acc_meter.update(final_acc, points.size(0))
            final_iou_meter.update(final_jaccard, points.size(0))

            # 업데이트 비율 계산
            # update_ratios = []
            # for g in optimizer.param_groups:
            #     lr = g["lr"]
            #     for param in g["params"]:
            #         if param.grad is not None:
            #             w = np.linalg.norm(param.data.cpu().numpy().reshape(-1))
            #             update = np.linalg.norm(
            #                 (-max(lr, 1e-10) * param.grad.cpu().numpy().reshape(-1))
            #             )
            #             update_ratios.append(update / max(w, 1e-10))
            # update_ratios = np.array(update_ratios)
            # update_mean = update_ratios.mean()
            # update_ratio_meter.update(update_mean)
            # 개선된 업데이트 비율 계산 (GPU 내장 함수 사용)
            ratios = []
            for group in optimizer.param_groups:
                lr = group["lr"]
                for param in group["params"]:
                    if param.grad is not None:
                        # 파라미터와 gradient의 L2 norm을 GPU에서 직접 계산합니다.
                        w = param.data.norm(p=2)
                        grad_norm = (lr * param.grad).norm(p=2)
                        ratios.append(grad_norm / (w + 1e-10))
            if ratios:
                update_mean = torch.stack(ratios).mean().item()
            else:
                raise Exception("No gradients found in optimizer parameters.")
                update_mean = 0.0
            update_ratio_meter.update(update_mean)

            # 일정 배치마다 로그 출력
            if i % report == 0:
                remaining_time = self.calculate_estimate(epoch, i)
                log_str = (
                    f"Epoch: [{epoch}][{i}/{len(train_loader)}] | "
                    f"Range Loss: {range_loss_meter.val:.4f} ({range_loss_meter.avg:.4f}) | "
                    f"BEV Loss: {bev_loss_meter.val:.4f} ({bev_loss_meter.avg:.4f}) | "
                    f"Final Loss: {final_loss_meter.val:.4f} ({final_loss_meter.avg:.4f}) | "
                    # f"Range Acc: {range_acc_meter.val:.3f} ({range_acc_meter.avg:.3f}) | "
                    f"Range IoU: {range_iou_meter.val:.3f} ({range_iou_meter.avg:.3f}) | "
                    # f"BEV Acc: {bev_acc_meter.val:.3f} ({bev_acc_meter.avg:.3f}) | "
                    f"BEV IoU: {bev_iou_meter.val:.3f} ({bev_iou_meter.avg:.3f}) | "
                    # f"Final Acc: {final_acc_meter.val:.3f} ({final_acc_meter.avg:.3f}) | "
                    f"Final IoU: {final_iou_meter.val:.3f} ({final_iou_meter.avg:.3f}) | "
                    # f"Update Ratio: {update_mean:.3e} | "
                    f"[{remaining_time}]"
                )
                print(log_str)
                save_to_txtlog(self.logdir, "log.txt", log_str)

            # 스케줄러 업데이트 (배치마다 step 호출)
            scheduler.step()

            # 배치 처리 시간 갱신 추가
            batch_time = time.time() - end
            # print("batch time:", batch_time)
            self.batch_time_t.update(batch_time)
            end = time.time()

    def validate(
        self,
        val_loader,
        model,
        all_criterion,
        all_evaluator,
        class_func,
        color_fn,
        save_scans=False,
    ):
        # 평가용 평균 측정기 초기화
        range_loss_meter = AverageMeter()
        bev_loss_meter = AverageMeter()
        final_loss_meter = AverageMeter()
        range_acc_meter = AverageMeter()
        range_iou_meter = AverageMeter()
        bev_acc_meter = AverageMeter()
        bev_iou_meter = AverageMeter()
        final_acc_meter = AverageMeter()
        final_iou_meter = AverageMeter()
        hetero_l = AverageMeter()  # 추후 필요시 사용
        rand_imgs = []

        # evaluators와 criterion unpack
        (
            range_evaluator,
            range_movable_evaluator,
            bev_evaluator,
            bev_movable_evaluator,
            final_evaluator,
        ) = all_evaluator
        criterion, movable_criterion = all_criterion

        # 평가 모드 전환 및 evaluator 초기화
        model.eval()
        range_evaluator.reset()
        range_movable_evaluator.reset()
        bev_evaluator.reset()
        bev_movable_evaluator.reset()
        final_evaluator.reset()

        with torch.no_grad():
            end = time.time()
            for i, (
                # (
                #     points,
                #     (GTs_moving, GTs_movable),
                # ),  # (B, n_points, 3), (B, n_points) 원본 점 및 원본 라벨
                # (
                #     proj_full,
                #     bev_full,
                # ),  # range: (B, 13, H, W), bev: (B, 13, H_bev, W_bev)
                # (proj_labels, proj_movable_labels),  # range: (B, H, W), (B, H, W)
                # (
                #     bev_labels,
                #     bev_movable_labels,
                # ),  # bev: (B, H_bev, W_bev), (B, H_bev, W_bev)
                # (
                #     path_seq,
                #     path_name,
                #     unproj_n_points,
                # ),  # e.g., '08', '000123.npy', 122319
                # (proj_x, proj_y),  # (B, 150000), (B, 150000)
                # (bev_proj_x, bev_proj_y),  # (B, 150000), (B, 150000)
                # (proj_xyz, bev_xyz),  # (B, H, W, 3), (B, H_bev, W_bev, 3)
                # (proj_range, bev_range),  # (B, H, W), (B, H_bev, W_bev)
                # (unproj_range, bev_unproj_range),  # (B, 150000), (B, 150000)
                (points, (GTs_moving, GTs_movable)),
                (proj_full, bev_full),
                (proj_labels, proj_movable_labels),
                (bev_labels, bev_movable_labels),
                (proj_x, proj_y),
                (bev_proj_x, bev_proj_y),
            ) in enumerate(tqdm(val_loader, desc="Validation:", ncols=80)):

                # GPU로 데이터 이동
                if self.gpu:
                    points = points.cuda(non_blocking=True)
                    proj_full = proj_full.cuda(non_blocking=True)
                    bev_full = bev_full.cuda(non_blocking=True)
                    proj_labels = proj_labels.cuda(non_blocking=True).long()
                    proj_movable_labels = proj_movable_labels.cuda(
                        non_blocking=True
                    ).long()
                    bev_labels = bev_labels.cuda(non_blocking=True).long()
                    bev_movable_labels = bev_movable_labels.cuda(
                        non_blocking=True
                    ).long()
                    proj_x = proj_x.cuda(non_blocking=True)
                    proj_y = proj_y.cuda(non_blocking=True)
                    bev_proj_x = bev_proj_x.cuda(non_blocking=True)
                    bev_proj_y = bev_proj_y.cuda(non_blocking=True)
                    # proj_xyz = proj_xyz.cuda(non_blocking=True)
                    # bev_xyz = bev_xyz.cuda(non_blocking=True)
                    # proj_range = proj_range.cuda(non_blocking=True)
                    # bev_range = bev_range.cuda(non_blocking=True)
                    # unproj_range = unproj_range.cuda(non_blocking=True)
                    # bev_unproj_range = bev_unproj_range.cuda(non_blocking=True)
                    GTs_moving = GTs_moving.cuda(non_blocking=True)
                    GTs_movable = GTs_movable.cuda(non_blocking=True)

                # 모델 추론 (range, bev 각 branch의 결과 반환)
                r_moving, r_movable, b_moving, b_movable = model(proj_full, bev_full)

                r_moving_probs = unproject(points, r_moving, proj_x, proj_y)
                r_movable_probs = unproject(points, r_movable, proj_x, proj_y)
                b_moving_probs = unproject(points, b_moving, bev_proj_x, bev_proj_y)
                b_movable_probs = unproject(points, b_movable, bev_proj_x, bev_proj_y)
                final_probs = (r_moving_probs * b_moving_probs) / 2.0

                log_r_moving_probs = torch.log(r_moving_probs.clamp(min=1e-8))
                log_r_movable_probs = torch.log(r_movable_probs.clamp(min=1e-8))
                log_b_moving_probs = torch.log(b_moving_probs.clamp(min=1e-8))
                log_b_movable_probs = torch.log(b_movable_probs.clamp(min=1e-8))
                log_final_probs = torch.log(final_probs.clamp(min=1e-8))

                r_moving_losses, r_moving_preds = loss_and_to_labels(
                    r_moving_probs,
                    log_r_moving_probs,
                    GTs_moving,
                    criterion,
                    self.ls,
                )

                r_movable_losses, r_movable_preds = loss_and_to_labels(
                    r_movable_probs,
                    log_r_movable_probs,
                    GTs_movable,
                    movable_criterion,
                    self.ls,
                )

                b_moving_losses, b_moving_preds = loss_and_to_labels(
                    b_moving_probs,
                    log_b_moving_probs,
                    GTs_moving,
                    criterion,
                    self.ls,
                )

                b_movable_losses, b_movable_preds = loss_and_to_labels(
                    b_movable_probs,
                    log_b_movable_probs,
                    GTs_movable,
                    movable_criterion,
                    self.ls,
                )

                final_losses, final_preds = loss_and_to_labels(
                    final_probs,
                    log_final_probs,
                    GTs_moving,
                    criterion,
                    self.ls,
                )

                range_losses = r_moving_losses + r_movable_losses
                bev_losses = b_moving_losses + b_movable_losses

                # 평가 metric 업데이트
                # Range 평가 (moving, movable 각각 평가 후 평균)
                range_evaluator.addBatch(r_moving_preds, GTs_moving)
                range_movable_evaluator.addBatch(r_movable_preds, GTs_movable)
                avg_range_acc = (
                    range_evaluator.getacc() + range_movable_evaluator.getacc()
                ) / 2.0
                avg_range_iou = (
                    range_evaluator.getIoU()[0] + range_movable_evaluator.getIoU()[0]
                ) / 2.0

                # BEV 평가 (moving, movable 각각 평가 후 평균)
                bev_evaluator.addBatch(b_moving_preds, GTs_moving)
                bev_movable_evaluator.addBatch(b_movable_preds, GTs_movable)
                avg_bev_acc = (
                    bev_evaluator.getacc() + bev_movable_evaluator.getacc()
                ) / 2.0
                avg_bev_iou = (
                    bev_evaluator.getIoU()[0] + bev_movable_evaluator.getIoU()[0]
                ) / 2.0

                # Final 평가 (주로 GTs_moving 기준)
                final_evaluator.addBatch(final_preds, GTs_moving)
                final_acc = final_evaluator.getacc()
                final_iou_val = final_evaluator.getIoU()[0]

                # 미터 업데이트 (배치 크기 기준)
                range_loss_meter.update(range_losses.item(), points.size(0))
                bev_loss_meter.update(bev_losses.item(), points.size(0))
                final_loss_meter.update(final_losses.item(), points.size(0))
                range_acc_meter.update(avg_range_acc, points.size(0))
                range_iou_meter.update(avg_range_iou, points.size(0))
                bev_acc_meter.update(avg_bev_acc, points.size(0))
                bev_iou_meter.update(avg_bev_iou, points.size(0))
                final_acc_meter.update(final_acc, points.size(0))
                final_iou_meter.update(final_iou_val, points.size(0))

                # 스캔 이미지 저장 (원하는 경우)
                if save_scans:
                    # 예시: proj_full의 첫 번째 배치를 이용하여 이미지 생성
                    img_np = proj_full[0].cpu().numpy()
                    pred_np = final_preds[0].cpu().numpy()
                    gt_np = GTs_moving[0].cpu().numpy()
                    out = make_log_img(img_np, None, pred_np, gt_np, color_fn)
                    rand_imgs.append(out)

                self.batch_time_e.update(time.time() - end)
                end = time.time()

            log_str = (
                "*" * 80 + "\n"
                "Validation set:\n"
                f"Time avg per batch {self.batch_time_e.avg:.3f}\n"
                f"Range Loss avg {range_loss_meter.avg:.4f}\n"
                f"BEV Loss avg {bev_loss_meter.avg:.4f}\n"
                f"Final Loss avg {final_loss_meter.avg:.4f}\n"
                f"Range Acc avg {range_acc_meter.avg:.3f} | Range IoU avg {range_iou_meter.avg:.3f}\n"
                f"BEV Acc avg {bev_acc_meter.avg:.3f} | BEV IoU avg {bev_iou_meter.avg:.3f}\n"
                f"Final Acc avg {final_acc_meter.avg:.3f} | Final IoU avg {final_iou_meter.avg:.3f}\n"
                f"Hetero Loss avg {hetero_l.avg:.4f}\n"
            )
            print(log_str)
            save_to_txtlog(self.logdir, "log.txt", log_str)

            print("-" * 80)
            for idx, jacc in enumerate(final_evaluator.getIoU()[1]):
                self.info["valid_classes/" + class_func(idx)] = jacc
                class_log = f"IoU class {idx} [{class_func(idx)}] = {jacc:.6f}"
                print(class_log)
                save_to_txtlog(self.logdir, "log.txt", class_log)

            print("-" * 80)
            final_log = "*" * 80
            print(final_log)
            save_to_txtlog(self.logdir, "log.txt", final_log)

        return (
            final_acc_meter.avg,
            final_iou_meter.avg,
            final_loss_meter.avg,
            rand_imgs,
            hetero_l.avg,
        )

    def update_training_info(self, epoch, acc, iou, loss, update_mean, hetero_l):
        # 학습 정보 업데이트
        self.info["train_update"] = update_mean
        self.info["train_loss"] = loss
        self.info["train_acc"] = acc
        self.info["train_iou"] = iou
        self.info["train_hetero"] = hetero_l

        # 체크포인트 저장 (현재 상태)
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "info": self.info,
            "scheduler": self.scheduler.state_dict(),
        }
        save_checkpoint(state, self.logdir, suffix="")

        # 현재 train_iou가 최고치일 경우 모델 저장
        if self.info["train_iou"] > self.info.get("best_train_iou", 0):
            print("현재까지 학습 세트에서 최고 mean IoU 달성, 모델 저장합니다!")
            self.info["best_train_iou"] = self.info["train_iou"]
            state = {
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "info": self.info,
                "scheduler": self.scheduler.state_dict(),
            }
            save_checkpoint(state, self.logdir, suffix="_train_best")

    def update_validation_info(self, epoch, acc, iou, loss, hetero_l):
        # 검증 정보 업데이트
        self.info["valid_loss"] = loss
        self.info["valid_acc"] = acc
        self.info["valid_iou"] = iou
        self.info["valid_heteros"] = hetero_l

        # 현재 valid_iou가 최고치일 경우 모델 저장
        if self.info["valid_iou"] > self.info.get("best_val_iou", 0):
            log_str = (
                "현재까지 검증 세트에서 최고 mean IoU 달성, 모델 저장합니다!\n"
                + "*" * 80
            )
            print(log_str)
            save_to_txtlog(self.logdir, "log.txt", log_str)
            self.info["best_val_iou"] = self.info["valid_iou"]

            state = {
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "info": self.info,
                "scheduler": self.scheduler.state_dict(),
            }
            save_checkpoint(state, self.logdir, suffix="_valid_best")
            save_checkpoint(state, self.logdir, suffix=f"_valid_best_{epoch}")
