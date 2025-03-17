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
from common.dataset.kitti.parser import SemanticKitti
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


def loss_and_pred(probs, log_probs, labels, each_criterion, ls_func):
    if probs.numel() == 0:
        raise ValueError("No unprojected predictions found")

    jacc = ls_func(probs, labels)
    wce = each_criterion(log_probs.T.double(), labels).float()
    loss = wce + jacc

    pred = probs.argmax(dim=0)  # (n, )
    return loss, pred, (jacc, wce)


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
            "train_loss": 0,
            "train_range_acc": 0,
            "train_range_iou": 0,
            "train_bev_acc": 0,
            "train_bev_iou": 0,
            "train_final_acc": 0,
            "train_final_iou": 0,
            "train_hetero_loss": 0,
            "train_update": 0,
            "best_train_range_iou": 0,
            "best_train_bev_iou": 0,
            "best_train_final_iou": 0,
            ############################
            "val_loss": 0,
            "val_range_acc": 0,
            "val_range_iou": 0,
            "val_bev_acc": 0,
            "val_bev_iou": 0,
            "val_final_acc": 0,
            "val_final_iou": 0,
            "val_hetero_loss": 0,
            "best_val_range_iou": 0,
            "best_val_bev_iou": 0,
            "best_val_final_iou": 0,
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
            print("사전 학습된 모델을 불러옵니다.")
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
        self.criterion = nn.NLLLoss(
            ignore_index=0, weight=self.loss_w.double()).cuda()
        self.movable_criterion = nn.NLLLoss(
            ignore_index=0, weight=self.movable_loss_w.double()
        ).cuda()
        self.ls = Lovasz_softmax_PointCloud(ignore=0).to(self.device)
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
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # print("Training on device: ", self.device)

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
        checkpoint = "MFMOS_valid_final_best"
        w_dict = torch.load(
            os.path.join(self.path, checkpoint),
            map_location=lambda storage, loc: storage,
        )
        if not self.point_refine:
            self.model.load_state_dict(w_dict["state_dict"], strict=True)
        else:
            self.model.load_state_dict(
                {k.replace("module.", ""): v for k,
                 v in w_dict["state_dict"].items()}
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
            self.parser.get_n_classes(
                movable=True), self.local_rank, self.ignore_class
        )

        # BEV evaluator (범위에 상관없이 별도로 구성)
        self.bev_evaluator = iouEval(
            self.parser.get_n_classes(), self.local_rank, self.ignore_class
        )
        self.bev_movable_evaluator = iouEval(
            self.parser.get_n_classes(
                movable=True), self.local_rank, self.ignore_class
        )

        self.final_evaluator = iouEval(
            self.parser.get_n_classes(), self.local_rank, self.ignore_class
        )

    def train(self):
        self.init_evaluator()
        for epoch in range(self.epoch, self.ARCH["train"]["max_epochs"]):
            # self.parser.train_sampler.set_epoch(epoch)
            (
                overall_loss,
                range_acc,
                range_iou,
                bev_acc,
                bev_iou,
                final_acc,
                final_iou,
                update_ratio,
                hetero_l,
            ) = self.train_epoch(
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
            if self.local_rank == 0:
                self.update_training_info(
                    epoch,
                    overall_loss,
                    range_acc,
                    range_iou,
                    bev_acc,
                    bev_iou,
                    final_acc,
                    final_iou,
                    update_ratio,
                    hetero_l,
                )
            if epoch % self.ARCH["train"]["report_epoch"] == 0 and self.local_rank == 0:
                (
                    overall_loss,
                    range_acc,
                    range_iou,
                    bev_acc,
                    bev_iou,
                    final_acc,
                    final_iou,
                    hetero_l,
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
                )
                self.update_validation_info(
                    epoch,
                    overall_loss,
                    range_acc,
                    range_iou,
                    bev_acc,
                    bev_iou,
                    final_acc,
                    final_iou,
                    hetero_l,
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
                )
            # # dist.barrier()
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
        # Range Moving
        moving_losses_meter = AverageMeter()
        moving_acc_meter = AverageMeter()
        moving_iou_meter = AverageMeter()

        # Range Movable
        movable_losses_meter = AverageMeter()
        movable_acc_meter = AverageMeter()
        movable_iou_meter = AverageMeter()

        # BEV Moving
        bev_moving_losses_meter = AverageMeter()
        bev_moving_acc_meter = AverageMeter()
        bev_moving_iou_meter = AverageMeter()

        # BEV Movable
        bev_movable_losses_meter = AverageMeter()
        bev_movable_acc_meter = AverageMeter()
        bev_movable_iou_meter = AverageMeter()

        # Final Moving
        final_moving_losses_meter = AverageMeter()
        final_moving_acc_meter = AverageMeter()
        final_moving_iou_meter = AverageMeter()

        # ETC.
        losses_meter = AverageMeter()
        hetero_l_meter = AverageMeter()
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

        for i, (
            (points, (GTs_moving, GTs_movable)),
            (proj_full, bev_full),
            (proj_labels, proj_movable_labels),
            (bev_labels, bev_movable_labels),
            (proj_x, proj_y),
            (bev_proj_x, bev_proj_y),
            (path_seq, path_name),
        ) in enumerate(train_loader):
            self.data_time_t.update(time.time() - end)
            if self.gpu:
                points = points.cuda(non_blocking=True)
                proj_full = proj_full.cuda(non_blocking=True)
                bev_full = bev_full.cuda(non_blocking=True)
                proj_labels = proj_labels.cuda(non_blocking=True).long()
                proj_movable_labels = proj_movable_labels.cuda(
                    non_blocking=True).long()
                bev_labels = bev_labels.cuda(non_blocking=True).long()
                bev_movable_labels = bev_movable_labels.cuda(
                    non_blocking=True).long()
                proj_x = proj_x.cuda(non_blocking=True)
                proj_y = proj_y.cuda(non_blocking=True)
                bev_proj_x = bev_proj_x.cuda(non_blocking=True)
                bev_proj_y = bev_proj_y.cuda(non_blocking=True)
                GTs_moving = GTs_moving.cuda(non_blocking=True)
                GTs_movable = GTs_movable.cuda(non_blocking=True)

            # softmax 거쳤으므로 0-1 범위.
            r_moving, r_movable, b_moving, b_movable = model(
                proj_full, bev_full)

            """모든것을 batch 해제"""
            """이 아래부터는 배치가 1임을 가정하고 배치를 해제합니다."""
            points = points[0]
            npoints = points.shape[0]
            proj_full, bev_full = proj_full[0], bev_full[0]
            proj_labels, proj_movable_labels = proj_labels[0], proj_movable_labels[0]
            bev_labels, bev_movable_labels = bev_labels[0], bev_movable_labels[0]
            proj_x, proj_y = proj_x[0], proj_y[0]
            bev_proj_x, bev_proj_y = bev_proj_x[0], bev_proj_y[0]
            GTs_moving, GTs_movable = GTs_moving[0], GTs_movable[0]
            r_moving, r_movable = r_moving[0], r_movable[0]
            b_moving, b_movable = b_moving[0], b_movable[0]

            # (n, 3)
            # (1, ) =: n
            # (13, h, w), (13, h_bev, w_bev)
            # h, w
            # h_bev, w_bev
            # (150000, ), (150000, )
            # (150000, ), (150000, )
            # (n, ), (n, )
            # (3, h, w), (3, h, w)
            # (3, h_bev, w_bev), (3, h_bev, w_bev)

            ##############################################
            proj_x = proj_x[:npoints]  # (n, )
            proj_y = proj_y[:npoints]  # (n, )
            bev_proj_x = bev_proj_x[:npoints]  # (n, )
            bev_proj_y = bev_proj_y[:npoints]  # (n, )

            # 3차원 공간에서 점별 클래스 확률
            moving = r_moving[:, proj_y, proj_x]  # (3, n)
            movable = r_movable[:, proj_y, proj_x]  # (3, n)
            bev_moving = b_moving[:, bev_proj_y, bev_proj_x]  # (3, n)
            bev_movable = b_movable[:, bev_proj_y, bev_proj_x]  # (3, n)
            final_moving = (moving + bev_moving) / 2  # (3, n)

            ##############################################

            log_moving = torch.log(moving.clamp(min=1e-8))
            log_movable = torch.log(movable.clamp(min=1e-8))
            log_bev_moving = torch.log(bev_moving.clamp(min=1e-8))
            log_bev_movable = torch.log(bev_movable.clamp(min=1e-8))
            log_final_moving = torch.log(final_moving.clamp(min=1e-8))

            l_moving, pred_moving, _ = loss_and_pred(
                moving, log_moving, GTs_moving, criterion, self.ls)
            l_movable, pred_movable, _ = loss_and_pred(
                movable, log_movable, GTs_movable, movable_criterion, self.ls)
            l_bev_moving, pred_bev_moving, _ = loss_and_pred(
                bev_moving, log_bev_moving, GTs_moving, criterion, self.ls)
            l_bev_movable, pred_bev_movable, _ = loss_and_pred(
                bev_movable, log_bev_movable, GTs_movable, movable_criterion, self.ls)
            l_final_moving, pred_final_moving, _ = loss_and_pred(
                final_moving, log_final_moving, GTs_moving, criterion, self.ls)

            l_range = l_moving + l_movable
            l_bev = l_bev_moving + l_bev_movable
            loss = l_range + l_bev + l_final_moving

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mean_loss = loss.mean()
            with torch.no_grad():
                range_evaluator.reset()
                range_evaluator.addBatch(pred_moving, GTs_moving)
                range_acc = range_evaluator.getacc()
                range_jaccard, _ = range_evaluator.getIoU()

                range_movable_evaluator.reset()
                range_movable_evaluator.addBatch(pred_movable, GTs_movable)
                range_movable_acc = range_movable_evaluator.getacc()
                range_movable_jaccard, _ = range_movable_evaluator.getIoU()

                # BEV 평가: b_moving과 b_movable 결과 평가 후 평균 처리
                bev_evaluator.reset()
                bev_evaluator.addBatch(pred_bev_moving, GTs_moving)
                bev_acc = bev_evaluator.getacc()
                bev_jaccard, _ = bev_evaluator.getIoU()

                bev_movable_evaluator.reset()
                bev_movable_evaluator.addBatch(pred_bev_movable, GTs_movable)
                bev_movable_acc = bev_movable_evaluator.getacc()
                bev_movable_jaccard, _ = bev_movable_evaluator.getIoU()

                # Final 평가: final_preds와 GTs_moving 비교
                final_evaluator.reset()
                final_evaluator.addBatch(pred_final_moving, GTs_moving)
                final_acc = final_evaluator.getacc()
                final_jaccard, _ = final_evaluator.getIoU()

            losses_meter.update(mean_loss.item())

            moving_losses_meter.update(l_moving.mean().item())
            movable_losses_meter.update(l_movable.mean().item())
            moving_acc_meter.update(range_acc.item())
            moving_iou_meter.update(range_jaccard.item())
            movable_acc_meter.update(range_movable_acc.item())
            movable_iou_meter.update(range_movable_jaccard.item())

            bev_moving_losses_meter.update(l_bev_moving.mean().item())
            bev_movable_losses_meter.update(l_bev_movable.mean().item())
            bev_moving_acc_meter.update(bev_acc.item())
            bev_moving_iou_meter.update(bev_jaccard.item())
            bev_movable_acc_meter.update(bev_movable_acc.item())
            bev_movable_iou_meter.update(bev_movable_jaccard.item())

            final_moving_losses_meter.update(l_final_moving.mean().item())
            final_moving_acc_meter.update(final_acc.item())
            final_moving_iou_meter.update(final_jaccard.item())

            # measure elapsed time
            self.batch_time_t.update(time.time() - end)
            end = time.time()

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
                ratios = torch.stack(ratios)
                update_mean = ratios.mean().item()
                update_std = ratios.std().item()
            else:
                raise Exception("No gradients found in optimizer parameters.")
            update_ratio_meter.update(update_mean)

            # 일정 배치마다 로그 출력
            if i % report == 0 and self.local_rank == 0:
                str_line = (
                    "Lr: {lr:.3e} | "
                    "Update: {umean:.3e} mean,{ustd:.3e} std | "
                    "Epoch: [{0}][{1}/{2}] | "
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | "
                    "Data {data_time.val:.3f} ({data_time.avg:.3f}) | "
                    "Overall-Loss {loss.val:.4f} ({loss.avg:.4f}) | "
                    "MovingLoss {moving_losses.val:.4f} ({moving_losses.avg:.4f}) | "
                    "MovableLoss {movable_losses.val:.4f} ({movable_losses.avg:.4f}) | "
                    "MovingAcc {moving_acc.val:.3f} ({moving_acc.avg:.3f}) | "
                    "MovingIoU {moving_iou.val:.3f} ({moving_iou.avg:.3f}) | "
                    "Movable_acc {movable_acc.val:.3f} ({movable_acc.avg:.3f}) | "
                    "Movable_IoU {movable_iou.val:.3f} ({movable_iou.avg:.3f}) | "
                    "BEV_MovingLoss {bev_moving_losses.val:.4f} ({bev_moving_losses.avg:.4f}) | "
                    "BEV_MovableLoss {bev_movable_losses.val:.4f} ({bev_movable_losses.avg:.4f}) | "
                    "BEV_MovingAcc {bev_moving_acc.val:.3f} ({bev_moving_acc.avg:.3f}) | "
                    "BEV_MovingIoU {bev_moving_iou.val:.3f} ({bev_moving_iou.avg:.3f}) | "
                    "BEV_Movable_acc {bev_movable_acc.val:.3f} ({bev_movable_acc.avg:.3f}) | "
                    "BEV_Movable_IoU {bev_movable_iou.val:.3f} ({bev_movable_iou.avg:.3f}) | "
                    "Final_MovingLoss {final_moving_losses.val:.4f} ({final_moving_losses.avg:.4f}) | "
                    "Final_MovingAcc {final_moving_acc.val:.3f} ({final_moving_acc.avg:.3f}) | "
                    "Final_MovingIoU {final_moving_iou.val:.3f} ({final_moving_iou.avg:.3f}) | "
                    "[{estim}]"
                ).format(
                    epoch,
                    i,
                    len(train_loader),
                    #############################
                    lr=lr,
                    umean=update_mean,
                    ustd=update_std,
                    batch_time=self.batch_time_t,
                    data_time=self.data_time_t,
                    loss=losses_meter,
                    moving_losses=moving_losses_meter,
                    movable_losses=movable_losses_meter,
                    moving_acc=moving_acc_meter,
                    moving_iou=moving_iou_meter,
                    movable_acc=movable_acc_meter,
                    movable_iou=movable_iou_meter,
                    bev_moving_losses=bev_moving_losses_meter,
                    bev_movable_losses=bev_movable_losses_meter,
                    bev_moving_acc=bev_moving_acc_meter,
                    bev_moving_iou=bev_moving_iou_meter,
                    bev_movable_acc=bev_movable_acc_meter,
                    bev_movable_iou=bev_movable_iou_meter,
                    final_moving_losses=final_moving_losses_meter,
                    final_moving_acc=final_moving_acc_meter,
                    final_moving_iou=final_moving_iou_meter,

                    estim=self.calculate_estimate(epoch, i),
                )
                print("*" * 80)
                print(str_line)
                save_to_txtlog(self.logdir, "log.txt", str_line)

            scheduler.step()

        return (
            losses_meter.avg,
            moving_acc_meter.avg,
            moving_iou_meter.avg,
            bev_moving_acc_meter.avg,
            bev_moving_iou_meter.avg,
            final_moving_acc_meter.avg,
            final_moving_iou_meter.avg,
            update_ratio_meter.avg,
            hetero_l_meter.avg,
        )

    def validate(
        self,
        val_loader,
        model,
        all_criterion,
        all_evaluator,
        class_func,
    ):
        losses_meter = AverageMeter()

        moving_losses_meter = AverageMeter()
        movable_losses_meter = AverageMeter()
        bev_moving_losses_meter = AverageMeter()
        bev_movable_losses_meter = AverageMeter()
        final_moving_losses_meter = AverageMeter()

        jaccs_meter = AverageMeter()
        wces_meter = AverageMeter()
        acc_meter = AverageMeter()
        iou_meter = AverageMeter()

        bev_jaccs_meter = AverageMeter()
        bev_wces_meter = AverageMeter()
        bev_acc_meter = AverageMeter()
        bev_iou_meter = AverageMeter()

        final_jaccs_meter = AverageMeter()
        final_wces_meter = AverageMeter()
        final_acc_meter = AverageMeter()
        final_iou_meter = AverageMeter()

        movable_jaccs_meter = AverageMeter()
        movable_wces_meter = AverageMeter()
        movable_acc_meter = AverageMeter()
        movable_iou_meter = AverageMeter()

        bev_movable_jaccs_meter = AverageMeter()
        bev_movable_wces_meter = AverageMeter()
        bev_movable_acc_meter = AverageMeter()
        bev_movable_iou_meter = AverageMeter()

        hetero_l_meter = AverageMeter()

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
                (points, (GTs_moving, GTs_movable)),
                (proj_full, bev_full),
                (proj_labels, proj_movable_labels),
                (bev_labels, bev_movable_labels),
                (proj_x, proj_y),
                (bev_proj_x, bev_proj_y),
                (path_seq, path_name),
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
                    GTs_moving = GTs_moving.cuda(non_blocking=True)
                    GTs_movable = GTs_movable.cuda(non_blocking=True)

                # softmax 거쳤으므로 0-1 범위.
                r_moving, r_movable, b_moving, b_movable = model(
                    proj_full, bev_full)

                """모든것을 batch 해제"""
                """이 아래부터는 배치가 1임을 가정하고 배치를 해제합니다."""
                points = points[0]
                npoints = points.shape[0]
                proj_full, bev_full = proj_full[0], bev_full[0]
                proj_labels, proj_movable_labels = proj_labels[0], proj_movable_labels[0]
                bev_labels, bev_movable_labels = bev_labels[0], bev_movable_labels[0]
                proj_x, proj_y = proj_x[0], proj_y[0]
                bev_proj_x, bev_proj_y = bev_proj_x[0], bev_proj_y[0]
                GTs_moving, GTs_movable = GTs_moving[0], GTs_movable[0]
                r_moving, r_movable = r_moving[0], r_movable[0]
                b_moving, b_movable = b_moving[0], b_movable[0]

                # (n, 3)
                # (1, ) =: n
                # (13, h, w), (13, h_bev, w_bev)
                # h, w
                # h_bev, w_bev
                # (150000, ), (150000, )
                # (150000, ), (150000, )
                # (n, ), (n, )
                # (3, h, w), (3, h, w)
                # (3, h_bev, w_bev), (3, h_bev, w_bev)

                ##############################################
                proj_x = proj_x[:npoints]  # (n, )
                proj_y = proj_y[:npoints]  # (n, )
                bev_proj_x = bev_proj_x[:npoints]  # (n, )
                bev_proj_y = bev_proj_y[:npoints]  # (n, )

                # 3차원 공간에서 점별 클래스 확률
                moving = r_moving[:, proj_y, proj_x]  # (3, n)
                movable = r_movable[:, proj_y, proj_x]  # (3, n)
                bev_moving = b_moving[:, bev_proj_y, bev_proj_x]  # (3, n)
                bev_movable = b_movable[:, bev_proj_y, bev_proj_x]  # (3, n)
                final_moving = (moving + bev_moving) / 2  # (3, n)

                ##############################################

                log_moving = torch.log(moving.clamp(min=1e-8))
                log_movable = torch.log(movable.clamp(min=1e-8))
                log_bev_moving = torch.log(bev_moving.clamp(min=1e-8))
                log_bev_movable = torch.log(bev_movable.clamp(min=1e-8))
                log_final_moving = torch.log(final_moving.clamp(min=1e-8))

                l_moving, pred_moving, (jacc, wce) = loss_and_pred(
                    moving, log_moving, GTs_moving, criterion, self.ls)
                l_movable, pred_movable, (movable_jacc, movable_wce) = loss_and_pred(
                    movable, log_movable, GTs_movable, movable_criterion, self.ls)
                l_bev_moving, pred_bev_moving, (bev_jacc, bev_wce) = loss_and_pred(
                    bev_moving, log_bev_moving, GTs_moving, criterion, self.ls)
                l_bev_movable, pred_bev_movable, (bev_movable_jacc, bev_movable_wce) = loss_and_pred(
                    bev_movable, log_bev_movable, GTs_movable, movable_criterion, self.ls)
                l_final_moving, pred_final_moving, (final_jacc, final_wce) = loss_and_pred(
                    final_moving, log_final_moving, GTs_moving, criterion, self.ls)

                l_range = l_moving + l_movable
                l_bev = l_bev_moving + l_bev_movable
                loss = l_range + l_bev + l_final_moving

                range_evaluator.addBatch(pred_moving, GTs_moving)
                range_movable_evaluator.addBatch(pred_movable, GTs_movable)
                bev_evaluator.addBatch(pred_bev_moving, GTs_moving)
                bev_movable_evaluator.addBatch(pred_bev_movable, GTs_movable)
                final_evaluator.addBatch(pred_final_moving, GTs_moving)

                losses_meter.update(loss.mean().item())

                moving_losses_meter.update(l_moving.mean().item())
                jaccs_meter.update(jacc.mean().item())
                wces_meter.update(wce.mean().item())
                movable_losses_meter.update(l_movable.mean().item())
                movable_jaccs_meter.update(movable_jacc.mean().item())
                movable_wces_meter.update(movable_wce.mean().item())

                bev_moving_losses_meter.update(l_bev_moving.mean().item())
                bev_jaccs_meter.update(bev_jacc.mean().item())
                bev_wces_meter.update(bev_wce.mean().item())
                bev_movable_losses_meter.update(l_bev_movable.mean().item())
                bev_movable_jaccs_meter.update(bev_movable_jacc.mean().item())
                bev_movable_wces_meter.update(bev_movable_wce.mean().item())

                final_moving_losses_meter.update(l_final_moving.mean().item())
                final_jaccs_meter.update(final_jacc.mean().item())
                final_wces_meter.update(final_wce.mean().item())

                self.batch_time_e.update(time.time() - end)
                end = time.time()

            accuracy = range_evaluator.getacc()
            jaccard, class_jaccard = range_evaluator.getIoU()
            movable_accuracy = range_movable_evaluator.getacc()
            movable_jaccard, movable_class_jaccard = range_movable_evaluator.getIoU()
            acc_meter.update(accuracy.item())
            iou_meter.update(jaccard.item())
            movable_acc_meter.update(movable_accuracy.item())
            movable_iou_meter.update(movable_jaccard.item())

            bev_accuracy = bev_evaluator.getacc()
            bev_jaccard, bev_class_jaccard = bev_evaluator.getIoU()
            bev_movable_accuracy = bev_movable_evaluator.getacc()
            bev_movable_jaccard, bev_movable_class_jaccard = bev_movable_evaluator.getIoU()
            bev_acc_meter.update(bev_accuracy.item())
            bev_iou_meter.update(bev_jaccard.item())
            bev_movable_acc_meter.update(bev_movable_accuracy.item())
            bev_movable_iou_meter.update(bev_movable_jaccard.item())

            final_accuracy = final_evaluator.getacc()
            final_jaccard, final_class_jaccard = final_evaluator.getIoU()
            final_acc_meter.update(final_accuracy.item())
            final_iou_meter.update(final_jaccard.item())

            str_line = (
                "*" * 80 + "\n"
                "Validation set:\n"
                "Time avg per batch {batch_time.avg:.3f}\n"
                "Loss avg {loss.avg:.4f}\n"
                "MovingLoss avg {moving_loss.avg:.4f}\n"
                "MovableLoss avg {movable_loss.avg:.4f}\n"
                "MovingJaccard avg {moving_jac.avg:.4f}\n"
                "MovingWCE avg {moving_wces.avg:.4f}\n"
                "MovingAcc avg {moving_acc.avg:.6f}\n"
                "MovingIoU avg {moving_iou.avg:.6f}\n"
                "MovableJaccard avg {movable_jac.avg:.4f}\n"
                "MovableWCE avg {movable_wce.avg:.4f}\n"
                "MovableAcc avg {movable_acc.avg:.6f}\n"
                "MovableIoU avg {movable_iou.avg:.6f}\n"
                "BEV_MovingLoss avg {bev_moving_loss.avg:.4f}\n"
                "BEV_MovableLoss avg {bev_movable_loss.avg:.4f}\n"
                "BEV_MovingJaccard avg {bev_moving_jac.avg:.4f}\n"
                "BEV_MovingWCE avg {bev_moving_wces.avg:.4f}\n"
                "BEV_MovingAcc avg {bev_moving_acc.avg:.6f}\n"
                "BEV_MovingIoU avg {bev_moving_iou.avg:.6f}\n"
                "BEV_MovableJaccard avg {bev_movable_jac.avg:.4f}\n"
                "BEV_MovableWCE avg {bev_movable_wce.avg:.4f}\n"
                "BEV_MovableAcc avg {bev_movable_acc.avg:.6f}\n"
                "BEV_MovableIoU avg {bev_movable_iou.avg:.6f}\n"
                "Final_MovingLoss avg {final_moving_loss.avg:.4f}\n"
                "Final_MovingJaccard avg {final_moving_jac.avg:.4f}\n"
                "Final_MovingWCE avg {final_moving_wces.avg:.4f}\n"
                "Final_MovingAcc avg {final_moving_acc.avg:.6f}\n"
                "Final_MovingIoU avg {final_moving_iou.avg:.6f}\n"
            ).format(
                batch_time=self.batch_time_e,
                loss=losses_meter,
                moving_loss=moving_losses_meter,
                movable_loss=movable_losses_meter,
                moving_jac=jaccs_meter,
                moving_wces=wces_meter,
                moving_acc=acc_meter,
                moving_iou=iou_meter,
                movable_jac=movable_jaccs_meter,
                movable_wce=movable_wces_meter,
                movable_acc=movable_acc_meter,
                movable_iou=movable_iou_meter,
                bev_moving_loss=bev_moving_losses_meter,
                bev_movable_loss=bev_movable_losses_meter,
                bev_moving_jac=bev_jaccs_meter,
                bev_moving_wces=bev_wces_meter,
                bev_moving_acc=bev_acc_meter,
                bev_moving_iou=bev_iou_meter,
                bev_movable_jac=bev_movable_jaccs_meter,
                bev_movable_wce=bev_movable_wces_meter,
                bev_movable_acc=bev_movable_acc_meter,
                bev_movable_iou=bev_movable_iou_meter,
                final_moving_loss=final_moving_losses_meter,
                final_moving_jac=final_jaccs_meter,
                final_moving_wces=final_wces_meter,
                final_moving_acc=final_acc_meter,
                final_moving_iou=final_iou_meter,
            )
            print(str_line)
            save_to_txtlog(self.logdir, "log.txt", str_line)

            print("-" * 80)
            # print also classwise
            for i, jacc in enumerate(class_jaccard):
                self.info["valid_classes/" + class_func(i)] = jacc
                str_line = "Range IoU class {i:} [{class_str:}] = {jacc:.6f}".format(
                    i=i, class_str=class_func(i), jacc=jacc
                )
                print(str_line)
                save_to_txtlog(self.logdir, "log.txt", str_line)

            print("-" * 80)
            for i, jacc in enumerate(movable_class_jaccard):
                self.info["valid_classes/" +
                          class_func(i, movable=True)] = jacc
                str_line = "Range IoU class {i:} [{class_str:}] = {jacc:.6f}".format(
                    i=i, class_str=class_func(i, movable=True), jacc=jacc
                )
                print(str_line)
                save_to_txtlog(self.logdir, "log.txt", str_line)

            print("-" * 80)
            # print also classwise
            for i, jacc in enumerate(bev_class_jaccard):
                if i == 0:
                    continue
                self.info["valid_classes/" + class_func(i)] = jacc
                str_line = "BEV IoU class {i:} [{class_str:}] = {jacc:.6f}".format(
                    i=i, class_str=class_func(i), jacc=jacc
                )
                print(str_line)
                save_to_txtlog(self.logdir, "log.txt", str_line)

            print("-" * 80)
            for i, jacc in enumerate(bev_movable_class_jaccard):
                if i == 0:
                    continue
                self.info["valid_classes/" +
                          class_func(i, movable=True)] = jacc
                str_line = "BEV IoU class {i:} [{class_str:}] = {jacc:.6f}".format(
                    i=i, class_str=class_func(i, movable=True), jacc=jacc
                )
                print(str_line)
                save_to_txtlog(self.logdir, "log.txt", str_line)

            print("-" * 80)
            # print also classwise
            for i, jacc in enumerate(final_class_jaccard):
                if i == 0:
                    continue
                self.info["valid_classes/" + class_func(i)] = jacc
                str_line = "Final IoU class {i:} [{class_str:}] = {jacc:.6f}".format(
                    i=i, class_str=class_func(i), jacc=jacc
                )
                print(str_line)
                save_to_txtlog(self.logdir, "log.txt", str_line)

            str_line = "*" * 80
            print(str_line)
            save_to_txtlog(self.logdir, "log.txt", str_line)

        return (
            losses_meter.avg,
            acc_meter.avg,
            iou_meter.avg,
            bev_acc_meter.avg,
            bev_iou_meter.avg,
            final_acc_meter.avg,
            final_iou_meter.avg,
            hetero_l_meter.avg,
        )

    def update_training_info(
        self,
        epoch,
        overall_loss,
        range_acc,
        range_iou,
        bev_acc,
        bev_iou,
        final_acc,
        final_iou,
        update_ratio,
        hetero_l,
    ):
        self.info["train_loss"] = overall_loss
        self.info["train_range_acc"] = range_acc
        self.info["train_range_iou"] = range_iou
        self.info["train_bev_acc"] = bev_acc
        self.info["train_bev_iou"] = bev_iou
        self.info["train_final_acc"] = final_acc
        self.info["train_final_iou"] = final_iou
        self.info["train_hetero_loss"] = hetero_l
        self.info["train_update"] = update_ratio

        # 4) 현재 상태로 체크포인트 저장
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "info": self.info,
            "scheduler": self.scheduler.state_dict(),
        }
        save_checkpoint(state, self.logdir, suffix="")

        # 5) best 기록 갱신 (range/BEV/final 각각)
        #    - 필요 없다면 생략 가능
        if range_iou > self.info["best_train_range_iou"]:
            print("Range branch: 최고 train IoU 갱신!")
            self.info["best_train_range_iou"] = range_iou
            save_checkpoint(state, self.logdir,
                            suffix=f"_train_range_best")

        if bev_iou > self.info["best_train_bev_iou"]:
            print("BEV branch: 최고 train IoU 갱신!")
            self.info["best_train_bev_iou"] = bev_iou
            save_checkpoint(state, self.logdir,
                            suffix=f"_train_bev_best")

        if final_iou > self.info["best_train_final_iou"]:
            print("Final branch: 최고 train IoU 갱신!")
            self.info["best_train_final_iou"] = final_iou
            save_checkpoint(state, self.logdir,
                            suffix=f"_train_final_best")

    def update_validation_info(
        self,
        epoch,
        overall_loss,
        range_acc,
        range_iou,
        bev_acc,
        bev_iou,
        final_acc,
        final_iou,
        hetero_l,
    ):
        """
        검증 과정에서 range/BEV/final 각각의 loss/acc/IoU를 받아 self.info에 기록하고,
        best 기록을 갱신하며 체크포인트를 저장합니다.
        """
        self.info["val_loss"] = overall_loss
        self.info["val_range_acc"] = range_acc
        self.info["val_range_iou"] = range_iou
        self.info["val_bev_acc"] = bev_acc
        self.info["val_bev_iou"] = bev_iou
        self.info["val_final_acc"] = final_acc
        self.info["val_final_iou"] = final_iou
        self.info["val_hetero_loss"] = hetero_l

        # 3) 체크포인트 저장
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "info": self.info,
            "scheduler": self.scheduler.state_dict(),
        }
        save_checkpoint(state, self.logdir, suffix="_valid_last")

        # 4) best 기록 갱신
        if range_iou > self.info["best_val_range_iou"]:
            print("Range branch: 최고 valid IoU 갱신!")
            self.info["best_val_range_iou"] = range_iou
            save_checkpoint(state, self.logdir,
                            suffix=f"_valid_range_best_{epoch}_iou_{range_iou}")

        if bev_iou > self.info["best_val_bev_iou"]:
            print("BEV branch: 최고 valid IoU 갱신!")
            self.info["best_val_bev_iou"] = bev_iou
            save_checkpoint(state, self.logdir,
                            suffix=f"_valid_bev_best_{epoch}_iou_{bev_iou}")

        if final_iou > self.info["best_val_final_iou"]:
            print("Final branch: 최고 valid IoU 갱신!")
            self.info["best_val_final_iou"] = final_iou
            save_checkpoint(state, self.logdir,
                            suffix=f"_valid_final_best_{epoch}_iou_{final_iou}")
