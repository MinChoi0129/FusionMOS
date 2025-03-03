#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
수정된 trainer.py 파일 (새로운 MFMOS.py 모델에 호환)
Range 관련 변수는 range_ 접두사, BEV 관련 변수는 bev_ 접두사를 사용합니다.
손실(loss)은 range_moving, range_movable, bev_moving, bev_movable 네 손실의 합으로 정의하며,
각 branch별로 별도의 AverageMeter 및 IoU evaluator를 구성하여 metric들을 산출합니다.
"""

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
from modules.loss.Lovasz_Softmax import Lovasz_softmax, Lovasz_softmax_PointCloud
from modules.tools import (
    AverageMeter,
    iouEval,
    save_checkpoint,
    save_to_txtlog,
    make_log_img,
)

from torch import distributed as dist


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
        self.criterion = nn.NLLLoss(weight=self.loss_w.double()).cuda()
        self.movable_criterion = nn.NLLLoss(weight=self.movable_loss_w.double()).cuda()
        if not point_refine:
            self.ls = Lovasz_softmax(ignore=0).cuda()
            self.movable_ls = Lovasz_softmax(ignore=0).cuda()
        else:
            self.ls = Lovasz_softmax_PointCloud(ignore=0).to(self.device)

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
        """
        Range 및 BEV 각각의 evaluator 초기화
        """
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
                    ),
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
        # AverageMeter 정의 (각 branch별 손실 및 metric)
        range_loss_meter = AverageMeter()
        range_moving_loss_meter = AverageMeter()
        range_movable_loss_meter = AverageMeter()
        bev_loss_meter = AverageMeter()
        bev_moving_loss_meter = AverageMeter()
        bev_movable_loss_meter = AverageMeter()
        range_acc_meter = AverageMeter()
        range_iou_meter = AverageMeter()
        bev_acc_meter = AverageMeter()
        bev_iou_meter = AverageMeter()
        update_ratio_meter = AverageMeter()

        (
            range_evaluator,
            range_movable_evaluator,
            bev_evaluator,
            bev_movable_evaluator,
        ) = all_evaluator
        criterion, movable_criterion = all_criterion  # 각각 NLLLoss 계열 사용

        model.train()
        end = time.time()
        for i, (
            (proj_full, bev_full),  # range (13, H, W) | bev (13, H_bev, W_bev)
            (proj_labels, proj_movable_labels),  # range (H, W), (H, W)
            (bev_labels, bev_movable_labels),  # bev (H_bev, W_bev), (H_bev, W_bev)
            (path_seq, path_name, unproj_n_points),  # ex. '08', '000123.npy', 122319
            (proj_x, proj_y),  # (150000, ), (150000, )
            (bev_proj_x, bev_proj_y),  # (150000, ), (150000, )
            (proj_range, bev_range),  # (H_range, W_range), (H_BEV, W_BEV)
            (unproj_range, bev_unproj_range),  # (150000, ), (150000, )
        ) in enumerate(train_loader):

            print(
                proj_x.shape,
                proj_y.shape,
                proj_range.shape,
                unproj_range.shape,
                bev_proj_x.shape,
                bev_proj_y.shape,
                bev_range.shape,
                bev_unproj_range.shape,
            )

            print("\n---------------------------------------")
            print(
                np.unique(proj_x),
                np.unique(proj_y),
                proj_range[:10],
                unproj_range[:10],
                np.unique(bev_proj_x),
                np.unique(bev_proj_y),
                bev_range[:10],
                bev_unproj_range[:10],
                sep="\n",
            )
            print("---------------------------------------")

            self.data_time_t.update(time.time() - end)
            if self.gpu:
                proj_full = proj_full.cuda()
                bev_full = bev_full.cuda()
                proj_labels = proj_labels.cuda().long()
                proj_movable_labels = proj_movable_labels.cuda().long()
                bev_labels = bev_labels.cuda().long()
                bev_movable_labels = bev_movable_labels.cuda().long()
            # 모델의 forward: 새로운 MFMOS는 (range_moving, range_movable, bev_moving, bev_movable)를 반환
            range_moving, range_movable, bev_moving, bev_movable = model(
                proj_full, bev_full
            )

            # 손실 계산 (각 branch별)
            range_moving_loss = criterion(
                torch.log(range_moving.clamp(min=1e-8)).double(), proj_labels
            ).float() + self.ls(range_moving, proj_labels.long())
            range_movable_loss = movable_criterion(
                torch.log(range_movable.clamp(min=1e-8)).double(), proj_movable_labels
            ).float() + self.movable_ls(range_movable, proj_movable_labels.long())
            bev_moving_loss = criterion(
                torch.log(bev_moving.clamp(min=1e-8)).double(), bev_labels
            ).float() + self.ls(bev_moving, bev_labels.long())
            bev_movable_loss = movable_criterion(
                torch.log(bev_movable.clamp(min=1e-8)).double(), bev_movable_labels
            ).float() + self.movable_ls(bev_movable, bev_movable_labels.long())
            loss_total = (
                range_moving_loss
                + range_movable_loss
                + bev_moving_loss
                + bev_movable_loss
            )

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            # 평가 (torch.no_grad() 내에서)
            with torch.no_grad():
                range_evaluator.reset()
                range_movable_evaluator.reset()
                bev_evaluator.reset()
                bev_movable_evaluator.reset()

                range_pred = range_moving.argmax(dim=1)
                range_evaluator.addBatch(range_pred, proj_labels)
                range_acc = range_evaluator.getacc()
                range_jacc, _ = range_evaluator.getIoU()

                bev_pred = bev_moving.argmax(dim=1)
                bev_evaluator.addBatch(bev_pred, bev_labels)
                bev_acc = bev_evaluator.getacc()
                bev_jacc, _ = bev_evaluator.getIoU()

                range_movable_pred = range_movable.argmax(dim=1)
                range_movable_evaluator.addBatch(
                    range_movable_pred, proj_movable_labels
                )
                range_movable_acc = range_movable_evaluator.getacc()
                range_movable_jacc, _ = range_movable_evaluator.getIoU()

                bev_movable_pred = bev_movable.argmax(dim=1)
                bev_movable_evaluator.addBatch(bev_movable_pred, bev_movable_labels)
                bev_movable_acc = bev_movable_evaluator.getacc()
                bev_movable_jacc, _ = bev_movable_evaluator.getIoU()

            # AverageMeter 업데이트
            range_loss_meter.update(
                (range_moving_loss + range_movable_loss).mean().item(),
                proj_full.size(0),
            )
            bev_loss_meter.update(
                (bev_moving_loss + bev_movable_loss).mean().item(), bev_full.size(0)
            )
            range_moving_loss_meter.update(
                range_moving_loss.mean().item(), proj_full.size(0)
            )
            range_movable_loss_meter.update(
                range_movable_loss.mean().item(), proj_full.size(0)
            )
            bev_moving_loss_meter.update(
                bev_moving_loss.mean().item(), bev_full.size(0)
            )
            bev_movable_loss_meter.update(
                bev_movable_loss.mean().item(), bev_full.size(0)
            )
            range_acc_meter.update(range_acc.item(), proj_full.size(0))
            range_iou_meter.update(range_jacc.item(), proj_full.size(0))
            bev_acc_meter.update(bev_acc.item(), bev_full.size(0))
            bev_iou_meter.update(bev_jacc.item(), bev_full.size(0))

            # gradient update ratio 계산
            update_ratios = []
            for g in optimizer.param_groups:
                lr = g["lr"]
                for param in g["params"]:
                    if param.grad is not None:
                        w = np.linalg.norm(param.data.cpu().numpy().reshape(-1))
                        upd = np.linalg.norm(
                            -max(lr, 1e-10) * param.grad.cpu().numpy().reshape(-1)
                        )
                        update_ratios.append(upd / max(w, 1e-10))
            if update_ratios:
                update_mean = np.mean(update_ratios)
                update_ratio_meter.update(update_mean)
            self.batch_time_t.update(time.time() - end)
            end = time.time()

            if i % report == 0 and self.local_rank == 0:
                log_str = (
                    "Lr: {lr:.3e} | Epoch: [{epoch}][{i}/{n}] | "
                    "Time {bt.val:.3f} ({bt.avg:.3f}) | Data {dt.val:.3f} ({dt.avg:.3f}) | "
                    "RangeLoss {rl.val:.4f} ({rl.avg:.4f}) | "
                    "RangeMovingLoss {rml.val:.4f} | RangeMovableLoss {rmvl.val:.4f} | "
                    "RangeAcc {ra.val:.3f} ({ra.avg:.3f}) | RangeIoU {ri.val:.3f} ({ri.avg:.3f}) || "
                    "BevLoss {bl.val:.4f} ({bl.avg:.4f}) | "
                    "BevMovingLoss {bml.val:.4f} | BevMovableLoss {bmvl.val:.4f} | "
                    "BevAcc {ba.val:.3f} ({ba.avg:.3f}) | BevIoU {bi.val:.3f} ({bi.avg:.3f}) || "
                    "[{estim}]"
                ).format(
                    lr=optimizer.param_groups[0]["lr"],
                    epoch=epoch,
                    i=i,
                    n=len(train_loader),
                    bt=self.batch_time_t,
                    dt=self.data_time_t,
                    rl=range_loss_meter,
                    rml=range_moving_loss_meter,
                    rmvl=range_movable_loss_meter,
                    ra=range_acc_meter,
                    ri=range_iou_meter,
                    bl=bev_loss_meter,
                    bml=bev_moving_loss_meter,
                    bmvl=bev_movable_loss_meter,
                    ba=bev_acc_meter,
                    bi=bev_iou_meter,
                    estim=self.calculate_estimate(epoch, i),
                )
                print(log_str)
                save_to_txtlog(self.logdir, "log.txt", log_str)
            scheduler.step()
        return (
            range_acc_meter.avg,
            range_iou_meter.avg,
            bev_acc_meter.avg,
            bev_iou_meter.avg,
            (range_loss_meter.avg + bev_loss_meter.avg),
            update_ratio_meter.avg,
        )

    def validate(self, val_loader, model, all_criterion, all_evaluator):
        (
            range_evaluator,
            range_movable_evaluator,
            bev_evaluator,
            bev_movable_evaluator,
        ) = all_evaluator
        criterion, movable_criterion = all_criterion

        range_loss_meter = AverageMeter()
        range_moving_loss_meter = AverageMeter()
        range_movable_loss_meter = AverageMeter()
        bev_loss_meter = AverageMeter()
        bev_moving_loss_meter = AverageMeter()
        bev_movable_loss_meter = AverageMeter()
        range_acc_meter = AverageMeter()
        range_iou_meter = AverageMeter()
        bev_acc_meter = AverageMeter()
        bev_iou_meter = AverageMeter()
        rand_imgs = []

        model.eval()
        range_evaluator.reset()
        range_movable_evaluator.reset()
        bev_evaluator.reset()
        bev_movable_evaluator.reset()

        with torch.no_grad():
            end = time.time()
            for i, (
                (proj_full, bev_full),  # range (13, H, W) | bev (13, H_bev, W_bev)
                (proj_labels, proj_movable_labels),  # range (H, W), (H, W)
                (bev_labels, bev_movable_labels),  # bev (H_bev, W_bev), (H_bev, W_bev)
                (
                    path_seq,
                    path_name,
                    unproj_n_points,
                ),  # ex. '08', '000123.npy', 122319
                (proj_x, proj_y),  # (150000, ), (150000, )
                (bev_proj_x, bev_proj_y),  # (150000, ), (150000, )
                (proj_range, bev_range),  # (H_range, W_range), (H_BEV, W_BEV)
                (unproj_range, bev_unproj_range),  # (150000, ), (150000, )
            ) in enumerate(tqdm(val_loader, desc="Validation:", ncols=80)):
                if self.gpu:
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
                range_moving, range_movable, bev_moving, bev_movable = model(
                    proj_full, bev_full
                )

                log_range = torch.log(range_moving.clamp(min=1e-8))
                log_bev = torch.log(bev_moving.clamp(min=1e-8))
                range_moving_loss = criterion(
                    log_range.double(), proj_labels
                ).float() + self.ls(range_moving, proj_labels.long())
                range_movable_loss = movable_criterion(
                    torch.log(range_movable.clamp(min=1e-8)).double(),
                    proj_movable_labels,
                ).float() + self.movable_ls(range_movable, proj_movable_labels.long())
                bev_moving_loss = criterion(
                    log_bev.double(), bev_labels
                ).float() + self.ls(bev_moving, bev_labels.long())
                bev_movable_loss = movable_criterion(
                    torch.log(bev_movable.clamp(min=1e-8)).double(), bev_movable_labels
                ).float() + self.movable_ls(bev_movable, bev_movable_labels.long())
                loss_total = (
                    range_moving_loss
                    + range_movable_loss
                    + bev_moving_loss
                    + bev_movable_loss
                )

                with torch.no_grad():
                    range_pred = range_moving.argmax(dim=1)
                    range_evaluator.addBatch(range_pred, proj_labels)
                    range_acc = range_evaluator.getacc()
                    range_jacc, range_class_jacc = range_evaluator.getIoU()

                    bev_pred = bev_moving.argmax(dim=1)
                    bev_evaluator.addBatch(bev_pred, bev_labels)
                    bev_acc = bev_evaluator.getacc()
                    bev_jacc, bev_class_jacc = bev_evaluator.getIoU()

                    range_movable_pred = range_movable.argmax(dim=1)
                    range_movable_evaluator.addBatch(
                        range_movable_pred, proj_movable_labels
                    )
                    range_movable_acc = range_movable_evaluator.getacc()
                    range_movable_jacc, _ = range_movable_evaluator.getIoU()

                    bev_movable_pred = bev_movable.argmax(dim=1)
                    bev_movable_evaluator.addBatch(bev_movable_pred, bev_movable_labels)
                    bev_movable_acc = bev_movable_evaluator.getacc()
                    bev_movable_jacc, _ = bev_movable_evaluator.getIoU()

                range_loss_meter.update(
                    (range_moving_loss + range_movable_loss).mean().item(),
                    proj_full.size(0),
                )
                range_moving_loss_meter.update(
                    range_moving_loss.mean().item(), proj_full.size(0)
                )
                range_movable_loss_meter.update(
                    range_movable_loss.mean().item(), proj_full.size(0)
                )
                bev_loss_meter.update(
                    (bev_moving_loss + bev_movable_loss).mean().item(), bev_full.size(0)
                )
                bev_moving_loss_meter.update(
                    bev_moving_loss.mean().item(), bev_full.size(0)
                )
                bev_movable_loss_meter.update(
                    bev_movable_loss.mean().item(), bev_full.size(0)
                )
                range_acc_meter.update(range_acc.item(), proj_full.size(0))
                range_iou_meter.update(range_jacc.item(), proj_full.size(0))
                bev_acc_meter.update(bev_acc.item(), bev_full.size(0))
                bev_iou_meter.update(bev_jacc.item(), bev_full.size(0))

                self.batch_time_e.update(time.time() - end)
                end = time.time()

            log_str = (
                "*" * 80 + "\nValidation set:\n"
                "Time avg per batch {bt.avg:.3f}\n"
                "RangeLoss avg {rl.avg:.4f}\n"
                "RangeMovingLoss avg {rml.avg:.4f}\n"
                "RangeMovableLoss avg {rmvl.avg:.4f}\n"
                "RangeAcc avg {ra.avg:.6f}\n"
                "RangeIoU avg {ri.avg:.6f}\n"
                "BevLoss avg {bl.avg:.4f}\n"
                "BevMovingLoss avg {bml.avg:.4f}\n"
                "BevMovableLoss avg {bmvl.avg:.4f}\n"
                "BevAcc avg {ba.avg:.6f}\n"
                "BevIoU avg {bi.avg:.6f}\n"
            ).format(
                bt=self.batch_time_e,
                rl=range_loss_meter,
                rml=range_moving_loss_meter,
                rmvl=range_movable_loss_meter,
                ra=range_acc_meter,
                ri=range_iou_meter,
                bl=bev_loss_meter,
                bml=bev_moving_loss_meter,
                bmvl=bev_movable_loss_meter,
                ba=bev_acc_meter,
                bi=bev_iou_meter,
            )
            print(log_str)
            save_to_txtlog(self.logdir, "log.txt", log_str)

            # 클래스별 IoU 출력 (Range 및 BEV 각각)
            for i, jacc in enumerate(range_class_jacc):
                self.info[
                    "valid_range_classes/" + self.parser.get_xentropy_class_string(i)
                ] = jacc
                line = f"Range IoU class {i} [{self.parser.get_xentropy_class_string(i)}] = {jacc:.6f}"
                print(line)
                save_to_txtlog(self.logdir, "log.txt", line)
            for i, jacc in enumerate(bev_class_jacc):
                self.info[
                    "valid_bev_classes/" + self.parser.get_xentropy_class_string(i)
                ] = jacc
                line = f"Bev IoU class {i} [{self.parser.get_xentropy_class_string(i)}] = {jacc:.6f}"
                print(line)
                save_to_txtlog(self.logdir, "log.txt", line)
            print("*" * 80)
        return (
            range_acc_meter.avg,
            range_iou_meter.avg,
            bev_acc_meter.avg,
            bev_iou_meter.avg,
            (range_loss_meter.avg + bev_loss_meter.avg),
            rand_imgs,
        )

    def update_training_info(
        self, epoch, range_acc, range_iou, bev_acc, bev_iou, loss_total, update_mean
    ):
        self.info["train_update"] = update_mean
        self.info["train_loss"] = loss_total
        self.info["train_range_acc"] = range_acc
        self.info["train_range_iou"] = range_iou
        self.info["train_bev_acc"] = bev_acc
        self.info["train_bev_iou"] = bev_iou

        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "info": self.info,
            "scheduler": self.scheduler.state_dict(),
        }
        save_checkpoint(state, self.logdir, suffix="")
        if (
            range_iou > self.info["best_train_range_iou"]
            or bev_iou > self.info["best_train_bev_iou"]
        ):
            print("Best mean IoU in training so far, saving model!")
            self.info["best_train_range_iou"] = max(
                range_iou, self.info["best_train_range_iou"]
            )
            self.info["best_train_bev_iou"] = max(
                bev_iou, self.info["best_train_bev_iou"]
            )
            save_checkpoint(state, self.logdir, suffix="_train_best")

    def update_validation_info(
        self, epoch, range_acc, range_iou, bev_acc, bev_iou, loss_val
    ):
        self.info["valid_loss"] = loss_val
        self.info["valid_range_acc"] = range_acc
        self.info["valid_range_iou"] = range_iou
        self.info["valid_bev_acc"] = bev_acc
        self.info["valid_bev_iou"] = bev_iou
        if (
            range_iou > self.info["best_val_range_iou"]
            or bev_iou > self.info["best_val_bev_iou"]
        ):
            line = "Best mean IoU in validation so far, saving model!\n" + "*" * 80
            print(line)
            save_to_txtlog(self.logdir, "log.txt", line)
            self.info["best_val_range_iou"] = max(
                range_iou, self.info["best_val_range_iou"]
            )
            self.info["best_val_bev_iou"] = max(bev_iou, self.info["best_val_bev_iou"])
            state = {
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "info": self.info,
                "scheduler": self.scheduler.state_dict(),
            }
            save_checkpoint(state, self.logdir, suffix="_valid_best")
            save_checkpoint(state, self.logdir, suffix=f"_valid_best_{epoch}")
