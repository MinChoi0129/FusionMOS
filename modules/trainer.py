#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
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

# from modules.SalsaNextWithMotionAttention import SalsaNextWithMotionAttention
from modules.MFMOS import MFMOS

from modules.loss.Lovasz_Softmax import Lovasz_softmax, Lovasz_softmax_PointCloud
from modules.tools import (
    AverageMeter,
    iouEval,
    save_checkpoint,
    show_scans_in_training,
    save_to_txtlog,
    make_log_img,
)

from torch import distributed as dist


class Trainer:
    def __init__(
        self, ARCH, DATA, datadir, logdir, path=None, point_refine=False, local_rank=0
    ):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.logdir = logdir
        self.path = path
        self.epoch = 0
        self.point_refine = point_refine
        self.local_rank = local_rank

        self.batch_time_t = AverageMeter()
        self.data_time_t = AverageMeter()
        self.batch_time_e = AverageMeter()

        # put logger where it belongs
        self.tb_logger = Logger(self.logdir + "/tb")
        self.info = {
            "train_update": 0,
            "train_loss": 0,
            "train_acc": 0,
            "train_iou": 0,
            "valid_loss": 0,
            "valid_acc": 0,
            "valid_iou": 0,
            "best_train_iou": 0,
            "best_val_iou": 0,
        }

        # get the data
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

        with torch.no_grad():
            self.model = MFMOS(
                nclasses=self.parser.get_n_classes(),
                movable_nclasses=self.parser.get_n_classes(movable=True),
                params=self.ARCH,
            )

        self.set_gpu_cuda()
        self.set_loss_function(point_refine)
        self.set_optim_scheduler()

        # if need load the pre-trained model from checkpoint
        if self.path is not None:
            self.load_pretrained_model()

    def set_loss_weight(self):
        """
        Used to calculate the weights for each class
        weights for loss (and bias)
        """
        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.parser.get_n_classes(), dtype=torch.float)
        movable_content = torch.zeros(
            self.parser.get_n_classes(movable=True), dtype=torch.float
        )
        for cl, freq in self.DATA["content"].items():
            # map actual class to xentropy class
            x_cl = self.parser.to_xentropy(cl)
            content[x_cl] += freq

            movable_x_cl = self.parser.to_xentropy(cl, movable=True)
            movable_content[movable_x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)  # get weights
        self.movable_loss_w = 1 / (movable_content + epsilon_w)  # get weights
        # ignore the ones necessary to ignore
        for x_cl, w in enumerate(self.loss_w):
            if self.DATA["learning_ignore"][x_cl]:  # don't weigh
                self.loss_w[x_cl] = 0

        # ignore the ones necessary to ignore
        for x_cl, w in enumerate(self.movable_loss_w):
            if self.DATA["learning_ignore"][x_cl]:  # don't weigh
                self.movable_loss_w[x_cl] = 0
        print("Loss weights from content: ", self.loss_w.data)
        print("Movable Loss weights from content: ", self.movable_loss_w.data)

    def set_loss_function(self, point_refine):
        """
        Used to define the loss function, multiple gpus need to be parallel
        # self.dice = DiceLoss().to(self.device)
        # self.dice = nn.DataParallel(self.dice).cuda()
        """
        self.criterion = nn.NLLLoss(weight=self.loss_w.double()).cuda()
        self.movable_criterion = nn.NLLLoss(weight=self.movable_loss_w.double()).cuda()
        if not point_refine:
            # self.ls = Lovasz_softmax(ignore=0).to(self.device)
            self.ls = Lovasz_softmax(ignore=0).cuda()
            self.movable_ls = Lovasz_softmax(ignore=0).cuda()
        else:
            self.ls = Lovasz_softmax_PointCloud(ignore=0).to(self.device)

        # loss as dataparallel too (more images in batch)
        # if self.n_gpus > 1:
        # self.criterion = nn.DataParallel(self.criterion).cuda()  # spread in gpus
        # self.movable_criterion = nn.DataParallel(self.movable_criterion).cuda()  # spread in gpus
        # self.ls = nn.DataParallel(self.ls).cuda()
        # print()

    def set_gpu_cuda(self):
        """
        Used to set gpus and cuda information
        """
        self.gpu = False
        self.multi_gpu = False
        self.n_gpus = 0
        self.model_single = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Training in device: ", self.device)

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            # cudnn.benchmark = True
            # cudnn.fastest = True
            self.gpu = True
            self.n_gpus = 1
            self.model.cuda()

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            print(f"Let's use {torch.cuda.device_count()} GPUs!")
            # self.model = nn.DataParallel(self.model)      # spread in gpus
            self.model = convert_model(self.model).cuda()  # sync batchnorm
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,
                broadcast_buffers=False,
            )

            # single model to get weight names
            self.model_single = self.model.module
            self.multi_gpu = True
            self.n_gpus = torch.cuda.device_count()

    def set_optim_scheduler(self):
        """
        Used to set the optimizer and scheduler
        """
        self.optimizer = optim.SGD(
            [{"params": self.model.parameters()}],
            lr=self.ARCH["train"]["lr"],
            momentum=self.ARCH["train"]["momentum"],
            weight_decay=self.ARCH["train"]["w_decay"],
        )

        # Use warmup learning rate
        # post decay and step sizes come in epochs and we want it in steps
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
        If you want to resume training, reload the model
        """
        torch.nn.Module.dump_patches = True
        if not self.point_refine:
            checkpoint = "MFMOS"
            w_dict = torch.load(
                f"{self.path}/{checkpoint}", map_location=lambda storage, loc: storage
            )
            self.model.load_state_dict(w_dict["state_dict"], strict=True)
            self.optimizer.load_state_dict(w_dict["optimizer"])
            self.epoch = w_dict["epoch"] + 1
            self.scheduler.load_state_dict(w_dict["scheduler"])
            print("dict epoch:", w_dict["epoch"])
            self.info = w_dict["info"]
            print("info", w_dict["info"])
            print("load the pretrained model of MFMOS")
        else:
            checkpoint = "MFMOS_valid_best"
            w_dict = torch.load(
                f"{self.path}/{checkpoint}", map_location=lambda storage, loc: storage
            )
            # self.model.load_state_dict(w_dict['state_dict'], strict=True)
            self.model.load_state_dict(
                {k.replace("module.", ""): v for k, v in w_dict["state_dict"].items()}
            )
            self.optimizer.load_state_dict(w_dict["optimizer"])
            print("load the coarse model of MFMOS_valid_best")

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
            * (self.ARCH["train"]["max_epochs"] - (epoch))
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
        # save scalars
        for tag, value in info.items():
            if "valid_classes" in tag:
                continue  # solve the bug of saving tensor type of value
            logger.add_scalar(tag, value, epoch)

        # save summaries of weights and biases
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
                name = os.path.join(directory, str(i) + ".png")
                cv2.imwrite(name, img)

    def init_evaluator(self):
        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
                print("Ignoring class ", i, " in IoU evaluation")
        # self.evaluator = iouEval(self.parser.get_n_classes(),
        #                          self.device, self.ignore_class)
        # self.movable_evaluator = iouEval(self.parser.get_n_classes(),
        #                                  self.device, self.ignore_class)
        self.evaluator = iouEval(
            self.parser.get_n_classes(), self.local_rank, self.ignore_class
        )
        self.movable_evaluator = iouEval(
            self.parser.get_n_classes(movable=True), self.local_rank, self.ignore_class
        )

    def train(self):

        self.init_evaluator()

        # train for n epochs
        print(
            "============================================================================"
        )
        print("학습을 시작합니다.")
        for epoch in range(self.epoch, self.ARCH["train"]["max_epochs"]):
            # self.parser.train_sampler.set_epoch(epoch)
            # train for 1 epoch
            acc, iou, loss, update_mean, hetero_l = self.train_epoch(
                train_loader=self.parser.get_train_set(),
                model=self.model,
                all_criterion=(self.criterion, self.movable_criterion),
                optimizer=self.optimizer,
                epoch=epoch,
                all_evaluator=(self.evaluator, self.movable_evaluator),
                scheduler=self.scheduler,
                color_fn=self.parser.to_color,
                report=self.ARCH["train"]["report_batch"],
                show_scans=self.ARCH["train"]["show_scans"],
            )

            if self.local_rank == 0:
                # update the info dict and save the training checkpoint
                self.update_training_info(epoch, acc, iou, loss, update_mean, hetero_l)

            # evaluate on validation set
            if epoch % self.ARCH["train"]["report_epoch"] == 0 and self.local_rank == 0:
                acc, iou, loss, rand_img, hetero_l = self.validate(
                    val_loader=self.parser.get_valid_set(),
                    model=self.model,
                    all_criterion=(self.criterion, self.movable_criterion),
                    all_evaluator=(self.evaluator, self.movable_evaluator),
                    class_func=self.parser.get_xentropy_class_string,
                    color_fn=self.parser.to_color,
                    save_scans=self.ARCH["train"]["save_scans"],
                )

                self.update_validation_info(epoch, acc, iou, loss, hetero_l)

            if self.local_rank == 0:
                # save to tensorboard log
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

            dist.barrier()

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
        color_fn,
        report=10,
        show_scans=False,
    ):
        losses = AverageMeter()

        moving_losses = AverageMeter()
        movable_losses = AverageMeter()

        acc = AverageMeter()
        iou = AverageMeter()

        movable_acc = AverageMeter()
        movable_iou = AverageMeter()

        hetero_l = AverageMeter()
        update_ratio_meter = AverageMeter()

        evaluator, movable_evaluator = all_evaluator
        criterion, movable_criterion = all_criterion

        # empty the cache to train now
        # if self.gpu:
        #     torch.cuda.empty_cache()

        # switch to train mode
        model.train()
        end = time.time()
        for i, (
            proj_full,  # range (5 + n_input_scans, H, W)
            bev_full,  # bev   (5 + n_input_scans, H_bev, W_bev)
            proj_labels,  # range labels (H, W)
            proj_movable_labels,  # range movable (H, W)
            bev_labels,  # bev   labels  (H_bev, W_bev)
            bev_movable_labels,  # bev   movable (H_bev, W_bev)
        ) in enumerate(train_loader):

            ###################################################################################
            import matplotlib.pyplot as plt

            def do_image_debug(
                proj_full,
                proj_labels,
                proj_movable_labels,
                bev_full,
                bev_labels,
                bev_movable_labels,
            ):
                print(
                    "Shape :",
                    proj_full[0][0].shape,
                    torch.min(proj_full[0][0]),
                    torch.max(proj_full[0][0]),
                )
                plt.imsave(
                    "/home/work_docker/MF-MOS/debug/range_image.png",
                    proj_full[0][0].cpu().numpy(),
                )

                print(
                    "Shape :",
                    proj_full[0][4].shape,
                    torch.min(proj_full[0][4]),
                    torch.max(proj_full[0][4]),
                )
                plt.imsave(
                    "/home/work_docker/MF-MOS/debug/range_remission.png",
                    proj_full[0][4].cpu().numpy(),
                )

                print(
                    "Shape :",
                    proj_full[0][5].shape,
                    torch.min(proj_full[0][5]),
                    torch.max(proj_full[0][5]),
                )
                plt.imsave(
                    "/home/work_docker/MF-MOS/debug/range_residual_1.png",
                    proj_full[0][5].cpu().numpy(),
                )

                print(
                    "Shape :",
                    proj_labels[0].shape,
                    torch.min(proj_labels[0]),
                    torch.max(proj_labels[0]),
                )
                plt.imsave(
                    "/home/work_docker/MF-MOS/debug/range_moving_label.png",
                    proj_labels[0].cpu().numpy(),
                )

                print(
                    "Shape :",
                    proj_movable_labels[0].shape,
                    torch.min(proj_movable_labels[0]),
                    torch.max(proj_movable_labels[0]),
                )
                plt.imsave(
                    "/home/work_docker/MF-MOS/debug/range_movable_label.png",
                    proj_movable_labels[0].cpu().numpy(),
                )

                print(
                    "Shape :",
                    bev_full[0][0].shape,
                    torch.min(bev_full[0][0]),
                    torch.max(bev_full[0][0]),
                )
                plt.imsave(
                    "/home/work_docker/MF-MOS/debug/bev_image.png",
                    bev_full[0][0].cpu().numpy(),
                )

                print(
                    "Shape :",
                    bev_full[0][4].shape,
                    torch.min(bev_full[0][4]),
                    torch.max(bev_full[0][4]),
                )
                plt.imsave(
                    "/home/work_docker/MF-MOS/debug/bev_remission.png",
                    bev_full[0][4].cpu().numpy(),
                )

                print(
                    "Shape :",
                    bev_full[0][5].shape,
                    torch.min(bev_full[0][5]),
                    torch.max(bev_full[0][5]),
                )
                plt.imsave(
                    "/home/work_docker/MF-MOS/debug/bev_residual_1.png",
                    bev_full[0][5].cpu().numpy(),
                )

                print(
                    "Shape :",
                    bev_labels[0].shape,
                    torch.min(bev_labels[0]),
                    torch.max(bev_labels[0]),
                )
                plt.imsave(
                    "/home/work_docker/MF-MOS/debug/bev_moving_label.png",
                    bev_labels[0].cpu().numpy(),
                )

                print(
                    "Shape :",
                    bev_movable_labels[0].shape,
                    torch.min(bev_movable_labels[0]),
                    torch.max(bev_movable_labels[0]),
                )
                plt.imsave(
                    "/home/work_docker/MF-MOS/debug/bev_movable_label.png",
                    bev_movable_labels[0].cpu().numpy(),
                )

            do_image_debug(
                proj_full,
                proj_labels,
                proj_movable_labels,
                bev_full,
                bev_labels,
                bev_movable_labels,
            )

            ###################################################################################

            # 1. Data loading time
            self.data_time_t.update(time.time() - end)

            if not self.multi_gpu and self.gpu:
                proj_full = proj_full.cuda()
                bev_full = bev_full.cuda()
            if self.gpu:
                proj_labels = proj_labels.cuda().long()
                proj_movable_labels = proj_movable_labels.cuda().long()
                bev_labels = bev_labels.cuda().long()
                bev_movable_labels = bev_movable_labels.cuda().long()

            (
                movable_range_logits,
                motion_range_logits,
                movable_bev_logits,
                motion_bev_logits,
            ) = model(proj_full, bev_full)

            moving_loss_m = criterion(
                torch.log(motion_range_logits.clamp(min=1e-8)).double(), proj_labels
            ).float() + self.ls(motion_range_logits, proj_labels.long())
            movable_loss_m = movable_criterion(
                torch.log(movable_range_logits.clamp(min=1e-8)).double(),
                proj_movable_labels,
            ).float() + self.movable_ls(
                movable_range_logits, proj_movable_labels.long()
            )

            bev_moving_loss_m = criterion(
                torch.log(motion_bev_logits.clamp(min=1e-8)).double(), bev_labels
            ).float() + self.ls(motion_bev_logits, bev_labels.long())
            bev_movable_loss_m = movable_criterion(
                torch.log(movable_bev_logits.clamp(min=1e-8)).double(),
                bev_movable_labels,
            ).float() + self.movable_ls(movable_bev_logits, bev_movable_labels.long())

            loss_range = moving_loss_m + movable_loss_m
            loss_bev = bev_moving_loss_m + bev_movable_loss_m
            # 최종 Loss: 두 branch의 평균
            loss_m = (loss_range + loss_bev) / 2

            optimizer.zero_grad()
            loss_m.backward()
            optimizer.step()

            loss_val = loss_m.mean()
            losses.update(loss_val.item(), proj_full.size(0))

            # 4) 2D Accuracy 및 IoU 계산: 각 branch에 대해 별도 계산 후 평균
            # Range branch 평가 (예: motion_range_logits를 기준)
            evaluator.reset()
            range_pred = motion_range_logits.argmax(dim=1)
            evaluator.addBatch(range_pred, proj_labels)
            range_acc = evaluator.getacc()
            range_iou, _ = evaluator.getIoU()

            # BEV branch 평가 (예: motion_bev_logits를 기준)
            evaluator.reset()
            bev_pred = motion_bev_logits.argmax(dim=1)
            evaluator.addBatch(bev_pred, bev_labels)
            bev_acc = evaluator.getacc()
            bev_iou, _ = evaluator.getIoU()

            final_acc = (range_acc + bev_acc) / 2.0
            final_iou = (range_iou + bev_iou) / 2.0

            acc.update(final_acc, proj_full.size(0))
            iou.update(final_iou, proj_full.size(0))

            # 5) Gradient Update Ratio 계산 (기존 그대로)
            update_ratios = []
            for g in self.optimizer.param_groups:
                lr = g["lr"]
                for value in g["params"]:
                    if value.grad is not None:
                        w = np.linalg.norm(value.data.cpu().numpy().reshape((-1)))
                        update = np.linalg.norm(
                            -max(lr, 1e-10) * value.grad.cpu().numpy().reshape((-1))
                        )
                        update_ratios.append(update / max(w, 1e-10))
            update_ratios = np.array(update_ratios)
            update_mean = update_ratios.mean()
            update_std = update_ratios.std()
            update_ratio_meter.update(update_mean)

            # 6) hetero_l 업데이트 (별도 heterogeneity loss 계산이 있다면 업데이트; 없으면 그대로 유지)
            hetero_l.update(hetero_l.val if hasattr(hetero_l, "val") else 0.0)

            self.batch_time_t.update(time.time() - end)
            end = time.time()

            if i % self.ARCH["train"]["report_batch"] == 0 and self.local_rank == 0:
                str_line = (
                    "Lr: {lr:.3e} | Update: {umean:.3e} mean, {ustd:.3e} std | "
                    "Epoch: [{0}][{1}/{2}] | Time {batch_time.val:.3f} ({batch_time.avg:.3f}) | "
                    "Data {data_time.val:.3f} ({data_time.avg:.3f}) | Loss {loss.val:.4f} ({loss.avg:.4f}) | "
                    "Acc {acc.val:.3f} ({acc.avg:.3f}) | IoU {iou.val:.3f} ({iou.avg:.3f}) | "
                    "[{estim}]"
                ).format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=self.batch_time_t,
                    data_time=self.data_time_t,
                    loss=losses,
                    acc=acc,
                    iou=iou,
                    lr=lr,
                    umean=update_mean,
                    ustd=update_std,
                    estim=self.calculate_estimate(epoch, i),
                )
                print(str_line)
                save_to_txtlog(self.logdir, "log.txt", str_line)

            scheduler.step()

        return acc.avg, iou.avg, losses.avg, update_ratio_meter.avg, hetero_l.avg

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
        evaluator, movable_evaluator = all_evaluator
        criterion, movable_criterion = all_criterion

        losses = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        hetero_l = AverageMeter()
        rand_imgs = []

        model.eval()
        evaluator.reset()
        movable_evaluator.reset()
        with torch.no_grad():
            end = time.time()
            for i, (
                proj_full,
                bev_full,
                proj_labels,
                proj_movable_labels,
                bev_labels,
                bev_movable_labels,
            ) in enumerate(val_loader):
                # 데이터 로딩 시간 측정
                self.data_time_t.update(time.time() - end)

                # GPU 전송 (train_epoch와 동일)
                if not self.multi_gpu and self.gpu:
                    proj_full = proj_full.cuda()
                    bev_full = bev_full.cuda()
                if self.gpu:
                    proj_labels = proj_labels.cuda(non_blocking=True).long()
                    proj_movable_labels = proj_movable_labels.cuda(
                        non_blocking=True
                    ).long()
                    bev_labels = bev_labels.cuda(non_blocking=True).long()
                    bev_movable_labels = bev_movable_labels.cuda(
                        non_blocking=True
                    ).long()

                # 모델 예측
                (
                    movable_range_logits,
                    motion_range_logits,
                    movable_bev_logits,
                    motion_bev_logits,
                ) = model(proj_full, bev_full)

                # Loss 계산 (각 branch별 계산 후 평균)
                moving_loss_m = criterion(
                    torch.log(motion_range_logits.clamp(min=1e-8)).double(), proj_labels
                ).float() + self.ls(motion_range_logits, proj_labels.long())
                movable_loss_m = movable_criterion(
                    torch.log(movable_range_logits.clamp(min=1e-8)).double(),
                    proj_movable_labels,
                ).float() + self.movable_ls(
                    movable_range_logits, proj_movable_labels.long()
                )
                bev_moving_loss_m = criterion(
                    torch.log(motion_bev_logits.clamp(min=1e-8)).double(), bev_labels
                ).float() + self.ls(motion_bev_logits, bev_labels.long())
                bev_movable_loss_m = movable_criterion(
                    torch.log(movable_bev_logits.clamp(min=1e-8)).double(),
                    bev_movable_labels,
                ).float() + self.movable_ls(
                    movable_bev_logits, bev_movable_labels.long()
                )

                loss_range = moving_loss_m + movable_loss_m
                loss_bev = bev_moving_loss_m + bev_movable_loss_m
                loss_m = (loss_range + loss_bev) / 2.0
                losses.update(loss_m.mean().item(), proj_full.size(0))

                # 평가 지표 계산: Range branch와 BEV branch 각각 평가 후 평균 산출
                # Range branch 평가
                evaluator.reset()
                range_pred = motion_range_logits.argmax(dim=1)
                evaluator.addBatch(range_pred, proj_labels)
                range_acc = evaluator.getacc()
                range_iou, _ = evaluator.getIoU()

                # BEV branch 평가
                evaluator.reset()
                bev_pred = motion_bev_logits.argmax(dim=1)
                evaluator.addBatch(bev_pred, bev_labels)
                bev_acc = evaluator.getacc()
                bev_iou, _ = evaluator.getIoU()

                final_acc = (range_acc + bev_acc) / 2.0
                final_iou = (range_iou + bev_iou) / 2.0

                acc.update(final_acc, proj_full.size(0))
                iou.update(final_iou, proj_full.size(0))

                # 스캔 이미지 저장 (옵션)
                if save_scans:
                    img = make_log_img(
                        None,
                        None,
                        range_pred[0].cpu().numpy(),
                        proj_labels[0].cpu().numpy(),
                        color_fn,
                    )
                    rand_imgs.append(img)

                self.batch_time_e.update(time.time() - end)
                end = time.time()

            # 전체 평가 결과 및 클래스별 IoU 출력
            overall_acc = acc.avg
            overall_iou = iou.avg
            # evaluator와 movable_evaluator는 마지막 배치의 결과를 포함하므로,
            # 클래스별 IoU는 해당 evaluator로부터 얻습니다.
            _, class_jaccard = evaluator.getIoU()
            _, movable_class_jaccard = movable_evaluator.getIoU()

            str_line = (
                "*" * 80
                + "\n"
                + "Validation set:\n"
                + f"Time avg per batch: {self.batch_time_e.avg:.3f}\n"
                + f"Loss avg: {losses.avg:.4f}\n"
                + f"Acc avg: {overall_acc:.3f}\n"
                + f"IoU avg: {overall_iou:.3f}\n"
                + "*" * 80
            )
            print(str_line)
            save_to_txtlog(self.logdir, "log.txt", str_line)

            for i, jacc in enumerate(class_jaccard):
                self.info["valid_classes/" + class_func(i)] = jacc
                line = f"IoU class {i} [{class_func(i)}] = {jacc:.6f}"
                print(line)
                save_to_txtlog(self.logdir, "log.txt", line)

            for i, jacc in enumerate(movable_class_jaccard):
                self.info["valid_classes/" + class_func(i, movable=True)] = jacc
                line = f"IoU class {i} [{class_func(i, movable=True)}] = {jacc:.6f}"
                print(line)
                save_to_txtlog(self.logdir, "log.txt", line)

            print("*" * 80)
            save_to_txtlog(self.logdir, "log.txt", "*" * 80)

        return acc.avg, iou.avg, losses.avg, rand_imgs, hetero_l.avg

    def update_training_info(self, epoch, acc, iou, loss, update_mean, hetero_l):
        # update info
        self.info["train_update"] = update_mean
        self.info["train_loss"] = loss
        self.info["train_acc"] = acc
        self.info["train_iou"] = iou
        self.info["train_hetero"] = hetero_l

        # remember best iou and save checkpoint
        state = {
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "info": self.info,
            "scheduler": self.scheduler.state_dict(),
        }
        save_checkpoint(state, self.logdir, suffix="")

        if self.info["train_iou"] > self.info["best_train_iou"]:
            print("Best mean iou in training set so far, save model!")
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
        # update info
        self.info["valid_loss"] = loss
        self.info["valid_acc"] = acc
        self.info["valid_iou"] = iou
        self.info["valid_heteros"] = hetero_l

        # remember best iou and save checkpoint
        if self.info["valid_iou"] > self.info["best_val_iou"]:
            str_line = "Best mean iou in validation so far, save model!\n" + "*" * 80
            print(str_line)
            save_to_txtlog(self.logdir, "log.txt", str_line)
            self.info["best_val_iou"] = self.info["valid_iou"]

            # save the weights!
            state = {
                "epoch": epoch,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "info": self.info,
                "scheduler": self.scheduler.state_dict(),
            }
            save_checkpoint(state, self.logdir, suffix="_valid_best")
            save_checkpoint(state, self.logdir, suffix=f"_valid_best_{epoch}")
