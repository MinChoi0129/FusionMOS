#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import os
import imp
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import __init__ as booger

from tqdm import tqdm
from modules.KNN import KNN

# from modules.SalsaNextWithMotionAttention import *
from modules.MFMOS import *

from modules.PointRefine.spvcnn import SPVCNN

# from modules.PointRefine.spvcnn_lite import SPVCNN


class User:
    def __init__(
        self,
        ARCH,
        DATA,
        datadir,
        outputdir,
        modeldir,
        split,
        point_refine=False,
        save_movable=False,
    ):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.outputdir = outputdir
        self.modeldir = modeldir
        self.split = split
        self.post = None
        self.infer_batch_size = 1
        self.point_refine = point_refine
        self.save_movable = save_movable
        # get the data
        parserModule = imp.load_source(
            "parserModule",
            f"{booger.TRAIN_PATH}/common/dataset/{self.DATA['name']}/parser.py",
        )
        self.parser = parserModule.Parser(
            root=self.datadir,
            train_sequences=self.DATA["split"]["train"],
            valid_sequences=self.DATA["split"]["valid"],
            test_sequences=self.DATA["split"]["test"],
            split=self.split,
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
            batch_size=self.infer_batch_size,
            workers=2,  # self.ARCH["train"]["workers"],
            gt=True,
            shuffle_train=False,
        )

        with torch.no_grad():
            torch.nn.Module.dump_patches = True
            if not point_refine:
                # 모델 초기화 및 DataParallel 래핑
                self.model = MFMOS(
                    nclasses=self.parser.get_n_classes(),
                    movable_nclasses=self.parser.get_n_classes(movable=True),
                    params=ARCH,
                    num_batch=self.infer_batch_size,
                )
                self.model = nn.DataParallel(self.model)

                checkpoint = "MFMOS_valid_best"
                w_dict = torch.load(
                    f"{self.modeldir}/{checkpoint}",
                    map_location=lambda storage, loc: storage,
                )

                # 체크포인트의 state_dict 키에 'module.' 접두어가 없는 경우 붙여주기
                state_dict = w_dict["state_dict"]
                new_state_dict = {}
                for k, v in state_dict.items():
                    # 만약 키가 이미 'module.'로 시작하지 않는다면 접두어 추가
                    if not k.startswith("module."):
                        new_state_dict[f"module.{k}"] = v
                    else:
                        new_state_dict[k] = v

                # 수정된 state_dict로 모델에 로드
                self.model.load_state_dict(new_state_dict, strict=True)

                self.set_knn_post()
            else:
                self.model = MFMOS(
                    nclasses=self.parser.get_n_classes(),
                    movable_nclasses=self.parser.get_n_classes(movable=True),
                    params=ARCH,
                    num_batch=self.infer_batch_size,
                )
                self.model = nn.DataParallel(self.model)
                checkpoint = "MFMOS_SIEM_valid_best"
                w_dict = torch.load(
                    f"{self.modeldir}/{checkpoint}",
                    map_location=lambda storage, loc: storage,
                )
                self.model.load_state_dict(
                    {f"module.{k}": v for k, v in w_dict["main_state_dict"].items()},
                    strict=True,
                )

                net_config = {
                    "num_classes": self.parser.get_n_classes(),
                    "cr": 1.0,
                    "pres": 0.05,
                    "vres": 0.05,
                }
                self.refine_module = SPVCNN(
                    num_classes=net_config["num_classes"],
                    cr=net_config["cr"],
                    pres=net_config["pres"],
                    vres=net_config["vres"],
                )
                self.refine_module = nn.DataParallel(self.refine_module)
                w_dict = torch.load(
                    f"{modeldir}/{checkpoint}",
                    map_location=lambda storage, loc: storage,
                )
                self.refine_module.load_state_dict(
                    {f"module.{k}": v for k, v in w_dict["refine_state_dict"].items()},
                    strict=True,
                )

        self.set_gpu_cuda()

    def set_knn_post(self):
        # use knn post processing?
        if self.ARCH["post"]["KNN"]["use"]:
            self.post = KNN(
                self.ARCH["post"]["KNN"]["params"], self.parser.get_n_classes()
            )

    def set_gpu_cuda(self):
        # GPU?
        self.gpu = False
        self.model_single = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Infering in device: ", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.model.cuda()
            if self.point_refine:
                self.refine_module.cuda()

    def infer(self):
        cnn, knn = [], []

        if self.split == "valid":
            self.infer_subset(
                loader=self.parser.get_valid_set(),
                to_orig_fn=self.parser.to_original,
                cnn=cnn,
                knn=knn,
            )
        elif self.split == "train":
            self.infer_subset(
                loader=self.parser.get_train_set(),
                to_orig_fn=self.parser.to_original,
                cnn=cnn,
                knn=knn,
            )
        elif self.split == "test":
            self.infer_subset(
                loader=self.parser.get_test_set(),
                to_orig_fn=self.parser.to_original,
                cnn=cnn,
                knn=knn,
            )
        elif self.split == None:
            self.infer_subset(
                loader=self.parser.get_train_set(),
                to_orig_fn=self.parser.to_original,
                cnn=cnn,
                knn=knn,
            )
            # do valid set
            self.infer_subset(
                loader=self.parser.get_valid_set(),
                to_orig_fn=self.parser.to_original,
                cnn=cnn,
                knn=knn,
            )
            # do test set
            self.infer_subset(
                loader=self.parser.get_test_set(),
                to_orig_fn=self.parser.to_original,
                cnn=cnn,
                knn=knn,
            )
        else:
            raise NotImplementedError

        print(
            f"Mean CNN inference time:{'%.8f'%np.mean(cnn)}\t std:{'%.8f'%np.std(cnn)}"
        )
        print(
            f"Mean KNN inference time:{'%.8f'%np.mean(knn)}\t std:{'%.8f'%np.std(knn)}"
        )
        print(f"Total Frames: {len(cnn)}")
        print("Finished Infering")

        return

    def infer_subset(self, loader, to_orig_fn, cnn, knn):

        # 평가 모드로 전환
        self.model.eval()

        # GPU 사용 시 캐시 비우기
        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():
            end = time.time()

            for i, (
                (proj_full, bev_full),  # (13, H, W) / (13, H_bev, W_bev)
                (proj_labels, proj_movable_labels),  # (H, W), (H, W)
                (bev_labels, bev_movable_labels),  # (H_bev, W_bev), (H_bev, W_bev)
                (path_seq, path_name, npoints),  # ex. '08', '000123.npy', 122319
                (proj_x, proj_y),  # (150000, ), (150000, )
                (bev_proj_x, bev_proj_y),  # (150000, ), (150000, )
                (proj_range, bev_range),  # (H_range, W_range), (H_BEV, W_BEV)
                (unproj_range, bev_unproj_range),  # (150000, ), (150000, )
            ) in enumerate(tqdm(loader, ncols=80)):

                # 데이터 자르기 (배치 크기가 1일 경우)
                proj_x = proj_x[0, :npoints]
                proj_y = proj_y[0, :npoints]
                proj_range = proj_range[0, :npoints]
                unproj_range = unproj_range[0, :npoints]

                bev_proj_x = bev_proj_x[0, :npoints]
                bev_proj_y = bev_proj_y[0, :npoints]
                bev_range = bev_range[0, :npoints]
                bev_unproj_range = bev_unproj_range[0, :npoints]

                path_seq = path_seq[0]
                path_name = path_name[0]

                if self.gpu:
                    proj_full = proj_full.cuda()
                    bev_full = bev_full.cuda()
                    proj_x = proj_x.cuda()
                    proj_y = proj_y.cuda()
                    bev_proj_x = bev_proj_x.cuda()
                    bev_proj_y = bev_proj_y.cuda()
                    if self.post:
                        proj_range = proj_range.cuda()
                        unproj_range = unproj_range.cuda()
                        bev_range = bev_range.cuda()
                        bev_unproj_range = bev_unproj_range.cuda()

                end = time.time()
                # 모델 추론: 두 뷰에 대한 출력
                range_moving, range_movable, bev_moving, bev_movable = self.model(
                    proj_full, bev_full
                )

                import matplotlib.pyplot as plt

                print(
                    range_moving.cpu().numpy().shape,
                    np.min(range_moving.cpu().numpy()),
                    np.max(range_moving.cpu().numpy()),
                )
                print(
                    range_movable.cpu().numpy().shape,
                    np.min(range_movable.cpu().numpy()),
                    np.max(range_movable.cpu().numpy()),
                )
                print(
                    bev_moving.cpu().numpy().shape,
                    np.min(bev_moving.cpu().numpy()),
                    np.max(bev_moving.cpu().numpy()),
                )
                print(
                    bev_movable.cpu().numpy().shape,
                    np.min(bev_movable.cpu().numpy()),
                    np.max(bev_movable.cpu().numpy()),
                )

                plt.imsave(
                    "/home/work/MF-MOS/debug_img/range_moving.png",
                    range_moving[0].permute(1, 2, 0).cpu().numpy(),
                )
                plt.imsave(
                    "/home/work/MF-MOS/debug_img/range_movable.png",
                    range_movable[0].permute(1, 2, 0).cpu().numpy(),
                )
                plt.imsave(
                    "/home/work/MF-MOS/debug_img/bev_moving.png",
                    bev_moving[0].permute(1, 2, 0).cpu().numpy(),
                )
                plt.imsave(
                    "/home/work/MF-MOS/debug_img/bev_movable.png",
                    bev_movable[0].permute(1, 2, 0).cpu().numpy(),
                )

                print(
                    proj_labels.shape,
                    bev_labels.shape,
                    proj_movable_labels.shape,
                    bev_movable_labels.shape,
                )
                plt.imsave(
                    "/home/work/MF-MOS/debug_img/proj_labels.png",
                    proj_labels[0].cpu().numpy(),
                )
                plt.imsave(
                    "/home/work/MF-MOS/debug_img/bev_labels.png",
                    bev_labels[0].cpu().numpy(),
                )
                plt.imsave(
                    "/home/work/MF-MOS/debug_img/proj_movable_labels.png",
                    proj_movable_labels[0].cpu().numpy(),
                )
                plt.imsave(
                    "/home/work/MF-MOS/debug_img/bev_movable_labels.png",
                    bev_movable_labels[0].cpu().numpy(),
                )

                res = time.time() - end
                cnn.append(res)

                range_moving

                # ======= Fusion 추가 =======
                # 방법: 두 뷰의 확률을 단순 평균한 후, argmax 계산 (더 부드러운 결합)
                fused_prob = (range_moving[0] + bev_moving[0]) / 2.0
                fused_argmax = fused_prob.argmax(dim=0)

                fused_movable_prob = (range_movable[0] + bev_movable[0]) / 2.0
                fused_movable_argmax = fused_movable_prob.argmax(dim=0)
                # ============================

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.time()

                if self.post:
                    # 후처리: 각 뷰별로 좌표 맵핑
                    unproj_argmax = self.post(
                        proj_range, unproj_range, proj_argmax, proj_x, proj_y
                    )
                    bev_unproj_argmax = self.post(
                        bev_range, bev_unproj_range, bev_argmax, bev_proj_x, bev_proj_y
                    )
                    fused_unproj_argmax = self.post(
                        proj_range, unproj_range, fused_argmax, proj_x, proj_y
                    )
                    if self.save_movable:
                        movable_unproj_argmax = self.post(
                            proj_range,
                            unproj_range,
                            movable_proj_argmax,
                            proj_x,
                            proj_y,
                        )
                        bev_movable_unproj_argmax = self.post(
                            bev_range,
                            bev_unproj_range,
                            movable_bev_argmax,
                            bev_proj_x,
                            bev_proj_y,
                        )
                        fused_movable_unproj_argmax = self.post(
                            proj_range,
                            unproj_range,
                            fused_movable_argmax,
                            proj_x,
                            proj_y,
                        )
                else:
                    unproj_argmax = proj_argmax[proj_y, proj_x]
                    bev_unproj_argmax = bev_argmax[bev_proj_y, bev_proj_x]
                    fused_unproj_argmax = fused_argmax[proj_y, proj_x]
                    if self.save_movable:
                        movable_unproj_argmax = movable_proj_argmax[proj_y, proj_x]
                        bev_movable_unproj_argmax = movable_bev_argmax[
                            bev_proj_y, bev_proj_x
                        ]
                        fused_movable_unproj_argmax = fused_movable_argmax[
                            proj_y, proj_x
                        ]

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - end
                knn.append(res)

                # === 개별 예측 저장 (Projection 뷰) ===
                pred_np = unproj_argmax.cpu().numpy().reshape((-1)).astype(np.int32)
                pred_np = to_orig_fn(pred_np)
                path = os.path.join(
                    self.outputdir, "sequences", path_seq, "predictions", path_name
                )
                pred_np.tofile(path)

                # === Fusion 결과 저장 ===
                fused_pred_np = (
                    fused_unproj_argmax.cpu().numpy().reshape((-1)).astype(np.int32)
                )
                fused_pred_np = to_orig_fn(fused_pred_np)
                path = os.path.join(
                    self.outputdir, "sequences", path_seq, "predictions_fuse", path_name
                )
                fused_pred_np.tofile(path)

                if self.save_movable:
                    movable_pred_np = (
                        movable_unproj_argmax.cpu()
                        .numpy()
                        .reshape((-1))
                        .astype(np.int32)
                    )
                    movable_pred_np = to_orig_fn(movable_pred_np, movable=True)
                    path = os.path.join(
                        self.outputdir,
                        "sequences",
                        path_seq,
                        "predictions_movable",
                        path_name,
                    )
                    movable_pred_np.tofile(path)

                    fused_movable_pred_np = (
                        fused_movable_unproj_argmax.cpu()
                        .numpy()
                        .reshape((-1))
                        .astype(np.int32)
                    )
                    fused_movable_pred_np = to_orig_fn(
                        fused_movable_pred_np, movable=True
                    )
                    path = os.path.join(
                        self.outputdir,
                        "sequences",
                        path_seq,
                        "predictions_movable_fuse",
                        path_name,
                    )
                    fused_movable_pred_np.tofile(path)
