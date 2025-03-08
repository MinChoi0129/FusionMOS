import os
import numpy as np
import torch
from torch.utils.data import Dataset
from common.laserscan import SemLaserScan
from common.bev_laserscan import process_scan_as_bev
import torch
import random
import time
from collections.abc import Sequence, Iterable
from common.dataset.kitti.utils import load_poses, load_calib

# import math
# import types
# import numbers
# import warnings
# import torchvision
# from PIL import Image
# try:
# 	import accimage
# except ImportError:
# 	accimage = None

EXTENSIONS_SCAN = [".bin"]
EXTENSIONS_LABEL = [".label"]
EXTENSIONS_RESIDUAL = [".npy"]


def is_scan(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


def is_residual(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_RESIDUAL)


def my_collate(batch):
    data = [item[0] for item in batch]
    project_mask = [item[1] for item in batch]
    proj_labels = [item[2] for item in batch]
    data = torch.stack(data, dim=0)
    project_mask = torch.stack(project_mask, dim=0)
    proj_labels = torch.stack(proj_labels, dim=0)

    to_augment = (proj_labels == 12).nonzero()
    to_augment_unique_12 = torch.unique(to_augment[:, 0])

    to_augment = (proj_labels == 5).nonzero()
    to_augment_unique_5 = torch.unique(to_augment[:, 0])

    to_augment = (proj_labels == 8).nonzero()
    to_augment_unique_8 = torch.unique(to_augment[:, 0])

    to_augment_unique = torch.cat(
        (to_augment_unique_5, to_augment_unique_8, to_augment_unique_12), dim=0
    )
    to_augment_unique = torch.unique(to_augment_unique)

    for k in to_augment_unique:
        data = torch.cat((data, torch.flip(data[k.item()], [2]).unsqueeze(0)), dim=0)
        proj_labels = torch.cat(
            (proj_labels, torch.flip(proj_labels[k.item()], [1]).unsqueeze(0)), dim=0
        )
        project_mask = torch.cat(
            (project_mask, torch.flip(project_mask[k.item()], [1]).unsqueeze(0)), dim=0
        )

    return data, project_mask, proj_labels


class SemanticKitti(Dataset):

    def __init__(
        self,
        root,  # directory where data is
        sequences,  # sequences for this data (e.g. [1,3,4,6])
        labels,  # label dict: (e.g 10: "car")
        # use the data augmentation for residual maps (True or False)
        residual_aug,
        color_map,  # colors dict bgr (e.g 10: [255, 0, 0])
        learning_map,  # classes to learn (0 to N-1 for xentropy)
        movable_learning_map,
        learning_map_inv,  # inverse of previous (recover labels)
        movable_learning_map_inv,
        sensor,  # sensor to parse scans from
        valid_residual_delta_t=1,  # modulation interval in data augmentation fro residual maps
        max_points=150000,  # max number of points present in dataset
        gt=True,  # send ground truth?
        transform=False,
        drop_few_static_frames=False,
    ):
        # save deats
        self.root = os.path.join(root, "sequences")
        self.sequences = sequences
        self.labels = labels
        self.residual_aug = residual_aug
        self.color_map = color_map
        self.learning_map = learning_map
        self.movable_learning_map = movable_learning_map
        self.learning_map_inv = learning_map_inv
        self.movable_learning_map_inv = movable_learning_map_inv
        self.sensor = sensor
        self.sensor_img_H = sensor["img_prop"]["height"]
        self.sensor_img_W = sensor["img_prop"]["width"]
        self.sensor_img_means = torch.tensor(sensor["img_means"], dtype=torch.float)
        self.sensor_img_stds = torch.tensor(sensor["img_stds"], dtype=torch.float)
        self.sensor_fov_up = sensor["fov_up"]
        self.sensor_fov_down = sensor["fov_down"]
        self.valid_residual_delta_t = valid_residual_delta_t
        self.max_points = max_points
        self.gt = gt
        self.transform = transform
        print(
            "self.residual_aug {}, self.valid_residual_delta_t {}".format(
                self.residual_aug, self.valid_residual_delta_t
            )
        )
        """
        Added stuff for dynamic object segmentation
        """
        # dictionary for mapping a dataset index to a sequence, frame_id tuple needed for using multiple frames
        self.dataset_size = 0
        self.index_mapping = {}
        dataset_index = 0
        # added this for dynamic object removal
        self.n_input_scans = sensor[
            "n_input_scans"
        ]  # This needs to be the same as in arch_cfg.yaml!
        self.use_residual = sensor["residual"]
        self.transform_mod = sensor["transform"]
        self.use_normal = (
            sensor["use_normal"] if "use_normal" in sensor.keys() else False
        )
        """"""

        # get number of classes (can't be len(self.learning_map) because there
        # are multiple repeated entries, so the number that matters is how many
        # there are for the xentropy)
        self.nclasses = len(self.learning_map_inv)
        self.movable_nclasses = len(self.movable_learning_map_inv)

        # sanity checks

        # make sure directory exists
        if os.path.isdir(self.root):
            print("Sequences folder exists! Using sequences from %s" % self.root)
        else:
            raise ValueError("Sequences folder doesn't exist! Exiting...")

        assert isinstance(self.labels, dict)  # make sure labels is a dict
        # make sure color_map is a dict
        assert isinstance(self.color_map, dict)
        # make sure learning_map is a dict
        assert isinstance(self.learning_map, dict)
        # make sure sequences is a list
        assert isinstance(self.sequences, list)

        self.all_residaul_id = [1 * i for i in range(self.n_input_scans)]

        # placeholder for filenames
        self.scan_files = {}
        self.label_files = {}
        self.poses = {}

        for i in self.all_residaul_id:
            exec("self.residual_files_" + str(str(i + 1)) + " = {}")
            exec("self.bev_residual_files_" + str(str(i + 1)) + " = {}")

        # fill in with names, checking that all sequences are complete
        for seq in self.sequences:

            seq = "{0:02d}".format(int(seq))  # to string
            print("parsing seq {}".format(seq))

            # get paths for each
            scan_path = os.path.join(self.root, seq, "velodyne")
            label_path = os.path.join(self.root, seq, "labels")

            for i in self.all_residaul_id:
                folder_name = "residual_images_" + str(i + 1)
                bev_folder_name = "bev_residual_images_" + str(i + 1)
                exec(
                    "residual_path_"
                    + str(i + 1)
                    + " = os.path.join(self.root, seq, folder_name)"
                )
                exec(
                    "bev_residual_path_"
                    + str(i + 1)
                    + " = os.path.join(self.root, seq, bev_folder_name)"
                )

            # get files
            scan_files = [
                os.path.join(dp, f)
                for dp, dn, fn in os.walk(os.path.expanduser(scan_path))
                for f in fn
                if is_scan(f)
            ]
            label_files = [
                os.path.join(dp, f)
                for dp, dn, fn in os.walk(os.path.expanduser(label_path))
                for f in fn
                if is_label(f)
            ]

            for i in self.all_residaul_id:
                exec(
                    "residual_files_"
                    + str(i + 1)
                    + " = "
                    + "[os.path.join(dp, f) for dp, dn, fn in "
                    "os.walk(os.path.expanduser(residual_path_" + str(i + 1) + "))"
                    " for f in fn if is_residual(f)]"
                )

                exec(
                    "bev_residual_files_"
                    + str(i + 1)
                    + " = "
                    + "[os.path.join(dp, f) for dp, dn, fn in "
                    "os.walk(os.path.expanduser(bev_residual_path_" + str(i + 1) + "))"
                    " for f in fn if is_residual(f)]"
                )

            # Get poses and transform them to LiDAR coord frame for transforming point clouds
            # load poses
            pose_file = os.path.join(self.root, seq, "poses.txt")
            poses = np.array(load_poses(pose_file))
            inv_frame0 = np.linalg.inv(poses[0])

            # load calibrations
            calib_file = os.path.join(self.root, seq, "calib.txt")
            T_cam_velo = load_calib(calib_file)
            T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
            T_velo_cam = np.linalg.inv(T_cam_velo)

            # convert kitti poses from camera coord to LiDAR coord
            new_poses = []
            for pose in poses:
                new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
            self.poses[seq] = np.array(new_poses)

            # check all scans have labels
            assert len(scan_files) == len(label_files)

            """
            Added for dynamic object segmentation
            """
            # fill index mapper which is needed when loading several frames
            # n_used_files = max(0, len(scan_files) - self.n_input_scans + 1)  # this is used for multi-scan attach
            n_used_files = max(
                0, len(scan_files)
            )  # this is used for multi residual images
            for start_index in range(n_used_files):
                self.index_mapping[dataset_index] = (seq, start_index)
                dataset_index += 1
            self.dataset_size += n_used_files
            """"""

            # extend list
            scan_files.sort()
            label_files.sort()

            self.scan_files[seq] = scan_files
            self.label_files[seq] = label_files

            if self.use_residual:
                # for i in range(self.n_input_scans):
                for i in self.all_residaul_id:
                    exec("residual_files_" + str(i + 1) + ".sort()")
                    exec(
                        "self.residual_files_"
                        + str(i + 1)
                        + "[seq]"
                        + " = "
                        + "residual_files_"
                        + str(i + 1)
                    )

                    exec("bev_residual_files_" + str(i + 1) + ".sort()")
                    exec(
                        "self.bev_residual_files_"
                        + str(i + 1)
                        + "[seq]"
                        + " = "
                        + "bev_residual_files_"
                        + str(i + 1)
                    )
        # print("\033[32m No model directory found.\033[0m")

        print(f"\033[32m There are {self.dataset_size} frames in total. \033[0m")
        if drop_few_static_frames:
            self.remove_few_static_frames()
            print(
                f"\033[32m Remove {self.total_remove} frames. \n New use {self.dataset_size} frames. \033[0m"
            )

        print(
            f"\033[32m Using {self.dataset_size} scans from sequences {self.sequences}\033[0m"
        )

    def get_multiple_data_from_scan(
        self,
        mode,
        config,
        scan_file=None,
        label_file=None,
        index_pose=None,
        current_pose=None,
    ):
        if mode == "Range":
            DA, flip_sign, drop_points = config

            scan = SemLaserScan(
                sem_color_dict=self.color_map,
                project=True,
                H=self.sensor_img_H,
                W=self.sensor_img_W,
                fov_up=self.sensor_fov_up,
                fov_down=self.sensor_fov_down,
                use_normal=self.use_normal,
                DA=DA,
                flip_sign=flip_sign,
                drop_points=drop_points,
            )

            scan.open_scan(
                scan_file, index_pose, current_pose, if_transform=self.transform_mod
            )

            scan.open_label(label_file)
            tmp_sem_label = scan.sem_label.copy()
            scan.sem_label = self.map(scan.sem_label, self.learning_map)
            scan.proj_sem_movable_label = scan.proj_sem_label.copy()
            scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)

            scan.proj_sem_movable_label = self.map(
                scan.proj_sem_movable_label, self.movable_learning_map
            )
            # scan.inst_label = self.map(scan.inst_label, self.movable_learning_map) # 혹시 몰라 실행 안함
            GTs_moving, GTs_movable = scan.sem_label, self.map(
                tmp_sem_label, self.movable_learning_map
            )

            GTs_moving = torch.from_numpy(GTs_moving).long()
            GTs_movable = torch.from_numpy(GTs_movable).long()

            unproj_n_points = scan.points.shape[0]
            proj_range = torch.from_numpy(scan.proj_range).clone()
            proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
            proj_remission = torch.from_numpy(scan.proj_remission).clone()
            proj_mask = torch.from_numpy(scan.proj_mask)
            proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
            proj_labels = proj_labels * proj_mask
            proj_movable_labels = torch.from_numpy(scan.proj_sem_movable_label).clone()
            proj_movable_labels = proj_movable_labels * proj_mask
            proj_x = torch.full([self.max_points], -1, dtype=torch.long)
            proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
            proj_y = torch.full([self.max_points], -1, dtype=torch.long)
            proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)
            # unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
            # unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)

            return (
                (GTs_moving, GTs_movable),
                proj_range,
                proj_xyz,
                scan.points,
                proj_remission,
                proj_mask,
                proj_labels,
                proj_movable_labels,
                # unproj_n_points,
                proj_x,
                proj_y,
                # proj_range,
                # unproj_range,
            )

        elif mode == "BEV":
            seq, frame_id = config

            dataset_sequences_path = "/home/ssd_4tb/minjae/KITTI/dataset/sequences"
            labels_folder = os.path.join(dataset_sequences_path, f"{seq:02d}", "labels")
            velodyne_folder = os.path.join(
                dataset_sequences_path, f"{seq:02d}", "velodyne"
            )

            velodyne_path = os.path.join(velodyne_folder, f"{frame_id:06d}.bin")
            label_path = os.path.join(labels_folder, f"{frame_id:06d}.label")

            bev_data = process_scan_as_bev(velodyne_path, label_path)

            # slicing을 통해 필요한 데이터를 한 번에 읽음
            bev_bunch = bev_data["bev_composite"]
            bev_proj_x = bev_data["bev_proj_x"]
            bev_proj_y = bev_data["bev_proj_y"]
            # bev_unproj_range = bev_data["bev_unproj_range"]

            # # bev_range는 bev_bunch의 첫번째 요소를 사용 (이미 numpy array임)
            # bev_range = bev_bunch[0]

            # torch tensor 변환 (불필요한 clone() 제거)
            bev = torch.from_numpy(bev_bunch[:5])
            bev_labels = torch.from_numpy(bev_bunch[5])
            bev_movable_labels = torch.from_numpy(bev_bunch[6])

            return (
                bev,
                bev_labels,
                bev_movable_labels,
                bev_proj_x,
                bev_proj_y,
                # bev_range,
                # bev_unproj_range,
            )
        else:
            raise Exception("Wrong mode at 'get_multiple_data_from_scan'")

    # def __getitem__(self, dataset_index):
    #     start_time = time.time()
    #     seq, start_index = self.index_mapping[dataset_index]
    #     current_index = start_index
    #     current_pose = self.poses[seq][current_index]
    #     proj_full = torch.Tensor()
    #     bev_full = torch.Tensor()
    #     for index in range(start_index, start_index + 1):
    #         scan_file = self.scan_files[seq][current_index]
    #         label_file = self.label_files[seq][current_index]

    #         # --------------------------------------------------------------------------
    #         # 1) residual_input_scans_id 결정 (원본 코드와 동일 로직)
    #         # --------------------------------------------------------------------------
    #         residual_input_scans_id = [1 * i for i in range(self.n_input_scans)]

    #         for i in residual_input_scans_id:
    #             exec(
    #                 "residual_file_"
    #                 + str(i + 1)
    #                 + " = "
    #                 + "self.residual_files_"
    #                 + str(i + 1)
    #                 + "[seq][index]"
    #             )
    #             exec(
    #                 "bev_residual_file_"
    #                 + str(i + 1)
    #                 + " = "
    #                 + "self.bev_residual_files_"
    #                 + str(i + 1)
    #                 + "[seq][index]"
    #             )

    #         index_pose = self.poses[seq][index]

    #         DA = False
    #         flip_sign = False
    #         drop_points = False
    #         if self.transform:
    #             if random.random() > 0.5:
    #                 if random.random() > 0.5:
    #                     DA = True
    #                 if random.random() > 0.5:
    #                     flip_sign = True
    #                 if random.random() > 0.5:
    #                     rot = True
    #                 drop_points = random.uniform(0, 0.5)

    #         (
    #             (GTs_moving, GTs_movable),
    #             proj_range,
    #             proj_xyz,
    #             points,
    #             proj_remission,
    #             proj_mask,
    #             proj_labels,
    #             proj_movable_labels,
    #             unproj_n_points,
    #             proj_x,
    #             proj_y,
    #             proj_range,
    #             unproj_range,
    #         ) = self.get_multiple_data_from_scan(
    #             "Range",
    #             [DA, flip_sign, drop_points],
    #             scan_file,
    #             label_file,
    #             index_pose,
    #             current_pose,
    #         )

    #         (
    #             bev,
    #             bev_labels,
    #             bev_movable_labels,
    #             bev_proj_x,
    #             bev_proj_y,
    #             bev_range,
    #             bev_unproj_range,
    #         ) = self.get_multiple_data_from_scan("BEV", [int(seq), int(current_index)])

    #         for i in residual_input_scans_id:
    #             exec(
    #                 "proj_residuals_"
    #                 + str(i + 1)
    #                 + " = torch.Tensor(np.load(residual_file_"
    #                 + str(i + 1)
    #                 + "))"
    #             )
    #             exec(
    #                 "bev_residuals_"
    #                 + str(i + 1)
    #                 + " = torch.Tensor(np.load(bev_residual_file_"
    #                 + str(i + 1)
    #                 + "))"
    #             )

    #         proj = torch.cat(
    #             [
    #                 proj_range.unsqueeze(0).clone(),
    #                 proj_xyz.clone().permute(2, 0, 1),
    #                 proj_remission.unsqueeze(0).clone(),
    #             ]
    #         )

    #         proj = (proj - self.sensor_img_means[:, None, None]) / self.sensor_img_stds[
    #             :, None, None
    #         ]

    #         bev = (bev - self.sensor_img_means[:, None, None]) / self.sensor_img_stds[
    #             :, None, None
    #         ]

    #         proj_full = torch.cat([proj_full, proj])
    #         bev_full = torch.cat([bev_full, bev])

    #         for i in residual_input_scans_id:
    #             proj_full = torch.cat(
    #                 [
    #                     proj_full,
    #                     torch.unsqueeze(eval("proj_residuals_" + str(i + 1)), 0),
    #                 ]
    #             )
    #             bev_full = torch.cat(
    #                 [
    #                     bev_full,
    #                     torch.unsqueeze(eval("bev_residuals_" + str(i + 1)), 0),
    #                 ]
    #             ).float()

    #     proj_full = proj_full * proj_mask.float()

    #     path_seq, path_name = seq, "%06d.npy" % int(current_index)

    #     # --------------------------------------------------------------------------
    #     # return
    #     # --------------------------------------------------------------------------
    #     end_time = time.time()
    #     print(f"Get data time: {end_time - start_time:.6f} sec")
    #     return (
    #         (points, (GTs_moving, GTs_movable)),  # (    (n, 3),  (  (n,), (n,)  )    )
    #         (proj_full, bev_full),  # range (13, H, W) | bev (13, H_bev, W_bev)
    #         (proj_labels, proj_movable_labels),  # range (H, W), (H, W)
    #         (bev_labels, bev_movable_labels),  # bev (H_bev, W_bev), (H_bev, W_bev)
    #         (path_seq, path_name, unproj_n_points),  # ex. '08', '000123.npy', 122319
    #         (proj_x, proj_y),  # (150000, ), (150000, )
    #         (bev_proj_x, bev_proj_y),  # (150000, ), (150000, )
    #         (proj_xyz, bev[1:4].permute(1, 2, 0)),  # (H, W, 3), (H_bev, W_bev, 3)
    #         (proj_range, bev_range),  # (H, W), (H_bev, W_bev)
    #         (unproj_range, bev_unproj_range),  # (150000, ), (150000, )
    #     )

    def __getitem__(self, dataset_index):
        start_time = time.time()
        # 인덱스 매핑 및 현재 시퀀스, 인덱스, 포즈 가져오기
        seq, start_index = self.index_mapping[dataset_index]
        current_index = start_index
        current_pose = self.poses[seq][current_index]

        scan_file = self.scan_files[seq][current_index]
        label_file = self.label_files[seq][current_index]

        # residual_input_scans_id 결정 (예: [0, 1, 2, ..., n_input_scans-1])
        residual_ids = list(range(self.n_input_scans))

        # exec 대신 getattr를 사용하여 residual 파일 경로를 리스트로 가져옵니다.
        residual_files = [
            getattr(self, f"residual_files_{i+1}")[seq][current_index]
            for i in residual_ids
        ]
        bev_residual_files = [
            getattr(self, f"bev_residual_files_{i+1}")[seq][current_index]
            for i in residual_ids
        ]

        index_pose = self.poses[seq][current_index]

        # 변형(augmentation) 플래그 결정
        DA = False
        flip_sign = False
        drop_points = False
        if self.transform:
            if random.random() > 0.5:
                if random.random() > 0.5:
                    DA = True
                if random.random() > 0.5:
                    flip_sign = True
                if random.random() > 0.5:
                    rot = True  # rot 변수는 사용 여부에 맞게 처리하세요.
                drop_points = random.uniform(0, 0.5)

        # Range 스캔 데이터 가져오기
        (
            (GTs_moving, GTs_movable),
            proj_range,
            proj_xyz,
            points,
            proj_remission,
            proj_mask,
            proj_labels,
            proj_movable_labels,
            # unproj_n_points,
            proj_x,
            proj_y,
            # _,
            # unproj_range,
        ) = self.get_multiple_data_from_scan(
            "Range",
            [DA, flip_sign, drop_points],
            scan_file,
            label_file,
            index_pose,
            current_pose,
        )

        # BEV 스캔 데이터 가져오기
        (
            bev,
            bev_labels,
            bev_movable_labels,
            bev_proj_x,
            bev_proj_y,
            # bev_range,
            # bev_unproj_range,
        ) = self.get_multiple_data_from_scan("BEV", [int(seq), int(current_index)])

        # residual 파일들을 torch.Tensor로 로드 (리스트 컴프리헨션 사용)
        proj_residuals = [torch.Tensor(np.load(f).copy()) for f in residual_files]
        bev_residuals = [torch.Tensor(np.load(f).copy()) for f in bev_residual_files]

        # Range 데이터 구성: proj_range, proj_xyz, proj_remission을 연결
        proj = torch.cat(
            [
                proj_range.unsqueeze(0).clone(),
                proj_xyz.clone().permute(2, 0, 1),
                proj_remission.unsqueeze(0).clone(),
            ],
            dim=0,
        )

        proj = (proj - self.sensor_img_means[:, None, None]) / self.sensor_img_stds[
            :, None, None
        ]
        bev = (bev - self.sensor_img_means[:, None, None]) / self.sensor_img_stds[
            :, None, None
        ]

        # residual들을 한 번에 연결 (매 반복마다 cat 호출 대신 리스트에 저장 후 연결)
        proj_parts = [proj] + [res.unsqueeze(0) for res in proj_residuals]
        bev_parts = [bev] + [res.unsqueeze(0) for res in bev_residuals]

        proj_tensor = torch.cat(proj_parts, dim=0)
        bev_tensor = torch.cat(bev_parts, dim=0).float()

        # proj_mask 적용 (broadcasting 적용)
        proj_tensor = proj_tensor * proj_mask.float()

        # 한 번만 반복하므로 첫 번째 항목을 사용
        proj_full = proj_tensor
        bev_full = bev_tensor

        # path_seq = seq
        # path_name = "%06d.npy" % int(current_index)

        # return (
        #     (points, (GTs_moving, GTs_movable)),
        #     (proj_full, bev_full),
        #     (proj_labels, proj_movable_labels),
        #     (bev_labels, bev_movable_labels),
        #     (path_seq, path_name, unproj_n_points),
        #     (proj_x, proj_y),
        #     (bev_proj_x, bev_proj_y),
        #     (proj_xyz, bev[1:4].permute(1, 2, 0)),
        #     (proj_range, bev_range),
        #     (unproj_range, bev_unproj_range),
        # )
        end_time = time.time()
        # print(
        #     f"Dataset getitem time: [{dataset_index} / {seq} / {start_index}] {end_time - start_time} sec"
        # )

        return (
            (points, (GTs_moving, GTs_movable)),
            (proj_full, bev_full),
            (proj_labels, proj_movable_labels),
            (bev_labels, bev_movable_labels),
            (proj_x, proj_y),
            (bev_proj_x, bev_proj_y),
        )

    def __len__(self):
        return self.dataset_size

    @staticmethod
    def map(label, mapdict):
        """
        주어진 mapdict에 따라 label 배열을 매핑합니다.
        - mapdict의 최대 key와 label 배열의 최대값 중 큰 값에 +1 (그리고 추가 여유를 위해 +100)을 하여 lut의 크기를 결정합니다.
        - 존재하지 않는 값은 기본값 0으로 처리합니다.
        - 만약 mapdict의 값이 리스트라면 첫 번째 값을 사용합니다.
        """
        # mapdict의 최대 key값과 label 배열의 최대값을 구합니다.
        max_key = max(mapdict.keys())
        label_max = np.max(label)
        # lut 크기를 결정: 두 값 중 큰 값 +1, 그리고 여유분 +100
        table_size = max(max_key, label_max) + 1 + 100
        lut = np.zeros(table_size, dtype=np.int32)

        # lookup table 생성: key에 해당하는 위치에 value 할당
        for key, value in mapdict.items():
            try:
                if isinstance(value, list):
                    lut[key] = value[0]
                else:
                    lut[key] = value
            except IndexError:
                print("Wrong key", key)
        # label 배열에 대해 lookup table 적용
        return np.take(lut, label)

    def remove_few_static_frames(self):
        # Developed by Jiadai Sun 2021-11-07
        # 너무 많은 정적 프레임이 훈련 시간을 증가시키므로 일부 프레임을 제거하기 위한 함수입니다.
        # 수정 대상: self.scan_files, self.label_files, self.residual_files_1~8, self.bev_residual_files_1~8,
        #             self.poses, self.index_mapping, self.dataset_size

        remove_mapping_path = os.path.join(
            os.path.dirname(__file__),
            "../../../config/train_split_dynamic_pointnumber.txt",
        )
        with open(remove_mapping_path) as fd:
            lines = fd.readlines()
            lines = [line.strip() for line in lines]

        pending_dict = {}
        for line in lines:
            if line != "":
                seq, fid, _ = line.split()
                if int(seq) in self.sequences:
                    if seq in pending_dict.keys():
                        if fid in pending_dict[seq]:
                            raise ValueError(
                                f"!!!! Duplicate {fid} in seq {seq} in .txt file"
                            )
                        pending_dict[seq].append(fid)
                    else:
                        pending_dict[seq] = [fid]

        self.total_remove = 0
        self.index_mapping = {}  # 재초기화
        dataset_index = 0
        self.dataset_size = 0
        for seq in self.sequences:
            seq = "{0:02d}".format(int(seq))
            if seq in pending_dict.keys():
                raw_len = len(self.scan_files[seq])

                # lidar scan 파일 필터링
                scan_files = self.scan_files[seq]
                useful_scan_paths = [
                    path
                    for path in scan_files
                    if os.path.split(path)[-1][:-4] in pending_dict[seq]
                ]
                self.scan_files[seq] = useful_scan_paths

                # label 파일 필터링
                label_files = self.label_files[seq]
                useful_label_paths = [
                    path
                    for path in label_files
                    if os.path.split(path)[-1][:-6] in pending_dict[seq]
                ]
                self.label_files[seq] = useful_label_paths

                # poses 필터링
                self.poses[seq] = self.poses[seq][list(map(int, pending_dict[seq]))]

                assert len(useful_scan_paths) == len(useful_label_paths)
                assert len(useful_scan_paths) == self.poses[seq].shape[0]

                # dataloader __getitem__에서 사용할 index_mapping과 dataset_size 재설정
                n_used_files = max(0, len(useful_scan_paths))
                for start_index in range(n_used_files):
                    self.index_mapping[dataset_index] = (seq, start_index)
                    dataset_index += 1
                self.dataset_size += n_used_files

                # range residuals와 함께 bev residuals도 필터링
                if self.use_residual:
                    for i in self.all_residaul_id:
                        # range residual 파일 필터링
                        tmp_residuals = eval(f"self.residual_files_{i+1}['{seq}']")
                        tmp_pending_list = eval(f"pending_dict['{seq}']")
                        tmp_usefuls = [
                            path
                            for path in tmp_residuals
                            if os.path.split(path)[-1][:-4] in tmp_pending_list
                        ]
                        exec(f"self.residual_files_{i+1}['{seq}'] = tmp_usefuls")
                        new_len = len(eval(f"self.residual_files_{i+1}['{seq}']"))
                        print(
                            f"  Drop residual_images_{i+1} in seq{seq}: {len(tmp_residuals)} -> {new_len}"
                        )

                        # bev residual 파일 필터링 추가
                        tmp_bev_residuals = eval(
                            f"self.bev_residual_files_{i+1}['{seq}']"
                        )
                        tmp_usefuls_bev = [
                            path
                            for path in tmp_bev_residuals
                            if os.path.split(path)[-1][:-4] in tmp_pending_list
                        ]
                        exec(
                            f"self.bev_residual_files_{i+1}['{seq}'] = tmp_usefuls_bev"
                        )
                        new_len_bev = len(
                            eval(f"self.bev_residual_files_{i+1}['{seq}']")
                        )
                        print(
                            f"  Drop bev_residual_images_{i+1} in seq{seq}: {len(tmp_bev_residuals)} -> {new_len_bev}"
                        )
                new_len = len(self.scan_files[seq])
                print(f"Seq {seq} drop {raw_len - new_len}: {raw_len} -> {new_len}")
                self.total_remove += raw_len - new_len


class Parser:
    # standard conv, BN, relu
    def __init__(
        self,
        root,  # directory for data
        train_sequences,  # sequences to train
        valid_sequences,  # sequences to validate.
        test_sequences,  # sequences to test (if none, don't get)
        split,  # split (train, valid, test)
        labels,  # labels in data
        residual_aug,  # the data augmentation for residual maps
        color_map,  # color for each label
        learning_map,  # mapping for training labels
        movable_learning_map,
        learning_map_inv,  # recover labels from xentropy
        movable_learning_map_inv,
        sensor,  # sensor to use
        max_points,  # max points in each scan in entire dataset
        batch_size,  # batch size for train and val
        workers,  # threads to load data
        valid_residual_delta_t=1,  # modulation interval in data augmentation fro residual maps
        gt=True,  # get gt?
        shuffle_train=False,
    ):  # shuffle training set?
        super(Parser, self).__init__()

        # if I am training, get the dataset
        self.root = root
        self.train_sequences = train_sequences
        self.valid_sequences = valid_sequences
        self.test_sequences = test_sequences
        self.split = split
        self.labels = labels
        self.residual_aug = residual_aug
        self.valid_residual_delta_t = valid_residual_delta_t
        self.color_map = color_map
        self.learning_map = learning_map
        self.movable_learning_map = movable_learning_map
        self.learning_map_inv = learning_map_inv
        self.movable_learning_map_inv = movable_learning_map_inv
        self.sensor = sensor
        self.max_points = max_points
        self.batch_size = batch_size
        self.workers = workers
        self.gt = gt
        self.shuffle_train = shuffle_train

        # number of classes that matters is the one for xentropy
        self.nclasses = len(self.learning_map_inv)
        self.movable_nclasses = len(self.movable_learning_map_inv)

        # Data loading code
        if self.split == "train":
            self.train_dataset = SemanticKitti(
                root=self.root,
                sequences=self.train_sequences,
                labels=self.labels,
                residual_aug=self.residual_aug,
                valid_residual_delta_t=self.valid_residual_delta_t,
                color_map=self.color_map,
                learning_map=self.learning_map,
                movable_learning_map=self.movable_learning_map,
                learning_map_inv=self.learning_map_inv,
                movable_learning_map_inv=self.movable_learning_map_inv,
                sensor=self.sensor,
                max_points=max_points,
                transform=True,
                gt=self.gt,
                drop_few_static_frames=True,
            )

            shuffle_train = True
            if torch.cuda.is_available() and torch.cuda.device_count() < 2:
                self.train_sampler = None
            else:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.train_dataset
                )
                shuffle_train = False
            # self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
            #                                                batch_size=self.batch_size,
            #                                                shuffle=self.shuffle_train,
            #                                                # shuffle=False,
            #                                                num_workers=self.workers,
            #                                                pin_memory=True,
            #                                                drop_last=True)
            self.trainloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=shuffle_train,
                num_workers=self.workers,
                pin_memory=True,
                drop_last=True,
                sampler=self.train_sampler,
            )

            assert len(self.trainloader) > 0
            self.trainiter = iter(self.trainloader)

            self.valid_dataset = SemanticKitti(
                root=self.root,
                sequences=self.valid_sequences,
                labels=self.labels,
                residual_aug=self.residual_aug,
                valid_residual_delta_t=self.valid_residual_delta_t,
                color_map=self.color_map,
                learning_map=self.learning_map,
                movable_learning_map=self.movable_learning_map,
                learning_map_inv=self.learning_map_inv,
                movable_learning_map_inv=self.movable_learning_map_inv,
                sensor=self.sensor,
                max_points=max_points,
                gt=self.gt,
            )

            self.validloader = torch.utils.data.DataLoader(
                self.valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=True,
                drop_last=True,
            )
            assert len(self.validloader) > 0
            self.validiter = iter(self.validloader)

        if self.split == "valid":
            self.valid_dataset = SemanticKitti(
                root=self.root,
                sequences=self.valid_sequences,
                labels=self.labels,
                residual_aug=self.residual_aug,
                valid_residual_delta_t=self.valid_residual_delta_t,
                color_map=self.color_map,
                learning_map=self.learning_map,
                movable_learning_map=self.movable_learning_map,
                learning_map_inv=self.learning_map_inv,
                movable_learning_map_inv=self.movable_learning_map_inv,
                sensor=self.sensor,
                max_points=max_points,
                gt=self.gt,
            )

            self.validloader = torch.utils.data.DataLoader(
                self.valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.workers,
                pin_memory=True,
                drop_last=True,
            )
            assert len(self.validloader) > 0
            self.validiter = iter(self.validloader)

        if self.split == "test":
            if self.test_sequences:
                self.test_dataset = SemanticKitti(
                    root=self.root,
                    sequences=self.test_sequences,
                    labels=self.labels,
                    residual_aug=self.residual_aug,
                    valid_residual_delta_t=self.valid_residual_delta_t,
                    color_map=self.color_map,
                    learning_map=self.learning_map,
                    learning_map_inv=self.learning_map_inv,
                    movable_learning_map_inv=self.movable_learning_map_inv,
                    sensor=self.sensor,
                    max_points=max_points,
                    gt=False,
                )

                self.testloader = torch.utils.data.DataLoader(
                    self.test_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.workers,
                    pin_memory=True,
                    drop_last=True,
                )
                assert len(self.testloader) > 0
                self.testiter = iter(self.testloader)

    def get_train_batch(self):
        scans = self.trainiter.next()
        return scans

    def get_train_set(self):
        return self.trainloader

    def get_valid_batch(self):
        scans = self.validiter.next()
        return scans

    def get_valid_set(self):
        return self.validloader

    def get_test_batch(self):
        scans = self.testiter.next()
        return scans

    def get_test_set(self):
        return self.testloader

    def get_train_size(self):
        return len(self.trainloader)

    def get_valid_size(self):
        return len(self.validloader)

    def get_test_size(self):
        return len(self.testloader)

    def get_n_classes(self, movable=False):
        if not movable:
            return self.nclasses
        else:
            return self.movable_nclasses

    def get_original_class_string(self, idx):
        return self.labels[idx]

    def get_xentropy_class_string(self, idx, movable=False):
        if not movable:
            return self.labels[self.learning_map_inv[idx]]
        else:
            return self.labels[self.movable_learning_map_inv[idx]]

    def to_original(self, label, movable=False):
        # put label in original values
        if not movable:
            return SemanticKitti.map(label, self.learning_map_inv)
        else:
            return SemanticKitti.map(label, self.movable_learning_map_inv)

    def to_xentropy(self, label, movable=False):
        # put label in xentropy values
        if not movable:
            return SemanticKitti.map(label, self.learning_map)
        else:
            return SemanticKitti.map(label, self.movable_learning_map)

    def to_color(self, label, movable=False):
        if not movable:
            # put label in original values
            label = SemanticKitti.map(label, self.learning_map_inv)
        else:
            # put label in original values
            label = SemanticKitti.map(label, self.movable_learning_map_inv)
        # put label in color
        return SemanticKitti.map(label, self.color_map)
