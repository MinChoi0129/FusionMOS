#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
from tqdm import tqdm
import multiprocessing

class BevScan:
    moving_learning_map = {
        0: 0,
        1: 0,
        9: 1,
        10: 1,
        11: 1,
        13: 1,
        15: 1,
        16: 1,
        18: 1,
        20: 1,
        30: 1,
        31: 1,
        32: 1,
        40: 1,
        44: 1,
        48: 1,
        49: 1,
        50: 1,
        51: 1,
        52: 1,
        60: 1,
        70: 1,
        71: 1,
        72: 1,
        80: 1,
        81: 1,
        99: 1,
        251: 2,
        252: 2,
        253: 2,
        254: 2,
        255: 2,
        256: 2,
        257: 2,
        258: 2,
        259: 2,
    }
    movable_learning_map = {
        0: 0,
        1: 0,
        9: 1,
        16: 1,
        40: 1,
        44: 1,
        48: 1,
        49: 1,
        50: 1,
        51: 1,
        52: 1,
        60: 1,
        70: 1,
        71: 1,
        72: 1,
        80: 1,
        81: 1,
        99: 1,
        10: 2,
        11: 2,
        13: 2,
        15: 2,
        18: 2,
        20: 2,
        30: 2,
        31: 2,
        32: 2,
        251: 2,
        252: 2,
        253: 2,
        254: 2,
        255: 2,
        256: 2,
        257: 2,
        258: 2,
        259: 2,
    }

    def __init__(self, velodyne_path, label_path, proj_H, proj_W, max_range, min_range):
        self.velodyne_path = velodyne_path
        self.label_path = label_path
        self.proj_H = proj_H
        self.proj_W = proj_W
        self.max_range = max_range
        self.min_range = min_range
        self.process()
    
    def load_velodyne(self):
        scan = np.fromfile(self.velodyne_path, dtype=np.float32).reshape((-1, 4)) # nx4
        points = scan[:, :3] # nx3
        remissions = scan[:, 3] # nx1
        homo_points = np.ones((points.shape[0], 4), dtype=np.float32) # nx4
        homo_points[:, :3] = points # nx4 (XYZ1, XYZ1, XYZ1, ... , XYZ1)

        self.homo_points = homo_points
        self.remissions = remissions
        return self.homo_points, self.remissions
    
    def load_label(self):
        """
        .label 파일에서 semantic 라벨과 instance 라벨을 읽어옴.
        반환: sem_label (n,), inst_label (n,)
        """
        label = np.fromfile(self.label_path, dtype=np.int32)
        self.sem_label = label & 0xFFFF
        self.inst_label = label >> 16
        return self.sem_label, self.inst_label

    def create_ternary_label(self, mapping_dict):
        mapped = np.full_like(self.bev_sem_label, -1, dtype=np.int32)
        unique = np.unique(self.bev_sem_label)
        for val in unique:
            if val in mapping_dict:
                mapped[self.bev_sem_label == val] = mapping_dict[val]
            else:
                mapped[self.bev_sem_label == val] = 0
        ternary = np.where(self.bev_sem_label == -1, 0, mapped)
        return ternary
    
    @staticmethod
    def bev_projection_only(homo_points, proj_H, proj_W, max_range, min_range):
        xy_dist = np.sqrt(homo_points[:, 0] ** 2 + homo_points[:, 1] ** 2)
        valid_mask = (xy_dist > min_range) & (xy_dist < max_range)
        filtered_points = homo_points[valid_mask]
        filtered_dist = xy_dist[valid_mask]

        x_img = (filtered_points[:, 0] + max_range) / (2.0 * max_range)
        y_img = (filtered_points[:, 1] + max_range) / (2.0 * max_range)
        x_img = np.clip(np.floor(x_img * proj_W), 0, proj_W - 1).astype(np.int32)
        y_img = np.clip(np.floor(y_img * proj_H), 0, proj_H - 1).astype(np.int32)

        # 먼 점부터 투영 (덮어쓰기를 위해)
        order = np.argsort(filtered_dist)[::-1]
        x_img = x_img[order]
        y_img = y_img[order]
        # filtered_dist = filtered_dist[order]

        bev_range = np.full((proj_H, proj_W), 0, dtype=np.float32)
        bev_range[y_img, x_img] = 1

        # norm_range = np.clip(bev_range, 0, max_range) / max_range

        return bev_range

    def bev_projection_full(self):
        xy_dist = np.sqrt(self.homo_points[:, 0] ** 2 + self.homo_points[:, 1] ** 2)
        valid_mask = (xy_dist > self.min_range) & (xy_dist < self.max_range)
        filtered_points = self.homo_points[valid_mask]
        # filtered_rem = self.remissions[valid_mask]
        filtered_dist = xy_dist[valid_mask]
        # orig_idx = np.nonzero(valid_mask)[0]  # 원본 점 인덱스

        # if filtered_points.shape[0] == 0:
        #     empty = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
        #     return (
        #         empty,
        #         np.full((self.proj_H, self.proj_W, 3), -1, dtype=np.float32),
        #         empty,
        #         np.array([]),
        #         np.array([]),
        #         np.array([]),
        #         np.full((self.proj_H, self.proj_W), -1, dtype=np.int32),
        #     )

        # x, y 좌표를 [0,1]로 정규화 후 이미지 크기에 맞게 변환
        x_img = (filtered_points[:, 0] + self.max_range) / (2.0 * self.max_range)
        y_img = (filtered_points[:, 1] + self.max_range) / (2.0 * self.max_range)
        x_img = np.clip(np.floor(x_img * self.proj_W), 0, self.proj_W - 1).astype(np.int32)
        y_img = np.clip(np.floor(y_img * self.proj_H), 0, self.proj_H - 1).astype(np.int32)

        # 먼 점부터 투영 (덮어쓰기를 위해)
        order = np.argsort(filtered_dist)[::-1]
        x_img = x_img[order]
        y_img = y_img[order]
        # filtered_dist = filtered_dist[order]
        # filtered_points = filtered_points[order]
        # filtered_rem = filtered_rem[order]
        # proj_idx_ordered = orig_idx[order]

        bev_range = np.full((self.proj_H, self.proj_W), 0, dtype=np.float32)
        # bev_xyz = np.full((self.proj_H, self.proj_W, 3), -1, dtype=np.float32)
        # bev_remission = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
        # proj_idx = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)

        bev_range[y_img, x_img] = 1
        # bev_xyz[y_img, x_img, :] = filtered_points[:, :3]
        # bev_remission[y_img, x_img] = filtered_rem
        # proj_idx[y_img, x_img] = proj_idx_ordered


        # # 정규화: range와 remission 채널
        # norm_range = np.clip(bev_range, 0, self.max_range) / self.max_range
        # if np.any(bev_remission > 0):
        #     norm_rem = np.clip(
        #         bev_remission, 0, np.max(bev_remission[bev_remission > 0])
        #     ) / np.max(bev_remission[bev_remission > 0])
        # else:
        #     norm_rem = bev_remission
        
        self.bev_range = bev_range
        # self.bev_xyz = bev_xyz.transpose((2, 0, 1))  # HWC -> CHW
        # self.bev_remission = norm_rem
        # self.bev_proj_x = x_img
        # self.bev_proj_y = y_img
        # self.bev_unproj_range = filtered_dist
        # self.bev_proj_idx = proj_idx

        # return self.bev_range, self.bev_xyz, self.bev_remission, self.bev_proj_x, self.bev_proj_y, self.bev_unproj_range, self.bev_proj_idx
    
    def bev_labels_full(self):
        self.bev_sem_label = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)
        valid = self.bev_proj_idx >= 0
        self.bev_sem_label[valid] = self.sem_label[self.bev_proj_idx[valid]]

        self.bev_moving_ternary_label = self.create_ternary_label(BevScan.moving_learning_map)
        self.bev_movable_ternary_label = self.create_ternary_label(BevScan.movable_learning_map)

        return self.bev_sem_label, self.bev_moving_ternary_label, self.bev_movable_ternary_label

    def process(self):
        self.load_velodyne()
        # self.load_label()
        self.bev_projection_full()
        # self.bev_labels_full()


def check_and_makedirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def load_poses(pose_path):
    """파일로부터 T_w_cam0 pose (n,4,4)를 불러옵니다."""
    poses = []
    try:
        with open(pose_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                T_w_cam0 = np.fromstring(line, dtype=float, sep=" ")
                T_w_cam0 = T_w_cam0.reshape(3, 4)
                T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                poses.append(T_w_cam0)
    except FileNotFoundError:
        print("Ground truth poses are not available.")
    return np.array(poses)


def load_calib(calib_path):
    """파일로부터 T_cam_velo (4,4) calibration을 불러옵니다."""
    T_cam_velo = []
    try:
        with open(calib_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "Tr:" in line:
                    line = line.replace("Tr:", "")
                    T_cam_velo = np.fromstring(line, dtype=float, sep=" ")
                    T_cam_velo = T_cam_velo.reshape(3, 4)
                    T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))
    except FileNotFoundError:
        print("Calibrations are not available.")
    return np.array(T_cam_velo)


def load_files(folder):
    """폴더 내의 모든 파일 (예: *.bin)을 정렬하여 불러옵니다."""
    file_paths = [
        os.path.join(dp, f)
        for dp, dn, fn in os.walk(os.path.expanduser(folder))
        for f in fn
    ]
    file_paths.sort()
    return file_paths



# -------------------------------------------------------------
# 필요한 함수들 (check_and_makedirs, load_poses, load_calib, load_files)
# 그리고 BevScan 클래스는 이전 코드와 동일하다고 가정
# -------------------------------------------------------------

def load_all_scans_into_memory(scan_paths):
    """
    모든 프레임의 (.bin)과 (.label)을 한 번에 로드해
    포인트 클라우드(또는 homo_points), pose, label 등을 메모리에 저장.
    """
    all_homo_points = []
    # all_labels = []  # 필요하다면 함께 저장
    for scan_p in tqdm(scan_paths, total=len(scan_paths), desc="Loading scans"):
        scan = np.fromfile(scan_p, dtype=np.float32).reshape((-1, 4))
        points = scan[:, :3]
        homo_points = np.ones((points.shape[0], 4), dtype=np.float32)
        homo_points[:, :3] = points
        all_homo_points.append(homo_points)
    
        
    return all_homo_points  # , all_labels


def process_one_seq_bev_in_memory(config):
    """
    프레임별로 파일 I/O를 계속 하지 않고,
    미리 메모리에 올려놓은 뒤 diff를 계산.
    """
    scans_folder = config["scans_folder"]
    labels_folder = config["labels_folder"]
    pose_file = config["pose_file"]
    calib_file = config["calib_file"]
    bev_residual_folder = config["bev_residual_folder"]
    bev_h = config["bev_h"]
    bev_w = config["bev_w"]
    max_range = config["max_range"]
    min_range = config["min_range"]
    num_last_n = config["num_last_n"]

    check_and_makedirs(bev_residual_folder)

    # 1) Pose 및 Calibration
    poses = np.array(load_poses(pose_file))
    inv_frame0 = np.linalg.inv(poses[0])
    T_cam_velo = load_calib(calib_file).reshape((4, 4))
    T_velo_cam = np.linalg.inv(T_cam_velo)
    new_poses = []
    for pose in poses:
        new_poses.append(T_velo_cam @ inv_frame0 @ pose @ T_cam_velo)
    poses = np.array(new_poses)

    # 2) 스캔/레이블 파일 목록
    scan_paths = load_files(scans_folder)
    frame_count = len(scan_paths)

    # 3) 모든 스캔을 메모리에 올려놓기
    all_homo_points = load_all_scans_into_memory(scan_paths)
    # => all_homo_points[i] : i번 프레임의 homo_points (nx4)

    # 4) 프레임 순회하면서 diff 계산
    for frame_idx in tqdm(range(frame_count), desc=f"Processing"):
        file_name = os.path.join(bev_residual_folder, str(frame_idx).zfill(6))
        if frame_idx < num_last_n:
            diff_image = np.zeros((bev_h, bev_w), dtype=np.float32)
            np.save(file_name, diff_image)
            continue

        # 현재 프레임 BEV
        current_pose = poses[frame_idx]
        current_points = all_homo_points[frame_idx]
        # current 라벨이 필요하다면 사용 (all_labels[frame_idx])
        
        # current frame의 BEV 생성
        current_bev = BevScan.bev_projection_only(
            current_points,
            bev_h, bev_w,
            max_range, min_range
        )

        # 이전 프레임 (frame_idx - num_last_n)
        last_pose = poses[frame_idx - num_last_n]
        last_points = all_homo_points[frame_idx - num_last_n]

        # last_points를 current 좌표계로 변환
        last_scan_transformed = (np.linalg.inv(current_pose) @ last_pose @ last_points.T).T[:, :3]

        # 변환된 점으로 BEV 만들기
        last_bev = BevScan.bev_projection_only(
            last_scan_transformed.astype(np.float32),
            bev_h, bev_w,
            max_range, min_range
        )

        # diff
        diff_image = np.abs(current_bev - last_bev)
        np.save(file_name + ".npy", diff_image)
    
    print(f"[Done] {bev_residual_folder} - 총 {frame_count}개 프레임 처리 완료 (In-Memory)")


def process_residual_for_num_last_n_in_memory(seq_id, num_last_n, dataset_folder):
    """
    한 시퀀스를 메모리에 전부 로드한 뒤 diff를 계산하는 예시
    """
    scans_folder = os.path.join(dataset_folder, f"{seq_id:02d}", "velodyne")
    labels_folder = os.path.join(dataset_folder, f"{seq_id:02d}", "labels")
    pose_file = os.path.join(dataset_folder, f"{seq_id:02d}", "poses.txt")
    calib_file = os.path.join(dataset_folder, f"{seq_id:02d}", "calib.txt")
    bev_residual_folder = os.path.join(dataset_folder, f"{seq_id:02d}", f"bev_residual_images_{num_last_n}")

    config = {
        "scans_folder": scans_folder,
        "labels_folder": labels_folder,
        "pose_file": pose_file,
        "calib_file": calib_file,
        "num_last_n": num_last_n,
        "bev_residual_folder": bev_residual_folder,
        "bev_h": 384,
        "bev_w": 384,
        "max_range": 50.0,
        "min_range": 2.0,
    }

    process_one_seq_bev_in_memory(config)


if __name__ == "__main__":
    dataset_folder = "/home/ssd_4tb/minjae/KITTI/dataset/sequences"
    seq_list = range(11)  # 0 ~ 10
    num_last_n_list = range(1, 9)  # 1~8

    # 시퀀스는 순차 처리 예시(원하면 Pool 사용 가능)
    for seq_id in seq_list:
        for n in num_last_n_list:
            process_residual_for_num_last_n_in_memory(seq_id, n, dataset_folder)

    print("모든 시퀀스 - In-Memory BEV residual 생성이 완료되었습니다.")