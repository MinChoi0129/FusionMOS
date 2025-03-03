#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
참고 코드의 흐름을 그대로 따르며, 변수명에 BEV를 포함하고,
config의 키들을 그대로 사용하며, fov_up, fov_down 없이 BEV Residual Image Generation 코드.
"""

import os

os.environ["OMP_NUM_THREADS"] = "4"
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from icecream import ic


###############################################################################
#                              유틸 함수들                                      #
###############################################################################
def check_and_makedirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_yaml(path):
    if yaml.__version__ >= "5.1":
        config = yaml.load(open(path), Loader=yaml.FullLoader)
    else:
        config = yaml.load(open(path))
    return config


def load_poses(pose_path):
    """파일로부터 T_w_cam0 pose (n,4,4)를 불러옵니다."""
    poses = []
    try:
        if ".txt" in pose_path:
            with open(pose_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    T_w_cam0 = np.fromstring(line, dtype=float, sep=" ")
                    T_w_cam0 = T_w_cam0.reshape(3, 4)
                    T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    poses.append(T_w_cam0)
        else:
            poses = np.load(pose_path)["arr_0"]
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


def load_vertex(scan_path):
    """KITTI .bin 파일로부터 3D 포인트 (n×4: x, y, z, 1)를 불러옵니다."""
    current_vertex = np.fromfile(scan_path, dtype=np.float32)
    current_vertex = current_vertex.reshape((-1, 4))
    current_points = current_vertex[:, 0:3]
    # homogeneous 좌표를 위해 마지막 열을 1로 채움
    current_vertex = np.ones((current_points.shape[0], current_points.shape[1] + 1))
    current_vertex[:, :-1] = current_points
    return current_vertex


def load_files(folder):
    """폴더 내의 모든 파일 (예: *.bin)을 정렬하여 불러옵니다."""
    file_paths = [
        os.path.join(dp, f)
        for dp, dn, fn in os.walk(os.path.expanduser(folder))
        for f in fn
    ]
    file_paths.sort()
    return file_paths


###############################################################################
#              BEV 투영 함수 (bev_projection) - fov_up, fov_down 제거            #
###############################################################################
def bev_projection(current_vertex, proj_H, proj_W, max_range, min_range):
    """
    current_vertex: n×4 (x, y, z, 1)
    proj_H, proj_W: range image 해상도
    max_range, min_range: XY 평면 거리 필터링 범위
    리턴: (proj_H, proj_W, 4) 배열
          [:,:,0:3] = (x, y, z)
          [:,:,3]   = XY 평면 거리 (range)
    """
    # XY 평면 거리 계산
    xy_dist = np.sqrt(current_vertex[:, 0] ** 2 + current_vertex[:, 1] ** 2)
    # 필터링
    valid_mask = (xy_dist > min_range) & (xy_dist < max_range)
    current_vertex = current_vertex[valid_mask]
    xy_dist = xy_dist[valid_mask]
    if current_vertex.shape[0] == 0:
        return np.full((proj_H, proj_W, 4), -1, dtype=np.float32)
    # 이미지 좌표로 변환: [-max_range, +max_range] -> [0, 1] -> [0, proj_W 또는 proj_H]
    x_img = (current_vertex[:, 0] + max_range) / (2.0 * max_range)
    y_img = (current_vertex[:, 1] + max_range) / (2.0 * max_range)
    x_img = np.floor(x_img * proj_W).astype(np.int32)
    y_img = np.floor(y_img * proj_H).astype(np.int32)
    x_img = np.clip(x_img, 0, proj_W - 1)
    y_img = np.clip(y_img, 0, proj_H - 1)
    # 깊이 기준 내림차순 소팅 (더 먼 점부터 처리)
    order = np.argsort(xy_dist)[::-1]
    xy_dist = xy_dist[order]
    current_vertex = current_vertex[order]
    x_img = x_img[order]
    y_img = y_img[order]
    # BEV 투영 결과 초기화: 유효하지 않은 영역은 -1
    proj_vertex = np.full((proj_H, proj_W, 4), -1, dtype=np.float32)
    proj_vertex[y_img, x_img, 0] = current_vertex[:, 0]
    proj_vertex[y_img, x_img, 1] = current_vertex[:, 1]
    proj_vertex[y_img, x_img, 2] = current_vertex[:, 2]
    proj_vertex[y_img, x_img, 3] = xy_dist
    return proj_vertex


###############################################################################
#              process_one_seq_bev 함수 (참고 코드 흐름 그대로)                #
###############################################################################
def process_one_seq_bev(config):
    # config의 키를 그대로 사용
    num_frames = config["num_frames"]
    debug = config["debug"]
    normalize = config["normalize"]
    num_last_n = config["num_last_n"]
    visualize = config["visualize"]
    bev_residual_folder = config["bev_residual_folder"]
    visualization_folder_bev = config["visualization_folder_bev"]

    check_and_makedirs(bev_residual_folder)
    if visualize:
        check_and_makedirs(visualization_folder_bev)

    # Pose 불러오기 및 첫 프레임 기준 inv 계산
    pose_file = config["pose_file"]
    poses = np.array(load_poses(pose_file))
    inv_frame0 = np.linalg.inv(poses[0])

    # Calibration 불러오기 및 좌표계 변환 (카메라→LiDAR)
    calib_file = config["calib_file"]
    T_cam_velo = load_calib(calib_file)
    T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    T_velo_cam = np.linalg.inv(T_cam_velo)
    new_poses = []
    for pose in poses:
        new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
    poses = np.array(new_poses)

    # LiDAR 스캔 파일 불러오기
    scan_folder = config["scan_folder"]
    scan_paths = load_files(scan_folder)

    if num_frames >= len(poses) or num_frames <= 0:
        print("generate training data for all frames with number of: ", len(poses))
    else:
        poses = poses[:num_frames]
        scan_paths = scan_paths[:num_frames]

    bev_image_params = config["bev_image"]

    # 시퀀스 전체에 대해 BEV residual 이미지 생성
    for frame_idx in tqdm(range(len(scan_paths))):
        file_name = os.path.join(bev_residual_folder, str(frame_idx).zfill(6))
        # residual 이미지 초기화: 0으로 초기화 (0은 데이터 없음)
        diff_image = np.full(
            (bev_image_params["height"], bev_image_params["width"]), 0, dtype=np.float32
        )
        # 처음 num_last_n 프레임은 dummy 파일 생성
        if frame_idx < num_last_n:
            np.save(file_name, diff_image)
        else:
            # 현재 스캔의 BEV range image 생성
            current_pose = poses[frame_idx]
            current_scan = load_vertex(scan_paths[frame_idx])
            current_bev = bev_projection(
                current_scan.astype(np.float32),
                bev_image_params["height"],
                bev_image_params["width"],
                bev_image_params["max_range"],
                bev_image_params["min_range"],
            )
            current_range = current_bev[:, :, 3]

            # 이전 스캔을 현재 좌표계로 변환 후 BEV range image 생성
            last_pose = poses[frame_idx - num_last_n]
            last_scan = load_vertex(scan_paths[frame_idx - num_last_n])
            last_scan_transformed = (
                np.linalg.inv(current_pose).dot(last_pose).dot(last_scan.T).T
            )
            last_bev = bev_projection(
                last_scan_transformed.astype(np.float32),
                bev_image_params["height"],
                bev_image_params["width"],
                bev_image_params["max_range"],
                bev_image_params["min_range"],
            )
            last_range = last_bev[:, :, 3]

            valid_mask = (
                (current_range > bev_image_params["min_range"])
                & (current_range < bev_image_params["max_range"])
                & (last_range > bev_image_params["min_range"])
                & (last_range < bev_image_params["max_range"])
            )
            difference = np.abs(current_range[valid_mask] - last_range[valid_mask])
            if normalize:
                difference = difference / np.clip(current_range[valid_mask], 1e-6, None)
            diff_image[valid_mask] = difference

            if debug:
                fig, axs = plt.subplots(3, 1, figsize=(8, 12))
                axs[0].imshow(
                    last_range,
                    cmap="viridis",
                    vmin=0,
                    vmax=bev_image_params["max_range"],
                )
                axs[0].set_title("Last BEV Range")
                axs[1].imshow(
                    current_range,
                    cmap="viridis",
                    vmin=0,
                    vmax=bev_image_params["max_range"],
                )
                axs[1].set_title("Current BEV Range")
                axs[2].imshow(diff_image, cmap="viridis", vmin=0, vmax=1)
                axs[2].set_title("BEV Residual")
                plt.tight_layout()
                plt.show()

            if visualize:
                image_name = os.path.join(
                    visualization_folder_bev, str(frame_idx).zfill(6) + ".png"
                )
                image_name2 = os.path.join(
                    visualization_folder_bev, str(frame_idx).zfill(6) + "_current.png"
                )
                plt.imsave(image_name, diff_image, cmap="viridis")
                plt.imsave(image_name2, current_range, cmap="viridis")

            np.save(file_name, diff_image)


###############################################################################
#             단일 (seq_id, num_last_n) 처리 및 시퀀스 병렬 처리 함수             #
###############################################################################
def process_residual_for_num_last_n(seq_id, num_last_n, dataset_folder):
    scan_folder = os.path.join(dataset_folder, f"{seq_id:02d}", "velodyne")
    pose_file = os.path.join(dataset_folder, f"{seq_id:02d}", "poses.txt")
    calib_file = os.path.join(dataset_folder, f"{seq_id:02d}", "calib.txt")
    bev_residual_folder = os.path.join(
        dataset_folder, f"{seq_id:02d}", f"bev_residual_images_{num_last_n}"
    )
    visualization_folder_bev = os.path.join(
        dataset_folder, f"{seq_id:02d}", f"vis_bev_residual_{num_last_n}"
    )

    config = {
        "num_frames": -1,  # -1이면 전체 프레임
        "debug": False,  # True 시 디버그 시각화
        "normalize": True,  # range normalize
        "num_last_n": num_last_n,  # offset
        "visualize": False,  # PNG 저장 여부
        "bev_residual_folder": bev_residual_folder,
        "visualization_folder_bev": visualization_folder_bev,
        "pose_file": pose_file,
        "calib_file": calib_file,
        "scan_folder": scan_folder,
        "bev_image": {
            "height": 768,
            "width": 768,
            "max_range": 50.0,
            "min_range": 2.0,
        },
    }

    process_one_seq_bev(config)


def process_sequence_in_parallel(seq_id, dataset_folder):
    import multiprocessing

    num_workers = multiprocessing.cpu_count() // 2  # 필요에 따라 조정
    num_last_n_list = range(1, 9)
    tasks = [(seq_id, n, dataset_folder) for n in num_last_n_list]
    with multiprocessing.Pool(num_workers) as pool:
        pool.starmap(process_residual_for_num_last_n, tasks)
    print(
        f"[Done] Sequence {seq_id:02d} - all {len(num_last_n_list)} residuals created"
    )


###############################################################################
#                              메인 실행 예시                                 #
###############################################################################
if __name__ == "__main__":
    dataset_folder = "/home/ssd_4tb/minjae/KITTI/dataset/sequences"
    seq_list = range(11)  # 0 ~ 10
    for seq_id in seq_list:
        process_sequence_in_parallel(seq_id, dataset_folder)
    print("모든 시퀀스의 BEV residual 생성이 완료되었습니다.")
