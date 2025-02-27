#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BEV Residual Image Generation with per-sequence parallelization
(시퀀스 하나당 여러 num_last_n 병렬 처리 후 완료 메시지 출력)
"""

import os
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


###############################################################################
#                                 유틸 함수들                                 #
###############################################################################
def check_and_makedirs(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_poses(pose_path):
    """Load ground truth poses (T_w_cam0) from file, shape=(n,4,4)."""
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
            # npz 등 다른 확장자일 경우
            poses = np.load(pose_path)["arr_0"]
    except FileNotFoundError:
        print("Ground truth poses are not available.")
    return np.array(poses)


def load_calib(calib_path):
    """Load calibrations (T_cam_velo) from file, shape=(4,4)."""
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
    """Load 3D points of a scan (KITTI .bin). Returns n×4 (x, y, z, 1)."""
    current_vertex = np.fromfile(scan_path, dtype=np.float32)
    current_vertex = current_vertex.reshape((-1, 4))
    current_points = current_vertex[:, 0:3]
    current_vertex = np.ones((current_points.shape[0], current_points.shape[1] + 1))
    current_vertex[:, :-1] = current_points
    return current_vertex


def load_files(folder):
    """Load and sort all files (e.g. *.bin) in a folder."""
    file_paths = [
        os.path.join(dp, f)
        for dp, dn, fn in os.walk(os.path.expanduser(folder))
        for f in fn
    ]
    file_paths.sort()
    return file_paths


###############################################################################
#                          Bird Eye View Projection 함수                      #
###############################################################################
def bev_projection(
    current_vertex, proj_H=512, proj_W=512, max_range=50.0, min_range=2.0
):
    """
    current_vertex: n×4 (x, y, z, 1)
    proj_H, proj_W: BEV 이미지 해상도
    max_range, min_range: XY평면 거리 필터링 범위
    리턴 shape: (proj_H, proj_W, 4)
      [:,:,0:3] = (x, y, z)
      [:,:,3]   = XY평면 거리(= BEV 상 'range')
    """
    # 1) XY 평면 거리
    xy_dist = np.sqrt(current_vertex[:, 0] ** 2 + current_vertex[:, 1] ** 2)

    # 2) 필터링
    valid_mask = (xy_dist > min_range) & (xy_dist < max_range)
    current_vertex = current_vertex[valid_mask]
    xy_dist = xy_dist[valid_mask]

    # 유효 포인트가 없으면 -1로 채워진 맵 반환
    if current_vertex.shape[0] == 0:
        return np.full((proj_H, proj_W, 4), -1, dtype=np.float32)

    # 3) [-max_range, +max_range] -> [0,1] -> [0, proj_H or proj_W]
    x_img = (current_vertex[:, 0] + max_range) / (2.0 * max_range)
    y_img = (current_vertex[:, 1] + max_range) / (2.0 * max_range)
    x_img = np.floor(x_img * proj_W).astype(np.int32)
    y_img = np.floor(y_img * proj_H).astype(np.int32)

    # 범위 클리핑
    x_img = np.clip(x_img, 0, proj_W - 1)
    y_img = np.clip(y_img, 0, proj_H - 1)

    # 4) 깊이에 해당하는 거리 기준 내림차순 소팅
    order = np.argsort(xy_dist)[::-1]
    xy_dist = xy_dist[order]
    current_vertex = current_vertex[order]
    x_img = x_img[order]
    y_img = y_img[order]

    # 5) proj_vertex 생성
    proj_vertex = np.full((proj_H, proj_W, 4), -1, dtype=np.float32)
    proj_vertex[y_img, x_img, 0] = current_vertex[:, 0]
    proj_vertex[y_img, x_img, 1] = current_vertex[:, 1]
    proj_vertex[y_img, x_img, 2] = current_vertex[:, 2]
    proj_vertex[y_img, x_img, 3] = xy_dist  # range(수평 거리)

    return proj_vertex


###############################################################################
#            Range residual 코드와 동일한 순서로 BEV residual 생성 함수       #
###############################################################################
def process_one_seq_bev(config):
    """
    Range residual과 동일한 포즈 변환 순서를 따라
    BEV 투영 방식으로 residual 이미지를 생성.
    """
    # 파라미터
    num_frames = config["num_frames"]
    debug = config["debug"]
    normalize = config["normalize"]
    num_last_n = config["num_last_n"]
    visualize = config["visualize"]

    # 폴더 설정
    bev_residual_folder = config["bev_residual_folder"]
    check_and_makedirs(bev_residual_folder)

    visualization_folder = config["visualization_folder_bev"]
    if visualize:
        check_and_makedirs(visualization_folder)

    # Pose 로드
    pose_file = config["pose_file"]
    poses = load_poses(pose_file)
    if len(poses) == 0:
        print(f"[Error] No poses found or file missing: {pose_file}")
        return

    inv_frame0 = np.linalg.inv(poses[0])  # 첫 프레임 기준

    # Calib 로드
    calib_file = config["calib_file"]
    T_cam_velo = load_calib(calib_file)
    if T_cam_velo.shape != (4, 4):
        print(f"[Error] Invalid calib file: {calib_file}")
        return
    T_velo_cam = np.linalg.inv(T_cam_velo)

    # Pose를 라이다 좌표계 기준으로 변환
    new_poses = []
    for pose in poses:
        new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
    poses = np.array(new_poses)

    # 스캔 파일
    scan_folder = config["scan_folder"]
    scan_paths = load_files(scan_folder)
    if len(scan_paths) == 0:
        print(f"[Error] No scan files in: {scan_folder}")
        return

    # 프레임 제한
    if num_frames >= len(poses) or num_frames <= 0:
        print(f"Generate BEV residual for all frames: {len(poses)}")
    else:
        poses = poses[:num_frames]
        scan_paths = scan_paths[:num_frames]

    # BEV 파라미터
    bev_params = config["bev_image"]

    # 시퀀스 전체 프레임에 대해 residual 생성
    for frame_idx in tqdm(range(len(scan_paths)), desc=f"num_last_n={num_last_n}"):
        file_name = os.path.join(bev_residual_folder, str(frame_idx).zfill(6))
        diff_image = np.zeros(
            (bev_params["height"], bev_params["width"]), dtype=np.float32
        )

        # 앞쪽 num_last_n 프레임은 dummy
        if frame_idx < num_last_n:
            np.save(file_name, diff_image)
            continue

        # 현재 스캔
        current_pose = poses[frame_idx]
        current_scan = load_vertex(scan_paths[frame_idx])
        current_bev = bev_projection(
            current_scan.astype(np.float32),
            proj_H=bev_params["height"],
            proj_W=bev_params["width"],
            max_range=bev_params["max_range"],
            min_range=bev_params["min_range"],
        )
        current_range = current_bev[:, :, 3]

        # 이전 스캔
        last_pose = poses[frame_idx - num_last_n]
        last_scan = load_vertex(scan_paths[frame_idx - num_last_n])
        # 현재 좌표계로 변환
        last_scan_transformed = (
            np.linalg.inv(current_pose).dot(last_pose).dot(last_scan.T).T
        )

        last_bev_transformed = bev_projection(
            last_scan_transformed.astype(np.float32),
            proj_H=bev_params["height"],
            proj_W=bev_params["width"],
            max_range=bev_params["max_range"],
            min_range=bev_params["min_range"],
        )
        last_range_transformed = last_bev_transformed[:, :, 3]

        # Residual 계산
        valid_mask = (
            (current_range > bev_params["min_range"])
            & (current_range < bev_params["max_range"])
            & (last_range_transformed > bev_params["min_range"])
            & (last_range_transformed < bev_params["max_range"])
        )

        difference = np.abs(
            current_range[valid_mask] - last_range_transformed[valid_mask]
        )
        if normalize:
            difference = difference / np.clip(current_range[valid_mask], 1e-6, None)

        diff_image[valid_mask] = difference

        # debug 모드 시 시각화
        if debug:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            img1 = axs[0].imshow(
                last_range_transformed,
                cmap="viridis",
                vmin=0,
                vmax=bev_params["max_range"],
            )
            axs[0].set_title("Last Range (BEV)")
            fig.colorbar(img1, ax=axs[0], fraction=0.046, pad=0.04)

            img2 = axs[1].imshow(
                current_range, cmap="viridis", vmin=0, vmax=bev_params["max_range"]
            )
            axs[1].set_title("Current Range (BEV)")
            fig.colorbar(img2, ax=axs[1], fraction=0.046, pad=0.04)

            img3 = axs[2].imshow(diff_image, cmap="viridis", vmin=0, vmax=1)
            axs[2].set_title("BEV Residual")
            fig.colorbar(img3, ax=axs[2], fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.show()

        # visualize 옵션 시 PNG 저장
        if visualize:
            image_name = (
                os.path.join(visualization_folder, str(frame_idx).zfill(6)) + ".png"
            )
            plt.imsave(image_name, diff_image, cmap="viridis")

        # 결과 .npy 파일로 저장
        np.save(file_name, diff_image)


###############################################################################
#                        단일 (seq_id, num_last_n) 처리 함수                 #
###############################################################################
def process_residual_for_num_last_n(seq_id, num_last_n, dataset_folder):
    """
    (seq_id, num_last_n) 조합에 대해 BEV residual 생성.
    병렬로 실행될 단위 작업 함수.
    """
    scan_folder = os.path.join(dataset_folder, f"{seq_id:02d}", "velodyne")
    pose_file = os.path.join(dataset_folder, f"{seq_id:02d}", "poses.txt")
    calib_file = os.path.join(dataset_folder, f"{seq_id:02d}", "calib.txt")

    bev_residual_folder = os.path.join(
        dataset_folder, f"{seq_id:02d}", f"bev_residual_images_{num_last_n}"
    )
    visualization_folder = os.path.join(
        dataset_folder, f"{seq_id:02d}", f"vis_bev_residual_{num_last_n}"
    )

    # config dict
    config = {
        "num_frames": -1,  # -1이면 전체 프레임
        "debug": False,  # True 시 디버그 시각화
        "normalize": True,  # range normalize
        "num_last_n": num_last_n,  # offset
        "visualize": False,  # PNG 저장 여부
        "bev_residual_folder": bev_residual_folder,
        "visualization_folder_bev": visualization_folder,
        "pose_file": pose_file,
        "calib_file": calib_file,
        "scan_folder": scan_folder,
        "bev_image": {
            "height": 360,
            "width": 360,
            "max_range": 50.0,
            "min_range": 2.0,
        },
    }

    process_one_seq_bev(config)


###############################################################################
#                      시퀀스 하나당 num_last_n 병렬 처리 함수               #
###############################################################################
def process_sequence_in_parallel(seq_id, dataset_folder):
    """
    단일 seq_id에 대해 num_last_n = 1~8 등 여러 값을 병렬 처리한 뒤,
    모두 끝나면 "Sequence X done" 메시지 출력.
    """
    num_workers = multiprocessing.cpu_count() - 4  # 필요에 따라 조정
    # 병렬 처리할 num_last_n 범위
    num_last_n_list = range(1, 9)
    tasks = [(seq_id, n, dataset_folder) for n in num_last_n_list]

    with multiprocessing.Pool(num_workers) as pool:
        pool.starmap(process_residual_for_num_last_n, tasks)

    print(
        f"[Done] Sequence {seq_id:02d} - all {len(num_last_n_list)} residuals created"
    )


###############################################################################
#                                 메인 실행 예시                               #
###############################################################################
if __name__ == "__main__":
    dataset_folder = "/home/work_docker/KITTI/dataset/sequences"
    seq_list = range(11)  # 0 ~ 10

    for seq_id in seq_list:
        # seq_id 하나에 대해 num_last_n=1~8을 병렬로 처리
        process_sequence_in_parallel(seq_id, dataset_folder)

    print("모든 시퀀스의 BEV residual 생성이 완료되었습니다.")
