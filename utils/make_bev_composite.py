import os
import numpy as np
import h5py
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import torch

# 매핑 딕셔너리 (moving, movable)
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


def load_scan_with_remission(scan_path):
    """
    KITTI .bin 파일에서 3D 포인트와 remission을 읽어옴.
    반환: homo_points (n×4, homogeneous), remissions (n,)
    """
    scan = np.fromfile(scan_path, dtype=np.float32).reshape((-1, 4))
    points = scan[:, :3]
    remissions = scan[:, 3]
    homo_points = np.ones((points.shape[0], 4), dtype=np.float32)
    homo_points[:, :3] = points
    return homo_points, remissions


def load_label(label_path):
    """
    .label 파일에서 semantic 라벨과 instance 라벨을 읽어옴.
    반환: sem_label (n,), inst_label (n,)
    """
    label = np.fromfile(label_path, dtype=np.int32)
    sem_label = label & 0xFFFF
    inst_label = label >> 16
    return sem_label, inst_label


def bev_projection_full(
    homo_points, remissions, proj_H, proj_W, max_range=50.0, min_range=2.0
):
    """
    입력:
      homo_points: n×4 (x, y, z, 1)
      remissions: n, float
    출력:
      bev_range: (proj_H, proj_W) 각 픽셀의 XY 평면 거리 (없으면 -1)
      bev_xyz:   (proj_H, proj_W, 3) 각 픽셀에 매핑된 (x, y, z) 좌표 (없으면 -1)
      bev_remission: (proj_H, proj_W) 각 픽셀의 remission 값 (없으면 -1)
      bev_proj_x:  유효 포인트들의 x 픽셀 좌표 (1D 배열)
      bev_proj_y:  유효 포인트들의 y 픽셀 좌표 (1D 배열)
      bev_unproj_range: 유효 포인트들의 원본 XY 거리 값 (1D 배열)
    """
    xy_dist = np.sqrt(homo_points[:, 0] ** 2 + homo_points[:, 1] ** 2)
    valid_mask = (xy_dist > min_range) & (xy_dist < max_range)
    filtered_points = homo_points[valid_mask]
    filtered_rem = remissions[valid_mask]
    filtered_dist = xy_dist[valid_mask]

    if filtered_points.shape[0] == 0:
        bev_range = np.full((proj_H, proj_W), -1, dtype=np.float32)
        bev_xyz = np.full((proj_H, proj_W, 3), -1, dtype=np.float32)
        bev_remission = np.full((proj_H, proj_W), -1, dtype=np.float32)
        return (
            bev_range,
            bev_xyz,
            bev_remission,
            np.array([]),
            np.array([]),
            np.array([]),
        )

    # x, y 좌표를 [0,1]로 정규화 후 이미지 크기에 맞게 변환
    x_img = (filtered_points[:, 0] + max_range) / (2.0 * max_range)
    y_img = (filtered_points[:, 1] + max_range) / (2.0 * max_range)
    x_img = np.clip(np.floor(x_img * proj_W), 0, proj_W - 1).astype(np.int32)
    y_img = np.clip(np.floor(y_img * proj_H), 0, proj_H - 1).astype(np.int32)

    # 먼 점부터 투영 (덮어쓰기를 위해)
    order = np.argsort(filtered_dist)[::-1]
    x_img = x_img[order]
    y_img = y_img[order]
    filtered_dist = filtered_dist[order]
    filtered_points = filtered_points[order]
    filtered_rem = filtered_rem[order]

    bev_range = np.full((proj_H, proj_W), -1, dtype=np.float32)
    bev_xyz = np.full((proj_H, proj_W, 3), -1, dtype=np.float32)
    bev_remission = np.full((proj_H, proj_W), -1, dtype=np.float32)

    bev_range[y_img, x_img] = filtered_dist
    bev_xyz[y_img, x_img, :] = filtered_points[:, :3]
    bev_remission[y_img, x_img] = filtered_rem

    return bev_range, bev_xyz, bev_remission, x_img, y_img, filtered_dist


def bev_label_projection_full(
    homo_points, sem_label, proj_H, proj_W, max_range=50.0, min_range=2.0
):
    """
    homo_points: n×4 (x, y, z, 1)
    sem_label: (n,) semantic 라벨
    출력: bev_sem_label, shape (proj_H, proj_W) (-1: empty)
    """
    xy_dist = np.sqrt(homo_points[:, 0] ** 2 + homo_points[:, 1] ** 2)
    valid_mask = (xy_dist > min_range) & (xy_dist < max_range)
    filtered_points = homo_points[valid_mask]
    filtered_labels = sem_label[valid_mask]
    filtered_dist = xy_dist[valid_mask]
    if filtered_points.shape[0] == 0:
        return np.full((proj_H, proj_W), -1, dtype=np.int32)
    x_img = (filtered_points[:, 0] + max_range) / (2.0 * max_range)
    y_img = (filtered_points[:, 1] + max_range) / (2.0 * max_range)
    x_img = np.clip(np.floor(x_img * proj_W), 0, proj_W - 1).astype(np.int32)
    y_img = np.clip(np.floor(y_img * proj_H), 0, proj_H - 1).astype(np.int32)
    order = np.argsort(filtered_dist)[::-1]
    x_img = x_img[order]
    y_img = y_img[order]
    filtered_labels = filtered_labels[order]
    bev_sem_label = np.full((proj_H, proj_W), -1, dtype=np.int32)
    bev_sem_label[y_img, x_img] = filtered_labels
    return bev_sem_label


def map_labels(label_array, mapping_dict):
    mapped = np.full_like(label_array, -1, dtype=np.int32)
    unique = np.unique(label_array)
    for val in unique:
        if val in mapping_dict:
            mapped[label_array == val] = mapping_dict[val]
        else:
            mapped[label_array == val] = 0
    return mapped


def create_ternary_label(bev_label, mapping_dict):
    """
    bev_label: BEV label 이미지 (2D, -1이면 empty)
    mapping_dict: 예) moving_learning_map 또는 movable_learning_map
    반환: 삼진 이미지: 빈 공간은 0, 그 외에는 mapping된 값 (예: static=1, moving=2)
    """
    mapped = map_labels(bev_label, mapping_dict)
    ternary = np.where(bev_label == -1, 0, mapped)
    return ternary


class BevScan:
    """
    BEV Scan 클래스:
      - velodyne 스캔과 라벨 파일을 받아 BEV 투영을 수행하여,
        bev_range, bev_xyz, bev_remission, bev_sem_label과 함께
        bev_proj_x, bev_proj_y, bev_unproj_range도 생성.
    """

    def __init__(self, proj_H, proj_W, max_range=50.0, min_range=2.0):
        self.proj_H = proj_H
        self.proj_W = proj_W
        self.max_range = max_range
        self.min_range = min_range

    def process(self, scan_path, label_path):
        homo_points, remissions = load_scan_with_remission(scan_path)
        sem_label, _ = load_label(label_path)
        (
            self.bev_range,
            self.bev_xyz,
            self.bev_remission,
            self.bev_proj_x,
            self.bev_proj_y,
            self.bev_unproj_range,
        ) = bev_projection_full(
            homo_points,
            remissions,
            proj_H=self.proj_H,
            proj_W=self.proj_W,
            max_range=self.max_range,
            min_range=self.min_range,
        )
        self.bev_sem_label = bev_label_projection_full(
            homo_points,
            sem_label,
            proj_H=self.proj_H,
            proj_W=self.proj_W,
            max_range=self.max_range,
            min_range=self.min_range,
        )
        return (
            self.bev_range,
            self.bev_xyz,
            self.bev_remission,
            self.bev_proj_x,
            self.bev_proj_y,
            self.bev_unproj_range,
            self.bev_sem_label,
        )


def process_one_frame_h5(velodyne_path, label_path, output_path):
    proj_H, proj_W = 768, 768
    max_range = 50.0
    min_range = 2.0

    bev_scan = BevScan(proj_H, proj_W, max_range, min_range)
    (
        bev_range_img,
        bev_xyz,
        bev_remission,
        bev_proj_x,
        bev_proj_y,
        bev_unproj_range,
        bev_sem_label,
    ) = bev_scan.process(velodyne_path, label_path)

    # 정규화: 범위와 remission 채널
    norm_range = np.clip(bev_range_img, 0, max_range) / max_range
    if np.any(bev_remission > 0):
        norm_rem = np.clip(
            bev_remission, 0, np.max(bev_remission[bev_remission > 0])
        ) / np.max(bev_remission[bev_remission > 0])
    else:
        norm_rem = bev_remission

    bev_moving_ternary = create_ternary_label(bev_sem_label, moving_learning_map)
    bev_movable_ternary = create_ternary_label(bev_sem_label, movable_learning_map)

    bev_composite = np.concatenate(
        [
            norm_range[np.newaxis, ...],
            bev_xyz.transpose(2, 0, 1),
            norm_rem[np.newaxis, ...],
            bev_moving_ternary[np.newaxis, ...],
            bev_movable_ternary[np.newaxis, ...],
        ],
        axis=0,
    )

    unproj_n_points = bev_proj_x.shape[0]
    tmp_x = torch.full([150000], -1, dtype=torch.long)
    tmp_x[:unproj_n_points] = torch.from_numpy(bev_proj_x)
    tmp_y = torch.full([150000], -1, dtype=torch.long)
    tmp_y[:unproj_n_points] = torch.from_numpy(bev_proj_y)
    tmp_unproj_range = torch.full([150000], -1.0, dtype=torch.float)
    tmp_unproj_range[:unproj_n_points] = torch.from_numpy(bev_unproj_range)

    # h5 파일로 저장
    with h5py.File(output_path, "w") as f:
        f.create_dataset("bev_composite", data=bev_composite)
        f.create_dataset("bev_proj_x", data=tmp_x.cpu().numpy())
        f.create_dataset("bev_proj_y", data=tmp_y.cpu().numpy())
        # f.create_dataset("bev_proj_range", data=bev_range_img)
        f.create_dataset("bev_unproj_range", data=tmp_unproj_range.cpu().numpy())
    print(f"Saved: {output_path}")


# 시퀀스 단위 처리 (h5 저장 버전)
dataset_sequences_path = "/home/ssd_4tb/minjae/KITTI/dataset/sequences"
output_root = "/home/ssd_4tb/minjae/KITTI/test_output_h5"
num_workers = mp.cpu_count()


def process_frame(seq_id, frame_file):
    velodyne_folder = os.path.join(dataset_sequences_path, "%02d" % seq_id, "velodyne")
    labels_folder = os.path.join(dataset_sequences_path, "%02d" % seq_id, "labels")
    output_folder = os.path.join(output_root, "%02d" % seq_id)
    os.makedirs(output_folder, exist_ok=True)

    frame_id = int(frame_file[:6])
    velodyne_path = os.path.join(velodyne_folder, "%06d.bin" % frame_id)
    label_path = os.path.join(labels_folder, "%06d.label" % frame_id)
    output_path = os.path.join(output_folder, "%06d.h5" % frame_id)
    process_one_frame_h5(velodyne_path, label_path, output_path)


def process_one_sequence(seq_id):
    velodyne_folder = os.path.join(dataset_sequences_path, "%02d" % seq_id, "velodyne")
    frame_files = os.listdir(velodyne_folder)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        list(
            tqdm(
                executor.map(lambda frame: process_frame(seq_id, frame), frame_files),
                total=len(frame_files),
                desc=f"Processing Seq {seq_id}",
            )
        )
    print(f"Sequence {seq_id} 완료")


if __name__ == "__main__":
    with mp.Pool(processes=num_workers // 4) as pool:
        pool.map(process_one_sequence, range(11))
