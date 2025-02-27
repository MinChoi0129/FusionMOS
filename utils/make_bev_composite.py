import os
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

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
    homo_points, remissions, proj_H=360, proj_W=360, max_range=50.0, min_range=2.0
):
    """
    homo_points: n×4 (x, y, z, 1)
    remissions: n, float
    출력:
      bev_range: (proj_H, proj_W) 각 픽셀의 XY 평면 거리 (없으면 -1)
      bev_xyz:   (proj_H, proj_W, 3) 각 픽셀에 매핑된 (x, y, z) 좌표 (없으면 -1)
      bev_remission: (proj_H, proj_W) 각 픽셀의 remission 값 (없으면 -1)
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
        return bev_range, bev_xyz, bev_remission

    # 정규화: x,y를 [-max_range, max_range] -> [0,1]
    x_img = (filtered_points[:, 0] + max_range) / (2.0 * max_range)
    y_img = (filtered_points[:, 1] + max_range) / (2.0 * max_range)
    x_img = np.clip(np.floor(x_img * proj_W), 0, proj_W - 1).astype(np.int32)
    y_img = np.clip(np.floor(y_img * proj_H), 0, proj_H - 1).astype(np.int32)

    # 거리가 큰 순으로 정렬 (먼 점부터 채워서 가까운 점이 덮어쓰도록)
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
    return bev_range, bev_xyz, bev_remission


def bev_label_projection_full(
    homo_points, sem_label, proj_H=360, proj_W=360, max_range=50.0, min_range=2.0
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
        bev_range (거리), bev_xyz (3채널), bev_remission, bev_sem_label을 생성.
    """

    def __init__(self, proj_H=360, proj_W=360, max_range=50.0, min_range=2.0):
        self.proj_H = proj_H
        self.proj_W = proj_W
        self.max_range = max_range
        self.min_range = min_range

    def process(self, scan_path, label_path):
        homo_points, remissions = load_scan_with_remission(scan_path)
        sem_label, _ = load_label(label_path)
        self.bev_range, self.bev_xyz, self.bev_remission = bev_projection_full(
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
        return self.bev_range, self.bev_xyz, self.bev_remission, self.bev_sem_label


def process_one_frame(velodyne_path, label_path, output_path):
    proj_H, proj_W = 360, 360
    max_range = 50.0
    min_range = 2.0

    # BEV Scan 클래스 생성 및 처리 (현재 프레임)
    bev_scan = BevScan(proj_H, proj_W, max_range, min_range)
    bev_range_img, bev_xyz, bev_remission, bev_sem_label = bev_scan.process(
        velodyne_path, label_path
    )

    # 1. Composite BEV 이미지: R: bev_range, G: 평균 bev_xyz, B: bev_remission (정규화)
    norm_range = np.clip(bev_range_img, 0, max_range) / max_range
    norm_rem = np.clip(bev_remission, 0, np.max(bev_remission[bev_remission > 0]))
    if np.any(norm_rem > 0):
        norm_rem = norm_rem / np.max(norm_rem)
    else:
        norm_rem = norm_rem

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

    # print(bev_composite.shape)

    np.save(output_path, bev_composite)


dataset_sequences_path = "/home/work_docker/KITTI/dataset/sequences"
output_root = "/home/work_docker/KITTI/test_output"
num_workers = mp.cpu_count()  # 사용 가능한 CPU 코어 개수


def process_frame(seq_id, frame_id_with_bin):
    velodyne_folder = os.path.join(dataset_sequences_path, "%02d" % seq_id, "velodyne")
    labels_folder = os.path.join(dataset_sequences_path, "%02d" % seq_id, "labels")
    output_folder = os.path.join(output_root, "%02d" % seq_id)
    os.makedirs(output_folder, exist_ok=True)

    frame_id = int(frame_id_with_bin[:6])
    velodyne_path = os.path.join(velodyne_folder, "%06d.bin" % frame_id)
    label_path = os.path.join(labels_folder, "%06d.label" % frame_id)
    output_path = os.path.join(output_folder, "%06d.npy" % frame_id)
    process_one_frame(velodyne_path, label_path, output_path)


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

    print(seq_id, "완료")


if __name__ == "__main__":
    with mp.Pool(processes=num_workers - 6) as pool:
        pool.map(process_one_sequence, range(11))
