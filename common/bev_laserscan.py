import numpy as np
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
      bev_proj_idx: 각 픽셀에 대응하는 원본 점의 인덱스 (-1: no data)
    """
    xy_dist = np.sqrt(homo_points[:, 0] ** 2 + homo_points[:, 1] ** 2)
    valid_mask = (xy_dist > min_range) & (xy_dist < max_range)
    filtered_points = homo_points[valid_mask]
    filtered_rem = remissions[valid_mask]
    filtered_dist = xy_dist[valid_mask]
    orig_idx = np.nonzero(valid_mask)[0]  # 원본 점 인덱스

    if filtered_points.shape[0] == 0:
        empty = np.full((proj_H, proj_W), -1, dtype=np.float32)
        return (
            empty,
            np.full((proj_H, proj_W, 3), -1, dtype=np.float32),
            empty,
            np.array([]),
            np.array([]),
            np.array([]),
            np.full((proj_H, proj_W), -1, dtype=np.int32),
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
    proj_idx_ordered = orig_idx[order]

    bev_range = np.full((proj_H, proj_W), -1, dtype=np.float32)
    bev_xyz = np.full((proj_H, proj_W, 3), -1, dtype=np.float32)
    bev_remission = np.full((proj_H, proj_W), -1, dtype=np.float32)
    proj_idx = np.full((proj_H, proj_W), -1, dtype=np.int32)

    bev_range[y_img, x_img] = filtered_dist
    bev_xyz[y_img, x_img, :] = filtered_points[:, :3]
    bev_remission[y_img, x_img] = filtered_rem
    proj_idx[y_img, x_img] = proj_idx_ordered

    return bev_range, bev_xyz, bev_remission, x_img, y_img, filtered_dist, proj_idx


def create_ternary_label(bev_label, mapping_dict):
    """
    bev_label: BEV label 이미지 (2D, -1이면 empty)
    mapping_dict: 예) moving_learning_map 또는 movable_learning_map
    반환: 삼진 이미지: 빈 공간은 0, 그 외에는 mapping된 값 (예: static=1, moving=2)
    """
    mapped = np.full_like(bev_label, -1, dtype=np.int32)
    unique = np.unique(bev_label)
    for val in unique:
        if val in mapping_dict:
            mapped[bev_label == val] = mapping_dict[val]
        else:
            mapped[bev_label == val] = 0
    ternary = np.where(bev_label == -1, 0, mapped)
    return ternary


def process_scan_as_bev(velodyne_path, label_path):
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

    # 정규화: range와 remission 채널
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

    return {
        "bev_composite": bev_composite,
        "bev_proj_x": tmp_x.cpu().numpy(),
        "bev_proj_y": tmp_y.cpu().numpy(),
        "bev_unproj_range": tmp_unproj_range.cpu().numpy(),
    }


class BevScan:
    """
    BEV Scan 클래스:
      - velodyne 스캔과 라벨 파일을 받아 BEV 투영을 수행하여,
        bev_range, bev_xyz, bev_remission, bev_sem_label과 함께
        bev_proj_x, bev_proj_y, bev_unproj_range, bev_proj_idx도 생성.
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
            self.bev_proj_idx,
        ) = bev_projection_full(
            homo_points,
            remissions,
            proj_H=self.proj_H,
            proj_W=self.proj_W,
            max_range=self.max_range,
            min_range=self.min_range,
        )

        # LaserScan 방식 label 채우기: proj_idx를 이용
        self.bev_sem_label = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)
        valid = self.bev_proj_idx >= 0
        self.bev_sem_label[valid] = sem_label[self.bev_proj_idx[valid]]
        return (
            self.bev_range,
            self.bev_xyz,
            self.bev_remission,
            self.bev_proj_x,
            self.bev_proj_y,
            self.bev_unproj_range,
            self.bev_sem_label,
        )
