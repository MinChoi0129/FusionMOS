import numpy as np

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
        filtered_dist = filtered_dist[order]

        bev_range = np.full((proj_H, proj_W), 0, dtype=np.float32)
        bev_range[y_img, x_img] = 1

        return bev_range

    def bev_projection_full(self):
        xy_dist = np.sqrt(self.homo_points[:, 0] ** 2 + self.homo_points[:, 1] ** 2)
        valid_mask = (xy_dist > self.min_range) & (xy_dist < self.max_range)
        filtered_points = self.homo_points[valid_mask]
        filtered_rem = self.remissions[valid_mask]
        filtered_dist = xy_dist[valid_mask]
        orig_idx = np.nonzero(valid_mask)[0]  # 원본 점 인덱스

        if filtered_points.shape[0] == 0:
            empty = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
            return (
                empty,
                np.full((self.proj_H, self.proj_W, 3), -1, dtype=np.float32),
                empty,
                np.array([]),
                np.array([]),
                np.array([]),
                np.full((self.proj_H, self.proj_W), -1, dtype=np.int32),
            )

        # x, y 좌표를 [0,1]로 정규화 후 이미지 크기에 맞게 변환
        x_img = (filtered_points[:, 0] + self.max_range) / (2.0 * self.max_range)
        y_img = (filtered_points[:, 1] + self.max_range) / (2.0 * self.max_range)
        x_img = np.clip(np.floor(x_img * self.proj_W), 0, self.proj_W - 1).astype(np.int32)
        y_img = np.clip(np.floor(y_img * self.proj_H), 0, self.proj_H - 1).astype(np.int32)

        # 먼 점부터 투영 (덮어쓰기를 위해)
        order = np.argsort(filtered_dist)[::-1]
        x_img = x_img[order]
        y_img = y_img[order]
        filtered_dist = filtered_dist[order]
        filtered_points = filtered_points[order]
        filtered_rem = filtered_rem[order]
        proj_idx_ordered = orig_idx[order]

        bev_range = np.full((self.proj_H, self.proj_W), 0, dtype=np.float32)
        bev_xyz = np.full((self.proj_H, self.proj_W, 3), -1, dtype=np.float32)
        bev_remission = np.full((self.proj_H, self.proj_W), -1, dtype=np.float32)
        proj_idx = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)

        bev_range[y_img, x_img] = 1
        bev_xyz[y_img, x_img, :] = filtered_points[:, :3]
        bev_remission[y_img, x_img] = filtered_rem
        proj_idx[y_img, x_img] = proj_idx_ordered


        # 정규화: range와 remission 채널
        # norm_range = np.clip(bev_range, 0, self.max_range) / self.max_range
        if np.any(bev_remission > 0):
            norm_rem = np.clip(
                bev_remission, 0, np.max(bev_remission[bev_remission > 0])
            ) / np.max(bev_remission[bev_remission > 0])
        else:
            norm_rem = bev_remission

        unproj_n_points = x_img.shape[0]
        tmp_x = np.full(150000, -1, dtype=np.longlong)
        tmp_x[:unproj_n_points] = x_img
        tmp_y = np.full(150000, -1, dtype=np.longlong)
        tmp_y[:unproj_n_points] = y_img
        tmp_unproj_range = np.full(150000, -1.0, dtype=np.float32)
        tmp_unproj_range[:unproj_n_points] = filtered_dist
            
        self.bev_range = bev_range
        self.bev_xyz = bev_xyz.transpose((2, 0, 1))  # HWC -> CHW
        self.bev_remission = norm_rem
        self.bev_proj_x = tmp_x
        self.bev_proj_y = tmp_y
        self.bev_unproj_range = tmp_unproj_range
        self.bev_proj_idx = proj_idx
    
        # return self.bev_range, self.bev_xyz, self.bev_remission, self.bev_proj_x, self.bev_proj_y, self.bev_unproj_range, self.bev_proj_idx
    
    def bev_labels_full(self):
        self.bev_sem_label = np.full((self.proj_H, self.proj_W), -1, dtype=np.int32)
        valid = self.bev_proj_idx >= 0
        self.bev_sem_label[valid] = self.sem_label[self.bev_proj_idx[valid]]

        self.bev_moving_ternary_label = self.create_ternary_label(BevScan.moving_learning_map)
        self.bev_movable_ternary_label = self.create_ternary_label(BevScan.movable_learning_map)

        # return self.bev_sem_label, self.bev_moving_ternary_label, self.bev_movable_ternary_label

    def process(self):
        self.load_velodyne()
        self.load_label()
        self.bev_projection_full()
        self.bev_labels_full()