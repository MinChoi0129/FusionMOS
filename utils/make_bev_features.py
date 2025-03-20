import torch, os
import numpy as np
from pointpillars.model import PointPillars
from tqdm import tqdm

def point_range_filter(pts, point_range):
    """
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    """
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = (
        flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    )
    pts = pts[keep_mask]
    return pts

model_path = "/home/workspace/work/PointPillars/pretrained/epoch_160.pth"
model = PointPillars(nclasses=3).cuda()
model.load_state_dict(torch.load(model_path))

model.eval()

base_path = "/home/workspace/KITTI/dataset/sequences"

for seq_id in range(0, 11):
    print(f"Trying {seq_id:02d}...")
    for frame_id in tqdm(range(len(os.listdir(os.path.join(base_path, f"{seq_id:02d}", "velodyne"))))):
        pcd_path = os.path.join(base_path, f"{seq_id:02d}", "velodyne", f"{frame_id:06d}.bin")
        save_folder = os.path.join(base_path, f"{seq_id:02d}", "bev_features")

        save_path = os.path.join(base_path, f"{seq_id:02d}", "bev_features", f"{frame_id:06d}.npz")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder, exist_ok=True)
            
        with torch.no_grad():
            pcd = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 4)
            pcd = point_range_filter(pcd, point_range=[-50, -50, -50, 50, 50, 50])
            pcd_torch = torch.from_numpy(pcd).cuda()
            all_features = model(batched_pts=[pcd_torch], mode="test")

            f1 = all_features[0][0].cpu().numpy()
            f2 = all_features[1][0].cpu().numpy()
            f3 = all_features[2][0].cpu().numpy()
            np.savez_compressed(save_path, f1=f1, f2=f2, f3=f3)