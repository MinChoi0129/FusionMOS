import torch
import torch.nn as nn
from torch.autograd import Variable


def lovasz_grad(gt_sorted):
    """
    Lovasz gradient 계산 (간단화된 버전)
    gt_sorted: 내림차순 정렬된 이진 ground truth (0 또는 1)
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0 : p - 1]
    return jaccard


def lovasz_softmax_flat(probas, labels):
    """
    단순화된 Lovasz-Softmax loss (이진 분류처럼 처리)
      probas: [P, 2] 텐서, 채널 0: static, 채널 1: moving 확률
      labels: [P] 텐서, -1: ignore, 0: static, 1: moving
    """
    # print(
    #     torch.unique(labels),
    #     torch.sum(labels == -1).item(),
    #     torch.sum(labels == 0).item(),
    #     torch.sum(labels == 1).item(),
    # )
    valid = labels != -1
    if valid.sum() == 0:
        print("lovasz_softmax_flat: 유효한 label이 없습니다. 0 반환")
        return torch.tensor(0.0, device=probas.device, dtype=probas.dtype)

    probas_valid = probas[valid]
    labels_valid = labels[valid].float()

    # moving (foreground) 확률: 두 번째 채널 사용
    p_pred = probas_valid[:, 1]
    errors = (labels_valid - p_pred).abs()
    errors_sorted, perm = torch.sort(errors, descending=True)
    fg_sorted = labels_valid[perm]
    grad = lovasz_grad(fg_sorted)
    loss = torch.dot(errors_sorted, grad)
    return loss


class Lovasz_softmax_PointCloud(nn.Module):
    def __init__(self):
        super(Lovasz_softmax_PointCloud, self).__init__()

    def forward(self, probas, labels):
        """
        probas: [B, C, N] 텐서, 채널 순서: 0: unlabeled, 1: static, 2: moving
        labels: [B, N] 텐서, 값은 0 (ignore), 1 (static), 2 (moving)
        """
        B, C, N = probas.size()
        assert C == 3, "probas는 채널 3개여야 합니다."

        # 채널 0(unlabeled)은 무시하고, 채널 1, 2만 사용
        probas_bin = probas[:, 1:, :]  # [B, 2, N]
        # [B, 2, N] -> [B*N, 2]
        probas_bin = probas_bin.permute(0, 2, 1).contiguous().view(-1, 2)

        labels = labels.view(-1)  # [B*N]
        # label 변환: 0 (ignore) -> -1, 1 (static) -> 0, 2 (moving) -> 1
        labels_bin = labels.clone()
        labels_bin[labels == 0] = -1
        labels_bin[labels == 1] = 0
        labels_bin[labels == 2] = 1

        loss = lovasz_softmax_flat(probas_bin, labels_bin)
        return loss
