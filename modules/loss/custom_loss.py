import torch


def loss_and_pred(probs, labels, each_criterion, ls_func, mode, *args_2D):
    if mode == "2D":
        """
        loss 계산은 2D로, pred는 3D로
        probs : (bs, 3, h, w)
        labels : (bs, h, w)
        """

        jacc = ls_func(probs, labels)
        log_probs = torch.log(probs.clamp(min=1e-8)).double()
        wce = each_criterion(log_probs, labels).float()
        loss = wce + jacc

        proj_ys, proj_xs, npoints = args_2D
        preds = []

        for prob, proj_y, proj_x, n in zip(probs, proj_ys, proj_xs, npoints):
            pred = prob[:, proj_y[:n], proj_x[:n]].argmax(dim=0)  # pred: (num_points, )
            preds.append(pred)  # 리스트에 텐서 형태로 추가

        preds = torch.cat(preds)  # 모든 텐서를 연결하여 하나의 1D 텐서로 만듦

        return loss, preds, (jacc, wce)

    elif mode == "3D":
        """
        probs : ( 3, sum(ith-#points) )
        labels : ( sum(ith-#points), )
        """
        if probs.numel() == 0:
            raise ValueError("No unprojected predictions found")

        jacc = ls_func(probs, labels)

        log_probs = torch.log(probs.clamp(min=1e-8))
        wce = each_criterion(log_probs.T.double(), labels).float()

        loss = wce + jacc

        preds = probs.argmax(dim=0)  # (sum(ith-#points), )

        return loss, preds, (jacc, wce)
