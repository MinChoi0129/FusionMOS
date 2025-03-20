import torch


def loss_and_pred(probs, labels, each_criterion, ls_func):
    """
    probas : ( 3, sum(ith-#points) )
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
