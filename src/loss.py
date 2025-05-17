import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        total_loss = 0
        for pred, target in zip(preds, targets):
            target = target.to(pred.device)
            n = min(pred.shape[0], target.shape[0])
            if n == 0:
                continue
            pred_boxes = pred[:, :4]
            pred_conf = pred[:, 4]



            conf_sorted_idx = torch.argsort(pred_conf, descending=True)[:n]
            selected_preds = pred_boxes[conf_sorted_idx]
            loss = F.mse_loss(selected_preds, target[:n])
            total_loss += loss
        return total_loss / len(preds)
