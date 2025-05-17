import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        total_loss = 0
        for pred, target in zip(preds, targets):
            target.to(pred.device)
            if target.shape[0] == 0:
                continue
            pred_boxes = pred[:, :4]
            pred_conf = pred[:, 4]

            conf_sorted_idx = torch.argsort(pred_conf, descending=True)[:target.shape[0]]
            selected_preds = pred_boxes[conf_sorted_idx]

            loss = F.mse_loss(selected_preds, target)
            total_loss += loss
        return total_loss / len(preds)
