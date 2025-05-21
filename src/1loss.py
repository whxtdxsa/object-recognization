import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, complete_box_iou_loss

class DetectionLoss(nn.Module):
    def __init__(self, lambda_box=7.5, lambda_conf=3.0, iou_thresh=0.1):
        super().__init__()
        self.lambda_box = lambda_box
        self.lambda_conf = lambda_conf
        self.iou_thresh = iou_thresh
        self.mse = nn.MSELoss()

    def forward(self, preds, targets):
        total_loss = torch.tensor(0.0, device=preds[0].device)

        for batch_idx, (pred, target) in enumerate(zip(preds, targets)):
            target = target.to(pred.device)
            pred_boxes = box_cxcywh_to_xyxy(pred[:, :4])  # [N, 4]
            pred_conf = pred[:, 4]  # [N]
            target_boxes = target[:, :4]  # [M, 4]

            conf_target = torch.zeros_like(pred_conf)
            matched_mask = torch.zeros_like(pred_conf).bool()

            if target_boxes.size(0) > 0:
                ious = box_iou(pred_boxes, target_boxes)  # [N, M]
                best_iou, best_gt = ious.max(dim=1)
                matched_mask = best_iou > self.iou_thresh
                conf_target[matched_mask] = best_iou[matched_mask].detach()
            if conf_target.sum() == 0:
                conf_target[torch.randint(0, len(conf_target), (1,))] = 0.1
            # conf loss (regression)
            conf_loss = self.mse(pred_conf, conf_target)

            # box loss
            if matched_mask.any():
                pos_pred_boxes = pred_boxes[matched_mask]
                pos_target_boxes = target_boxes[best_gt[matched_mask]]
                box_loss = complete_box_iou_loss(pos_pred_boxes, pos_target_boxes).mean()
                total_loss += self.lambda_conf * conf_loss + self.lambda_box * box_loss
            else:
                box_loss = torch.tensor(0.0, device=pred.device)
                total_loss += self.lambda_conf * conf_loss

            # ===== DEBUGGING BLOCK =====
            if torch.rand(1).item() < 0.002:
                print(f"[DEBUG] batch {batch_idx}")
                print(f"  conf_target > 0: {(conf_target > 0).sum().item()} / {len(conf_target)}")
                print(f"  pred_conf min/max: {pred_conf.min().item():.3f} / {pred_conf.max().item():.3f}")
                print(f"  conf_loss: {conf_loss.item():.4f}")
                if matched_mask.any():
                    print(f"  box_loss: {box_loss.item():.4f}")
                print("-" * 50)
            # ===========================

        return total_loss / len(preds)

def box_cxcywh_to_xyxy(boxes):
    x_c, y_c, w, h = boxes.unbind(dim=1)
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)
