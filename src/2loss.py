import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_iou, complete_box_iou_loss

class DetectionLoss(nn.Module):
    def __init__(self, lambda_box=1.0, lambda_obj=1.0, iou_threshold=0.2):
        super().__init__()
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.iou_threshold = iou_threshold
        self.bce = F.binary_cross_entropy_with_logits

    def focal_loss(self, logits, targets, alpha=0.25, gamma=2.0):
        prob = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        p_t = torch.clamp(p_t, min=1e-6, max=1 - 1e-6)
        loss = alpha_t * (1 - p_t) ** gamma * ce

        return loss.mean()

    def forward(self, preds, targets):
        total_loss = torch.tensor(0.0, device=preds[0].device)
        if len(preds) == 0: return total_loss 
        for pred, target in zip(preds, targets):
            target = target.to(pred.device)

                            
            pred_boxes = box_cxcywh_to_xyxy(pred[:, :4])
            pred_conf = pred[:, 4]

            conf_target = torch.zeros_like(pred_conf)
            matched_preds = []
            matched_targets = []

            for gt in target:
                gt_box = gt[:4]
                ious = box_iou(pred_boxes, gt_box.unsqueeze(0))[:, 0]
                best_match = torch.argmax(ious)
                if ious[best_match] > self.iou_threshold and conf_target[best_match] < 0.01:
                    # conf_target[best_match] = ious[best_match].clamp(min=0.3)
                    conf_target[best_match] = 1.0
                    matched_preds.append(pred_boxes[best_match])
                    matched_targets.append(gt_box)  # assuming gt has (x1, y1, x2, y2, class)
            
            conf_loss = self.focal_loss(pred_conf, conf_target)
            # conf_loss = self.bce(pred_conf, conf_target)
            # Bounding box regression loss (L1 or smooth L1)
            if matched_preds:
                pred_box_tensor = torch.stack(matched_preds)
                target_box_tensor = torch.stack(matched_targets).to(pred_box_tensor.device)
                box_loss = complete_box_iou_loss(pred_box_tensor, target_box_tensor).mean()
                total_loss += self.lambda_obj * conf_loss + self.lambda_box * box_loss
                if torch.rand(1).item() < 0.002:
                    print(f"[DEBUG] matched: {len(matched_preds)}")
                    print(f"[DEBUG] conf_loss: {conf_loss.item():.4f}, box_loss: {box_loss.item():.4f}, total: {total_loss.item():.4f}")
                    print(f"[DEBUG] pred_conf range: min={pred_conf.min().item():.3f}, max={pred_conf.max().item():.3f}")
            else:
                total_loss += self.lambda_obj * conf_loss

        return total_loss / len(preds)

def box_cxcywh_to_xyxy(boxes):
    x_c, y_c, w, h = boxes.unbind(dim=1)
    w = w.clamp(min=1e-4)
    h = h.clamp(min=1e-4)
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)
