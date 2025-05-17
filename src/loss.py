import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        total_loss = 0
        bce = torch.nn.BCEWithLogitsLoss()
        for pred, target in zip(preds, targets):
            target = target.to(pred.device)
            pred_boxes = pred[:, :4]
            pred_conf = pred[:, 4]

            conf_target = torch.zeros_like(pred_conf)
            
            for gt in target:
                ious = torch.tensor([compute_iou(pb, gt) for pb in pred_boxes], device=pred.device)
                best_match = torch.argmax(ious)
                if ious[best_match] > 0.5:
                    conf_target[best_match] = 1.0

            conf_loss = bce(pred_conf, conf_target)

            n = int(conf_target.sum().item())
            if n == 0:
                continue

            selected_preds = pred_boxes[conf_target == 1]
            selected_target = target[:n]

            bbox_loss = torch.nn.functional.mse_loss(selected_preds, selected_target)
            total_loss += bbox_loss + conf_loss
        return total_loss / len(preds)
def compute_iou(box1, box2):
    # box: [x, y, w, h] 형식, 정규화된 0~1 값
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # (x1, y1) ~ (x2, y2) 형태로 좌표 변환
    box1 = [x1, y1, x1 + w1, y1 + h1]
    box2 = [x2, y2, x2 + w2, y2 + h2]

    xa = max(box1[0], box2[0])
    ya = max(box1[1], box2[1])
    xb = min(box1[2], box2[2])
    yb = min(box1[3], box2[3])

    inter_area = max(0, xb - xa) * max(0, yb - ya)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0
