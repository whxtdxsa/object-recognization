# utils/postprocess.py
import torch
import numpy as np

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, max_det=300):
    """
    YOLOv5 스타일 NMS
    prediction: (batch, num_detections, 85)
    return: list[tensor] of detections per image: (num_dets, 6) → [x1, y1, x2, y2, conf, cls]
    """
    # Use torchvision NMS for simplicity
    from torchvision.ops import nms

    result = []
    for i in range(prediction.shape[0]):
        pred = prediction[i]

        # Compute confidence = obj_conf * class_conf
        scores = pred[:, 4:5] * pred[:, 5:]
        conf, cls = scores.max(1)
        mask = conf > conf_thres

        boxes = pred[mask, :4]
        boxes = xywh2xyxy(boxes)
        conf = conf[mask]
        cls = cls[mask].float()

        if boxes.shape[0] == 0:
            result.append(torch.zeros((0, 6)))
            continue

        keep = nms(boxes, conf, iou_thres)[:max_det]
        result.append(torch.cat((boxes[keep], conf[keep].unsqueeze(1), cls[keep].unsqueeze(1)), dim=1))

    return result


def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = x.clone()
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return y
