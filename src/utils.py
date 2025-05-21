import random
import numpy as np
import torch
import os
import csv

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def init_csv_log(path, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

def log_to_csv(path, data_dict):
    with open(path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data_dict.keys())
        writer.writerow(data_dict)

from contextlib import nullcontext

def get_amp_components(device):
    """
    device (torch.device): 'cuda' or 'cpu'
    return:
        amp_context: autocast context or nullcontext
        scaler: GradScaler or None
    """
    if device.type == 'cuda':
        from torch.amp import autocast
        from torch.cuda.amp import GradScaler
        amp_context = autocast(device_type='cuda')
        scaler = GradScaler()
    else:
        amp_context = nullcontext()
        scaler = None
    return amp_context, scaler



from PIL import ImageDraw
import torchvision.transforms.functional as F
import os

def draw_bboxes(image_tensor, pred_tensor, conf_threshold=0.1, save_path="output.jpg"):
    pred_tensor[:,:4] = box_cxcywh_to_xyxy(pred_tensor[:,:4])
    pred_tensor[:,4] = torch.sigmoid(pred_tensor[:,4])
    img = F.to_pil_image(image_tensor.cpu())
    draw = ImageDraw.Draw(img)
    for box in pred_tensor:
        cx, cy, w, h, conf = box.tolist()
        if conf < conf_threshold:
            continue
        center_x_pixel = cx * img.width
        center_y_pixel = cy * img.height
        width_pixel = w * img.width
        height_pixel = h * img.height

        x1 = center_x_pixel - (width_pixel / 2)
        y1 = center_y_pixel - (height_pixel / 2)
        x2 = center_x_pixel + (width_pixel / 2)
        y2 = center_y_pixel + (height_pixel / 2)

        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        draw.text((x1, y1 - 10), f"{conf:.2f}", fill="red")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    img.save(save_path)
    print(f"Saved: {save_path}")

def box_cxcywh_to_xyxy(boxes):
    x_c, y_c, w, h = boxes.unbind(dim=1)
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)
