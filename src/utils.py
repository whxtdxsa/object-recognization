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


def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets

from PIL import ImageDraw
import torchvision.transforms.functional as F
import os

def draw_bboxes(image_tensor, pred_tensor, conf_threshold=0.5, save_path="output.jpg"):
    img = F.to_pil_image(image_tensor.cpu())
    draw = ImageDraw.Draw(img)

    for box in pred_tensor:
        x, y, w, h, conf = box.tolist()
        if conf < conf_threshold:
            continue
        x1 = x * img.width
        y1 = y * img.height
        x2 = x1 + w * img.width
        y2 = y1 + h * img.height
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    img.save(save_path)
    print(f"Saved: {save_path}")
