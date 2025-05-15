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
        from torch.amp import GradScaler
        amp_context = autocast(device_type='cuda')
        scaler = GradScaler()
    else:
        amp_context = nullcontext()
        scaler = None
    return amp_context, scaler


