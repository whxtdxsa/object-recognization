# src/model.py
import torch
import torch.nn as nn
import torchvision.models as models

class SimpleDetector(nn.Module):
    def __init__(self, num_boxes=16):
        super().__init__()
        self.num_boxes = num_boxes

        # ResNet18 백본 (마지막 풀링/FC 제거)
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # output: [B, 512, H/32, W/32]

        # Detection Head
        self.head = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_boxes * 5, kernel_size=1)  # 5 = [x, y, w, h, conf]
        )

    def forward(self, x):
        feats = self.backbone(x)  # [B, 512, H/32, W/32]
        out = self.head(feats)    # [B, num_boxes * 5, H/32, W/32]
        B = out.size(0)

        # Global average pooling over spatial dims
        out = out.mean(dim=[2, 3])  # [B, num_boxes * 5]
        out = out.view(B, self.num_boxes, 5)  # [B, N, 5] → (x, y, w, h, conf)

        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        total_loss = 0
        for pred, target in zip(preds, targets):
            if target.shape[0] == 0:
                continue
            pred_boxes = pred[:, :4]
            pred_conf = pred[:, 4]

            conf_sorted_idx = torch.argsort(pred_conf, descending=True)[:target.shape[0]]
            selected_preds = pred_boxes[conf_sorted_idx]

            loss = F.mse_loss(selected_preds, target)
            total_loss += loss
        return total_loss / len(preds)
