# src/model.py
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class SimpleDetector(nn.Module):
    def __init__(self):
        super().__init__()

        # ResNet18 백본 (마지막 풀링/FC 제거)
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # output: [B, 512, H/32, W/32]
        # Detection Head
        self.reduction = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),# 추가
        )
        self.head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.box_head = nn.Sequential(
            nn.Conv2d(128, 4, kernel_size=1)
        )
        self.conf_head = nn.Sequential(
            nn.Conv2d(132, 1, kernel_size=1)
        )
        self._initialize_biases()
    
    def _initialize_biases(self):
        nn.init.constant_(self.conf_head[-1].bias, 1.0)

    def forward(self, x):
        B = x.size(0)
        feats = self.backbone(x)
        feats = self.reduction(feats)
        feats = self.head(feats)  # [B, 5, H/32, W/32]

        box_out = self.box_head(feats)
        conf_feats = torch.cat([feats, box_out], dim=1) 
        conf_out = self.conf_head(conf_feats)

        B, _, H, W = feats.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=feats.device),
            torch.arange(W, device=feats.device),
            indexing='ij'
        )  # [H, W]

        # [1, H, W, 1]
        grid_x = grid_x.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1).float()
        grid_y = grid_y.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, -1).float()
        norm = torch.tensor([W, H], device=feats.device).reshape(1, 1, 1, 2)
        # 1. box_out: [B, 4, H, W] → [B, H, W, 4]
        box_out = box_out.permute(0, 2, 3, 1)

        # 2. conf_out: [B, 1, H, W] → [B, H, W, 1]
        conf_out = conf_out.permute(0, 2, 3, 1)

        # 3. pred_xy, pred_wh: [B, H, W, 2]
        pred_xy = (torch.sigmoid(box_out[..., :2]) + torch.cat([grid_x, grid_y], dim=-1)) / norm
        pred_wh = 2.0 * F.softplus(box_out[..., 2:4]) / norm

        # 4. pred_conf: [B, H, W, 1]
        pred_conf = torch.sigmoid(conf_out)

        # 5. cat + reshape: [B, H, W, 5] → [B, N, 5]
        pred = torch.cat([pred_xy, pred_wh, pred_conf], dim=-1)
        pred = pred.view(B, -1, 5)
        return pred
