import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops import box_iou, complete_box_iou_loss

class SimpleDetector(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])
        self.layer2 = resnet.layer2  # [B, 128, 80, 80]
        self.layer3 = resnet.layer3  # [B, 256, 40, 40]
        self.layer4 = resnet.layer4  # [B, 512, 20, 20]

        self.lat_conv2 = nn.Conv2d(128, 128, kernel_size=1)
        self.lat_conv3 = nn.Conv2d(256, 128, kernel_size=1)
        self.lat_conv4 = nn.Conv2d(512, 128, kernel_size=1)

        self.smooth2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.smooth4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.box_head_20 = nn.Conv2d(128, 4, kernel_size=1)
        self.box_head_40 = nn.Conv2d(128, 4, kernel_size=1)
        self.box_head_80 = nn.Conv2d(128, 4, kernel_size=1)

        self.conf_head_20 = nn.Conv2d(128, 1, kernel_size=1)
        self.conf_head_40 = nn.Conv2d(128, 1, kernel_size=1)
        self.conf_head_80 = nn.Conv2d(128, 1, kernel_size=1)

        self._init_bias()

    def _init_bias(self):
        pi = 0.01
        initial_bias = -torch.log(torch.tensor((1.0 - pi) / pi))
        for head_module in [self.conf_head_20, self.conf_head_40, self.conf_head_80]:
            if head_module is not None:
                nn.init.constant_(head_module.bias, initial_bias.to(head_module.bias.device))

    def forward(self, x):
        B = x.size(0)
        x = self.layer1(x)

        c2 = self.layer2(x)      # [B, 128, 80, 80]
        c3 = self.layer3(c2)     # [B, 256, 40, 40]
        c4 = self.layer4(c3)     # [B, 512, 20, 20]

        p5_fpn = self.lat_conv4(c4)
        p4_fpn = self.lat_conv3(c3) + F.interpolate(p5_fpn, scale_factor=2, mode='nearest')
        p3_fpn = self.lat_conv2(c2) + F.interpolate(p4_fpn, scale_factor=2, mode='nearest')

        p5_out = self.smooth4(p5_fpn)
        p4_out = self.smooth3(p4_fpn)
        p3_out = self.smooth2(p3_fpn)
        
        raw_box_20 = self.box_head_20(p5_out).permute(0, 2, 3, 1).reshape(B, -1, 4)
        raw_box_40 = self.box_head_40(p4_out).permute(0, 2, 3, 1).reshape(B, -1, 4)
        raw_box_80 = self.box_head_80(p3_out).permute(0, 2, 3, 1).reshape(B, -1, 4)

        raw_box_pred_batch = torch.cat([raw_box_20, raw_box_40, raw_box_80], dim=1)
        normalized_box_pred_batch = torch.sigmoid(raw_box_pred_batch)

        conf_20 = self.conf_head_20(p5_out).permute(0, 2, 3, 1).reshape(B, -1, 1)
        conf_40 = self.conf_head_40(p4_out).permute(0, 2, 3, 1).reshape(B, -1, 1)
        conf_80 = self.conf_head_80(p3_out).permute(0, 2, 3, 1).reshape(B, -1, 1)

        conf_logits_batch = torch.cat([conf_20, conf_40, conf_80], dim=1)
        pred = torch.cat([normalized_box_pred_batch, conf_logits_batch], dim=-1)  # [B, N, 5]

        return pred

