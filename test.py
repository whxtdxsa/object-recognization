import torch
from src.utils import set_seed
set_seed(42)

from src.loader import get_custom_dataloaders
train_loader, test_loader = get_custom_dataloaders(batch_size=10) 

from src.utils import draw_bboxes
for i, (images, targets) in enumerate(train_loader):
    target = targets[0]
    conf = torch.ones((target.shape[0], 1), dtype=torch.float32)
    target_with_conf = torch.cat([target, conf], dim=1)
    draw_bboxes(images[0], target_with_conf, conf_threshold=0.6, save_path=f"outputs/pred-{i}.png")
    if i == 9:
        break  



