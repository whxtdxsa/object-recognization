import torch
# Env
from src.utils import set_seed
set_seed(42)



from src.loader import get_custom_dataloaders
_, test_loader = get_custom_dataloaders(batch_size=10) 

from src.utils import draw_bboxes
for i, (images, targets) in enumerate(test_loader):
    targets = torch.ones((targets.shape[0], 1), dtype=torch.float32)
    draw_bboxes(images[0], targets[0], conf_threshold=0.6, save_path=f"outputs/pred_{i}.png")
    if i == 9:
        break  



