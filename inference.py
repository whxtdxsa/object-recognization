import torch
# Env
from src.utils import set_seed
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Model, Criterion, Optimizer
from src.model import SimpleDetector
network = SimpleDetector()


# --------------------------
# Define Train Procedure
# --------------------------
from src.loader import get_custom_dataloaders
_, test_loader = get_custom_dataloaders(batch_size=10) 

network.to(device)
network.load_state_dict(torch.load("weights/bs128_ep10_lr0.0001/e_77.pt", map_location=device))
network.eval()

from src.utils import draw_bboxes, postprocess_single_image_predictions
with torch.no_grad():
    for i, (images, _) in enumerate(test_loader):
        images = images.to(device)
        preds = network(images)  # [1, 16, 5]
        pred = postprocess_single_image_predictions(preds[0])

        draw_bboxes(images[0], pred, conf_threshold=0.6, save_path=f"outputs/pred_{i}.png")
        if i == 9:
            break  



