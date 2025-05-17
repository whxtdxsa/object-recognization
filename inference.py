import torch
import torchvision
import torch.optim as optim

# --------------------------
# Set Env
# --------------------------
# Path
img_val_path = "./data/images/val2017/"
label_path = "./data/labels/annotations/"
train_file = "instances_train2017.json"
val_file = "instances_val2017.json"

# Preprocess Data
from src.preprocess import extract_person_data
extract_person_data(label_path, train_file)
extract_person_data(label_path, val_file)

train_file = "instances_train_person_only.json"
val_file = "instances_val_person_only.json"

# Env
from src.utils import set_seed, get_amp_components
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
train_loader, test_loader = get_custom_dataloaders(batch_size=10) 

network.load_state_dict(torch.load("weights/detector.pt", map_location="cpu"))
network.to(device)
network.eval()

# Training
from src.utils import draw_bboxes
with torch.no_grad():
    for i, (images, _) in enumerate(test_loader):
        images = images.to(device)
        preds = network(images)  # [1, 16, 5]
        draw_bboxes(images[0], preds[0], conf_threshold=0.5, save_path=f"outputs/pred_{i}.png")
        if i == 4:
            break  



