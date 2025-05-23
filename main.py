import json

import torch
import torchvision
import torch.optim as optim

from src.utils import make_dir_if_not_exists
from src.utils import set_seed, get_amp_components
# --------------------------
# Set Env
# --------------------------
# Path
train_img_dir = './data/images/train2017'
val_img_dir = './data/images/val2017'

label_dir = "./data/labels/annotations/"
train_file = "instances_train_person_only.json"
val_file = "instances_val_person_only.json"
train_ann_path = label_dir + train_file
val_ann_path = label_dir + val_file
    
# Preprocess Data
from src.preprocess import extract_person_data
extract_person_data(label_dir, "instances_train2017.json")
extract_person_data(label_dir, "instances_val2017.json")

# Env
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")
amp_context, scaler = get_amp_components(device)



# --------------------------
# Experiment EDA
# --------------------------
from src.eda import *
with open(train_ann_path, 'r') as f:
    train_annos = json.load(f)

extract_annotations(train_annos)

# category_counts = get_category_counts(annotations)
# plot_category_nums(category_counts)
img_sizes = get_img_sizes(train_annos)
plot_img_sizes(img_sizes)
bbox_areas = get_bbox_areas(train_annos)
plot_bbox_areas(bbox_areas)
bbox_counts = get_bbox_counts(train_annos)
plot_bbox_counts(bbox_counts)

print(f"The number of train data: {len(img_sizes)}\n")
with open(val_ann_path, 'r') as f:
    val_annos = json.load(f)
img_sizes = get_img_sizes(val_annos)
print(f"The number of val data: {len(img_sizes)}\n")

image_id_to_info, image_id_to_bboxes = get_image_id_dict(annotations)
id = list(image_id_to_info.keys())[0]
show_bbox(img_val_path + image_id_to_info[id]['file_name'], image_id_to_bboxes[id])



# --------------------------
# Define Params
# --------------------------
from src.model import SimpleDetector
from src.loss import DetectionLoss
from src.utils import set_backbone_requires_grad
# HyperParams
batch_size = 128
epochs = 10
lr = 0.0001
experiment_name = f"bs{batch_size}_ep{epochs}_lr{lr}"

network = SimpleDetector()
criterion = DetectionLoss()
optimizer = optim.AdamW(network.parameters(), lr=lr, weight_decay=1e-4)

start_ep = 82
weight_path = "bs128_ep10_lr0.0001"
make_dir_if_not_exists(f"weights/{experiment_name}")
if start_ep != 0:
    network.load_state_dict(torch.load(f"./weights/{weight_path}/e_{start_ep}.pt", map_location=device))

set_backbone_requires_grad(network, requires_grad=False)



# --------------------------
# Define Train Procedure
# --------------------------
from src.loader import get_custom_dataloaders
train_loader, test_loader = get_custom_dataloaders(train_ann_path, train_img_dir, val_ann_path, val_img_dir, batch_size=batch_size, input_size=(640, 640)) 
network.to(device)

train_losses = []
test_losses = []

# Training
from src.trainer import train_one_epoch, evaluate_loss, evaluate_accuracy
for epoch in range(1, epochs + 1):
    print(f"Epoch {epoch}/{epochs}")
    train_loss = train_one_epoch(network, train_loader, optimizer, criterion, device, amp_context, scaler)
    train_losses.append(train_loss)

    test_loss = evaluate_loss(network, test_loader, criterion, device, amp_context)
    test_losses.append(test_loss)

    print(f"Train_loss: {train_loss:.4f}, Test_loss: {test_loss:.4f}")
    torch.save(network.state_dict(), f'weights/{experiment_name}/e_{epoch + start_ep}.pt') 

# test_acc = evaluate_accuracy(network, test_loader, device, amp_context)
# print(f"Test Acc: {test_acc:.4f}")

# Logging
from src.utils import init_csv_log, log_to_csv
log_path = f"experiments/{experiment_name}/metrics_{epochs + start_ep}.csv"

init_csv_log(log_path, ["epoch", "train_loss", "test_loss"])
for epoch in range(len(train_losses)):
    log_to_csv(log_path, {
        "epoch": epoch + start_ep,
        "train_loss": train_losses[epoch],
        "test_loss": test_losses[epoch]
    })
