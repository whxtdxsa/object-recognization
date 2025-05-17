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
amp_context, scaler = get_amp_components(device)



# --------------------------
# Experiment EDA
# --------------------------
from src.eda import *
import json
with open(label_path + train_file, 'r') as f:
    annotations_t = json.load(f)

img_sizes = get_img_sizes(annotations_t)
print(len(img_sizes))

with open(label_path + val_file, 'r') as f:
    annotations = json.load(f)

extract_annotations(annotations)

# category_counts = get_category_counts(annotations)
# plot_category_nums(category_counts)
img_sizes = get_img_sizes(annotations)
plot_img_sizes(img_sizes)
bbox_areas = get_bbox_areas(annotations)
plot_bbox_areas(bbox_areas)
bbox_counts = get_bbox_counts(annotations)
plot_bbox_counts(bbox_counts)
print(f"The number of data: {len(img_sizes)}\n")

image_id_to_info, image_id_to_bboxes = get_image_id_dict(annotations)
id = list(image_id_to_info.keys())[0]
show_bbox(img_val_path + image_id_to_info[id]['file_name'], image_id_to_bboxes[id])



# --------------------------
# Define Params
# --------------------------
# HyperParams
batch_size = 128
epochs = 5
lr = 0.1

# Model, Criterion, Optimizer
from src.model import SimpleDetector
from src.loss import DetectionLoss
network = SimpleDetector()
criterion = DetectionLoss()
optimizer = optim.Adam(network.parameters(), lr)



# --------------------------
# Define Train Procedure
# --------------------------
from src.loader import get_custom_dataloaders
train_loader, test_loader = get_custom_dataloaders(batch_size=batch_size) 
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

torch.save(network.state_dict(), 'weights/detector.pt') 

test_acc = evaluate_accuracy(network, test_loader, device, amp_context)
print(f"Test Acc: {test_acc:.4f}")

# Logging
from src.misc import init_csv_log, log_to_csv
experiment_name = f"{class_name}_bs{batch_size}_ep{epochs}_lr{lr}"
log_path = f"experiments/{experiment_name}/metrics.csv"

init_csv_log(log_path, ["epoch", "train_loss", "test_loss"])
for epoch in range(len(train_losses)):
    log_to_csv(log_path, {
        "epoch": epoch,
        "train_loss": train_losses[epoch],
        "test_loss": test_losses[epoch]
    })


