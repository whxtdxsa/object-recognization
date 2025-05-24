import os
import json

import torch
import torchvision
import torch.optim as optim

from src.utils import (
    set_seed, get_amp_components, init_csv_log, log_to_csv
)
from src.preprocess import extract_person_data
from src.eda import (
    extract_annotations, get_img_sizes, plot_img_sizes, get_bbox_areas, plot_bbox_areas, 
    get_bbox_counts, plot_bbox_counts, get_image_id_dict, show_bbox
)
from src.model import SimpleDetector
from src.loss import DetectionLoss
from src.utils import set_backbone_requires_grad
from src.loader import get_custom_dataloaders
from src.trainer import train_one_epoch, evaluate_loss, evaluate_accuracy

# --------------------------
# Configuration
# --------------------------
config = {
    "seed": 42,
    "run_eda": False,
    "data": {
        "base_dir": "./data",
        "train_img_subdir": "images/train2017",
        "val_img_subdir": "images/val2017",
        "label_subdir": "labels/annotations",
        "train_ann_file_original": "instances_train2017.json",
        "val_ann_file_original": "instances_val2017.json",
        "train_ann_file_person_only": "instances_train_person_only.json",
        "val_ann_file_person_only": "instances_val_person_only.json",
        "input_size": (640, 640)
    },
    "training": {
        "batch_size": 128,
        "epochs_to_run_this_session": 10,
        "initial_start_epoch_manual": 82,
        "lr": 0.0001,
        "weight_decay": 1e-4,
        "freeze_backbone": True
    },
    "experiment_name_template": "bs{bs}_lr{lr}",
    "weights_dir_base": "weights",
    "logs_dir_base": "experiments"
}
experiment_name = config["experiment_name_template"].format(bs=config["training"]["batch_size"], lr=config["training"]["lr"])

# Path
train_img_dir = os.path.join(config["data"]["base_dir"], config["data"]["train_img_subdir"])
val_img_dir = os.path.join(config["data"]["base_dir"], config["data"]["val_img_subdir"])
label_dir = os.path.join(config["data"]["base_dir"], config["data"]["label_subdir"])

train_ann_path_original = os.path.join(label_dir, config["data"]["train_ann_file_original"])
val_ann_path_original = os.path.join(label_dir, config["data"]["val_ann_file_original"])

train_ann_path_person_only = os.path.join(label_dir, config["data"]["train_ann_file_person_only"])
val_ann_path_person_only = os.path.join(label_dir, config["data"]["val_ann_file_person_only"])

weights_dir = os.path.join(config["weights_dir_base"], experiment_name)
log_dir = os.path.join(config["logs_dir_base"], experiment_name)   
log_path = os.path.join(log_dir, "metrics.csv")
log_fieldnames = ["epoch", "train_loss", "test_loss"]

os.makedirs(weights_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# --------------------------
# Env
# --------------------------
set_seed(config["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")
amp_context, scaler = get_amp_components(device)

# --------------------------
# Data Preprocessing
# --------------------------
if not os.path.exists(train_ann_path_person_only):
    extract_person_data(train_ann_path_original, train_ann_path_person_only)

if not os.path.exists(val_ann_path_person_only):
    extract_person_data(val_ann_path_original, val_ann_path_person_only)

# --------------------------
# Experiment EDA
# --------------------------
if config["run_eda"]:
    print("--- Running EDA ---")
    with open(train_ann_path_person_only, 'r') as f:
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
    with open(val_ann_path_person_only, 'r') as f:
        val_annos = json.load(f)
    img_sizes = get_img_sizes(val_annos)
    print(f"The number of val data: {len(img_sizes)}\n")

    image_id_to_info, image_id_to_bboxes = get_image_id_dict(train_annos)
    id = list(image_id_to_info.keys())[0]
    show_bbox(img_val_path + image_id_to_info[id]['file_name'], image_id_to_bboxes[id])
    print("--- EDA Finished ---")


# --------------------------
# Model, Criterion, Optimizer
# --------------------------
network = SimpleDetector().to(device)
criterion = DetectionLoss()
optimizer = optim.AdamW(network.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])

current_start_epoch = config["training"]["initial_start_epoch_manual"]
if current_start_epoch != 0:
    weight_to_load = os.path.join(weights_dir, f"e_{current_start_epoch}.pt")
    network.load_state_dict(torch.load(weight_to_load, map_location=device))

if config["training"]["freeze_backbone"]:
    set_backbone_requires_grad(network, requires_grad=False)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, network.parameters()), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])
# --------------------------
# DataLoaders
# --------------------------
train_loader, test_loader = get_custom_dataloaders(
    train_ann_path_person_only, train_img_dir, val_ann_path_person_only, val_img_dir, 
    batch_size=config["training"]["batch_size"], input_size=config["data"]["input_size"]
) 

# --------------------------
# Training Loop 
# --------------------------
init_csv_log(log_path, log_fieldnames)

for i in range(config["training"]["epochs_to_run_this_session"]):
    epoch = current_start_epoch + i + 1
    print(f"Epoch {epoch}/{current_start_epoch + 1 + config['training']['epochs_to_run_this_session']}")
    train_loss = train_one_epoch(network, train_loader, optimizer, criterion, device, amp_context, scaler)
    test_loss = evaluate_loss(network, test_loader, criterion, device, amp_context)

    log_to_csv(log_path, {
        "epoch": epoch,
        "train_loss": train_losses[epoch],
        "test_loss": test_losses[epoch]
    })
    print(f"Train_loss: {train_loss:.4f}, Test_loss: {test_loss:.4f}")
    torch.save(network.state_dict(), os.path.join(weights_dir, f"e_{epoch}.pt"))

# test_acc = evaluate_accuracy(network, test_loader, device, amp_context)
# print(f"Test Acc: {test_acc:.4f}")
