from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from src.utils import custom_collate_fn

def get_custom_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])

    train_dataset = CustomDataset('./data/labels/annotations/instances_train_person_only.json', './data/images/train2017', transform=transform)
    test_dataset = CustomDataset('./data/labels/annotations/instances_val_person_only.json', './data/images/val2017', transform=transform)
    train_dataset = Subset(train_dataset, range(20000))
    test_dataset = Subset(test_dataset, range(1000))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=custom_collate_fn)

    return train_loader, test_loader


import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json

class CustomDataset(Dataset):
    def __init__(self, annotation_path, image_dir, transform=None):
        with open(annotation_path, 'r') as f:
            self.coco = json.load(f)

        self.image_dir = image_dir
        self.transform = transform

        self.image_id_to_filename = {img['id']: img['file_name'] for img in self.coco['images']}
        self.annotations = self.coco['annotations']

        # 이미지별 annotation 모으기
        self.image_to_anns = {}
        for ann in self.annotations:
            self.image_to_anns.setdefault(ann['image_id'], []).append(ann)

        self.image_ids = list(self.image_to_anns.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, self.image_id_to_filename[image_id])
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        # 바운딩 박스 처리 (정규화된 x, y, w, h)
        anns = self.image_to_anns.get(image_id, [])
        bboxes = []
        for ann in anns:
            x, y, bw, bh = ann['bbox']
            bbox = [x / w, y / h, bw / w, bh / h]
            bboxes.append(bbox)

        if self.transform:
            img = self.transform(img)

        if len(bboxes) > 0:
            target = torch.tensor(bboxes, dtype=torch.float32)  # [N, 4]
        else:
            target = torch.zeros((0, 4), dtype=torch.float32)

        return img, target

