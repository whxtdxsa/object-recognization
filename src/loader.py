import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets

def get_custom_dataloaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = CustomDataset('./data/labels/annotations/instances_train_person_only.json', './data/images/train2017', transform=transform)
    test_dataset = CustomDataset('./data/labels/annotations/instances_val_person_only.json', './data/images/val2017', transform=transform)

    train_dataset = Subset(train_dataset, range(60000))
    test_dataset = Subset(test_dataset, range(1000))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, collate_fn=custom_collate_fn)

    return train_loader, test_loader


from torch.utils.data import Dataset
from PIL import Image
import json
import os

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

        img, scale, pad_x, pad_y = letterbox_image(img, target_size=(640, 640))

        anns = self.image_to_anns.get(image_id, [])
        bboxes = []
        for ann in anns:
            x, y, bw, bh = ann['bbox']
            w_letter = bw * scale
            h_letter = bh * scale

            norm_x = ((x * scale + pad_x) + (w_letter / 2.0)) / 640
            norm_y = ((y * scale + pad_y) + (h_letter / 2.0)) / 640
            norm_w = w_letter / 640
            norm_h = h_letter / 640
            bboxes.append([norm_x, norm_y, norm_w, norm_h])

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(bboxes, dtype=torch.float32)

def letterbox_image(image, target_size=(640, 640), fill_color=(128, 128, 128)):
    iw, ih = image.size
    w, h = target_size
    scale = min(w / iw, h / ih)
    nw, nh = int(iw * scale), int(ih * scale)

    image_resized = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w, h), fill_color)
    pad_x = (w - nw) // 2
    pad_y = (h - nh) // 2
    new_image.paste(image_resized, (pad_x, pad_y))
    return new_image, scale, pad_x, pad_y
