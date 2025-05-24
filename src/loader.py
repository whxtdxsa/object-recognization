import json
import os

import albumentations as A
import cv2
import numpy as np
from PIL import Image
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets


def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets

def get_custom_dataloaders(
    train_ann_path,
    train_img_dir,
    val_ann_path,
    val_img_dir,
    batch_size=128, input_size=(640, 640)
):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    IMG_WIDTH, IMG_HEIGHT = input_size
    letterbox_fill_color = (128, 128, 128)

    train_transform = A.Compose([
        # Letter Box
        A.LongestMaxSize(max_size=max(IMG_WIDTH, IMG_HEIGHT), p=1.0),
        A.PadIfNeeded(
            min_width=IMG_WIDTH,
            min_height=IMG_HEIGHT,
            border_mode=cv2.BORDER_CONSTANT,
            value=letterbox_fill_color,
            p=1.0
        ),

        # Augmentations
        A.RandomSizedBBoxSafeCrop(width=IMG_WIDTH, height=IMG_HEIGHT, erosion_rate=0.1, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=(-0.1, 0.15),
            rotate_limit=15,
            p=0.7,
            border_mode=cv2.BORDER_CONSTANT,
            value=letterbox_fill_color
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.2),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.OneOf([
            A.Blur(blur_limit=3, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),

        ], p=0.15),
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
        ToTensorV2()
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['category_ids'],
        min_visibility=0.1,
        min_area=100
    ))

    val_transform = A.Compose([
        # Letter Box
        A.LongestMaxSize(max_size=max(IMG_WIDTH, IMG_HEIGHT), p=1.0),
        A.PadIfNeeded(
            min_width=IMG_WIDTH,
            min_height=IMG_HEIGHT,
            border_mode=cv2.BORDER_CONSTANT,
            value=letterbox_fill_color,
            p=1.0
        ),
        A.Normalize(mean=imagenet_mean, std=imagenet_std),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['category_ids']))

    train_dataset = CustomDataset(train_ann_path, train_img_dir, transform=train_transform)
    test_dataset = CustomDataset(val_ann_path, val_img_dir, transform=val_transform)

    train_dataset = Subset(train_dataset, range(60000))
    test_dataset = Subset(test_dataset, range(1000))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=10, pin_memory=True, collate_fn=custom_collate_fn)

    return train_loader, test_loader


class CustomDataset(Dataset):
    def __init__(self, annotation_path, image_dir, transform=None):
        with open(annotation_path, 'r') as f:
            self.coco = json.load(f)

        self.image_dir = image_dir
        self.transform = transform

        self.image_id_to_filename = {img['id']: img['file_name'] for img in self.coco['images']}
        self.image_id_to_dims = {img['id']: (img['width'], img['height']) for img in self.coco['images']}

        self.annotations = self.coco['annotations']
        self.image_to_anns = {}
        for ann in self.annotations:
            self.image_to_anns.setdefault(ann['image_id'], []).append(ann)

        self.image_ids = list(self.image_to_anns.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, image_filename)
        
        img_pil = Image.open(image_path).convert("RGB")
        img_np = np.array(img_pil)
        
        original_width, original_height = self.image_id_to_dims[image_id]

        anns = self.image_to_anns.get(image_id, [])
        bboxes = []
        category_ids = []
        
        for ann in anns:
            x, y, w, h = ann['bbox']

            norm_x = (x + w / 2) / original_width
            norm_y = (y + h / 2) / original_height
            norm_w = w / original_width
            norm_h = h / original_height

            bboxes.append([norm_x, norm_y, norm_w, norm_h])
            category_ids.append(0)
        # src/loader.py의 CustomDataset 내 __getitem__ 메서드 수정
        problematic_bboxes_for_debug = []
        for i, bbox_coords in enumerate(initial_yolo_bboxes):
            # 각 좌표가 (0, 1] 범위를 만족하는지 확인
            if not all(0.0 < coord <= 1.0 for coord in bbox_coords):
                problematic_bboxes_for_debug.append({
                    "original_coco": anns[i]['bbox'], # 원본 COCO 박스
                    "calculated_yolo": bbox_coords,   # 계산된 YOLO 박스
                    "image_id": image_id,
                    "image_path": image_path,
                    "original_dims": (original_width, original_height)
                })

        if problematic_bboxes_for_debug:
            print(f"DEBUG: Found bboxes not in (0, 1] range for image_id {image_id}:")
            for info in problematic_bboxes_for_debug:
                print(f"  Image: {info['image_path']}, OrigDims: {info['original_dims']}")
                print(f"  Orig COCO: {info['original_coco']}, Calc YOLO: {info['calculated_yolo']}")
            # 여기서 에러를 발생시키거나, 특정 조건으로 필터링할 수 있습니다.
            # raise ValueError(f"Problematic YOLO bbox calculated for image_id {image_id}") # 디버깅을 위해 강제 에러

        try:
            transformed = self.transform(image=img_np, bboxes=initial_yolo_bboxes, category_ids=category_ids)
        except ValueError as e:
            if "In YOLO format all coordinates must be float and in range (0, 1]" in str(e):
                print(f"--- Albumentations ValueError Caught ---")
                print(f"Image ID: {image_id}, Path: {image_path}")
                print(f"Original Dims (W, H): ({original_width}, {original_height})")
                print("Initial YOLO BBoxes passed to Albumentations:")
                for i, bbox_item in enumerate(initial_yolo_bboxes):
                    print(f"  Idx {i}: {bbox_item} (Original COCO: {anns[i]['bbox'] if i < len(anns) else 'N/A'})")
                print("--- End of Debug Info ---")
            raise e # 원래 에러를 다시 발생시켜 학습 중단
            
        transformed = self.transform(image=img_np, bboxes=bboxes, category_ids=category_ids)
        img_transformed_tensor = transformed['image']
        transformed_bboxes = transformed['bboxes']
        if not transformed_bboxes:
            target = torch.empty((0, 4), dtype=torch.float32)
        else:
            target = torch.tensor(transformed_bboxes, dtype=torch.float32)


        return img_transformed_tensor, target
