# custom_yolo_dataset.py
# Simple dataset loader for YOLO-style images and labels

import os
import torch
from torch.utils.data import Dataset
import cv2

class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=832):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")

        img = cv2.resize(img, (self.img_size, self.img_size))

        # Convert BGR to RGB, transpose to (C, H, W), and ensure it's contiguous
        img = img[:, :, ::-1].transpose(2, 0, 1).copy()

        # Convert to tensor
        img = torch.from_numpy(img).float() / 255.0

        # Load label
        label_name = img_name.replace('.jpg', '.txt')
        label_path = os.path.join(self.labels_dir, label_name)
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    vals = list(map(float, line.strip().split()))
                    labels.append(vals)

        labels = torch.tensor(labels) if labels else torch.zeros((0, 5))

        return img, labels

        return img, labels