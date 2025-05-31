# train.py — Quick Training with YOLOv5 Official Loss

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from custom_yolo_dataset import YOLODataset
from yolov5.models.yolo import Model
from yolov5.utils.loss import ComputeLoss

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLOv5 small model with 1 class (pedestrian)
model = Model(cfg='yolov5/models/yolov5s.yaml', ch=3, nc=1).to(device)

# Training config
img_size = 416
epochs = 1
batch_size = 4

# Load training dataset (limit to 20 samples inside custom_yolo_dataset.py)
train_dataset = YOLODataset(
    images_dir='processed_data/images/train',
    labels_dir='processed_data/labels/train',
    img_size=img_size
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Use official YOLOv5 loss
compute_loss = ComputeLoss(model)

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for imgs, targets in train_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        # Forward pass
        preds = model(imgs)
        loss, _ = compute_loss(preds, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"✅ Epoch [{epoch+1}/{epochs}] complete. Loss: {total_loss:.4f}")

# Save weights
torch.save(model.state_dict(), 'best.pt')
print("🚀 Training finished. Model saved as best.pt")
