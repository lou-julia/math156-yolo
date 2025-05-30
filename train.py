# train.py
# Custom training script for YOLOv5 pedestrian detection using official ComputeLoss

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from custom_yolo_dataset import YOLODataset
from yolov5.models.yolo import Model
from yolov5.utils.loss import ComputeLoss

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLOv5 small model (1 class = pedestrian)
model = Model(cfg='models/yolov5s.yaml', ch=3, nc=1).to(device)

# Set image size
img_size = 640

# Load dataset
train_dataset = YOLODataset(
    images_dir='/Users/pengqi/Desktop/processed_data/images/train',
    labels_dir='/Users/pengqi/Desktop/processed_data/labels/train',
    img_size=img_size
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# Use SGD optimizer and LR scheduler
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.937)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# Initialize official loss function from YOLOv5
compute_loss = ComputeLoss(model)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for imgs, targets in train_loader:
        imgs = torch.stack(imgs).to(device)
        targets = torch.cat(targets).to(device)  # [image_idx, class, x, y, w, h]

        preds = model(imgs)
        loss, _ = compute_loss(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    print(f"✅ Epoch {epoch+1}/{epochs} complete | Loss: {total_loss:.4f}")

# Save trained weights
torch.save(model.state_dict(), 'best.pt')
print("✅ Training complete. Model saved as best.pt")
