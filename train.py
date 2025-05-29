# train.py
# Custom training script for YOLOv5 pedestrian detection project

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from custom_yolo_dataset import YOLODataset
from yolov5.models.yolo import Model

# Device setup 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the YOLOv5 small model architecture (1 class = person)
model = Model(cfg='models/yolov5s.yaml', ch=3, nc=1).to(device)

# Initialize weights for better training
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(init_weights)

# Set image resolution to higher value for small object detection (e.g. pedestrians)
img_size = 832

# Load training data using our custom dataset class
train_dataset = YOLODataset(
    images_dir='your_dataset/images/train',
    labels_dir='your_dataset/labels/train',
    img_size=img_size
)

# Wrap the dataset in a DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Set up optimizer and learning rate scheduler (with decay)
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)  # LR halves every 3 epochs

# Simple placeholder loss 
def yolo_loss(pred, targets):
    return pred[0].sum() * 0.0  # dummy loss, for structure only

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for imgs, targets in train_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        preds = model(imgs)
        loss = yolo_loss(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    scheduler.step()  # apply learning rate decay
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Save the trained model weights
torch.save(model.state_dict(), 'best.pt')
print("Training complete. Weights saved as best.pt")
