# This is our implementation of YOLOv1 since this is a simple and succinct model to show. We switched to YOLOv5 for a better training pipeline and better accuracy. 
# The comments are meant to give an idea of how YOLOv5 builds on v1.
# A full YOLOv5 implementaion is not practical to show here, you can refer to the ultralytics repo:
# https://github.com/ultralytics/yolov5/tree/master for the full implementation of YOLOv5.
# We made some modifications so the model would work better for the pedestrain dataset.
# These are mainly in the optimizer, the training loop, and changing the default hyperparameters. See report for details.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class YOLOv1(nn.Module):
    def __init__(self, S=13, B=1, C=1):
        super(YOLOv1, self).__init__()
        self.S = S  #Grid size - YOLOv5 does not use a fixed grid size, it uses "feature pyramids" 
        self.B = B  #boxes - YOLOv5 uses predefined anchor boxes, which 
        self.C = C  #classes

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 208x208

            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 104x104

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 52x52

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 26x26

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 13x13
        )
# YOLOv5 uses a special backbone called CSPDarknet53. It has many more layers and is purposefully designed to extract useful features.
# It also has a neck that uses PANet. It makes sure to account for broad and fine details to detect small and alrge objects.
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * self.S * self.S, 4096), nn.ReLU(),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C))
        )
#YOLOv5 has no fully connected layers. Any layer only sees some of the input.
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        x = x.view(-1, self.S, self.S, self.B * 5 + self.C)
        return x
# In YOLOv5 the model generates multiple outputs per forward pass. It includes predictions for several of te predetermined anchor boxes.

def yolo_loss(pred, target):
    lambda_coord = 5
    lambda_noobj = 0.5

    obj_mask = target[..., 4] == 1
    noobj_mask = target[..., 4] == 0

    mse = nn.MSELoss(reduction='sum')

    loss_coord = lambda_coord * mse(pred[obj_mask][..., :4], target[obj_mask][..., :4])
    loss_obj = mse(pred[obj_mask][..., 4], target[obj_mask][..., 4])
    loss_noobj = lambda_noobj * mse(pred[noobj_mask][..., 4], target[noobj_mask][..., 4])
    loss_class = mse(pred[obj_mask][..., 5:], target[obj_mask][..., 5:])

    total_loss = loss_coord + loss_obj + loss_noobj + loss_class
    return total_loss
# This amounts to just a simple mean squared error loss function. YOLOv5 uses a much more complicated loss function.
# It has special loss functions that are combined. 
# These account for error on the bounding box, the model's prediction of an object, and its classification of that object.

def train(model, dataloader, optimizer, epochs=5):
    model.train() 
    for epoch in range(epochs):
        total_loss = 0
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()            
            outputs = model(images)            
            loss = yolo_loss(outputs, targets) 
            loss.backward()                    
            optimizer.step()                  
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
# YOLOv5 trains with more nuance. It does data augmentation to accentuate particular features,
# it uses custom learning rate schedules, we can make a choice of optimizer, and more.
def detect(model, image_tensor, conf_thresh=0.5):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0).to(device))
        output = output.squeeze(0).cpu() 
        detections = []
        S = 13
        for i in range(S):
            for j in range(S):
                cell = output[i, j]
                x, y, w, h, conf, cls_score = cell
                if conf > conf_thresh:
                    box_x = (j + x.item()) / S * 416
                    box_y = (i + y.item()) / S * 416
                    box_w = w.item() * 416
                    box_h = h.item() * 416
                    detections.append([box_x, box_y, box_w, box_h, conf.item()])
        return detections
# YOLOv5 combines its 3 scales to make an inference.
# It also uses non-max suppression to drop overlapping boxes with less evidence in their favor.
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOv1().to(device)
    dataset = putdatasethere(num_samples=100)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, dataloader, optimizer, epochs=10)
    test_img, _ = dataset[0]
    boxes = detect(model, test_img, conf_thresh=0.3)
    print("Detections:", boxes)