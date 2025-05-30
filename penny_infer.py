# infer.py — Run inference with trained YOLOv5 model

import os
import cv2
import torch
import numpy as np
from yolov5.models.yolo import Model
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.datasets import letterbox
from yolov5.utils.plots import plot_boxes

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = Model(cfg='yolov5/models/yolov5s.yaml', ch=3, nc=1).to(device)
model.load_state_dict(torch.load('best.pt', map_location=device))
model.eval()

# Path config
image_dir = 'processed_data/images/val'
output_img_dir = 'inference_output/images'
output_txt_dir = 'inference_output/labels'
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_txt_dir, exist_ok=True)

# Inference settings
img_size = 416
conf_thres = 0.25
iou_thres = 0.45

# Loop through images
for filename in os.listdir(image_dir):
    if not filename.endswith('.jpg'):
        continue

    img_path = os.path.join(image_dir, filename)
    img0 = cv2.imread(img_path)
    h0, w0 = img0.shape[:2]

    # Preprocess image
    img = letterbox(img0, new_shape=img_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = torch.from_numpy(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        pred = model(img)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

    # Postprocess + Save results
    base = os.path.splitext(filename)[0]
    label_lines = []

    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = [coord.item() for coord in xyxy]

                # Draw box
                plot_boxes(img0, [[x1, y1, x2, y2]], labels=[f"person {conf:.2f}"], colors=[(0, 255, 0)])

                # Normalize YOLO format for saving
                x_center = ((x1 + x2) / 2) / w0
                y_center = ((y1 + y2) / 2) / h0
                bw = (x2 - x1) / w0
                bh = (y2 - y1) / h0
                label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

    # Save annotated image
    cv2.imwrite(os.path.join(output_img_dir, filename), img0)

    # Save label file
    with open(os.path.join(output_txt_dir, base + '.txt'), 'w') as f:
        f.write('\n'.join(label_lines))

print("\n✅ Inference complete. Results saved to inference_output/")
