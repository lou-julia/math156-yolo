# infer.py
# Custom inference script with annotation

import os
import cv2
import torch
import numpy as np
from yolov5.models.yolo import Model
from yolov5.utils.general import non_max_suppression  # removed scale_coords
from yolov5.utils.datasets import letterbox
from yolov5.utils.plots import plot_boxes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLOv5 model architecture and trained weights
model = Model(cfg='models/yolov5s.yaml', ch=3, nc=1).to(device)
model.load_state_dict(torch.load('best.pt', map_location=device))
model.eval()

# Input and output paths
image_dir = 'your_dataset/images/val'
gt_label_dir = 'your_dataset/labels/val'
output_img_dir = 'inference_output/images'
output_txt_dir = 'inference_output/labels'
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_txt_dir, exist_ok=True)

# Inference settings
img_size = 640
conf_thres = 0.25
iou_thres = 0.3

# Initialize counters
tp = 0
fp = 0
fn = 0

# Helper functions
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / (box1_area + box2_area - inter_area)

def load_ground_truth(txt_path, img_shape):
    h, w = img_shape
    boxes = []
    if not os.path.exists(txt_path):
        return boxes
    with open(txt_path, 'r') as f:
        for line in f:
            cls, x, y, bw, bh = map(float, line.strip().split())
            x1 = (x - bw / 2) * w
            y1 = (y - bh / 2) * h
            x2 = (x + bw / 2) * w
            y2 = (y + bh / 2) * h
            boxes.append([x1, y1, x2, y2])
    return boxes

# Define missing scale_coords function
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """Rescale coords (xyxy) from img1_shape to img0_shape."""
    if ratio_pad is None:
        # Calculate from shapes
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2,
               (img1_shape[0] - img0_shape[0] * gain) / 2)  # width, height
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    return coords

# Inference loop
for filename in os.listdir(image_dir):
    if not filename.endswith('.jpg'):
        continue

    img_path = os.path.join(image_dir, filename)
    img0 = cv2.imread(img_path)
    h0, w0 = img0.shape[:2]
    img = letterbox(img0, new_shape=img_size)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = torch.from_numpy(img).float() / 255.0
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img)
        pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)

    label_lines = []
    matched = set()
    base_name = os.path.splitext(filename)[0]
    gt_boxes = load_ground_truth(os.path.join(gt_label_dir, base_name + ".txt"), (h0, w0))

    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                x1, y1, x2, y2 = [coord.item() for coord in xyxy]
                plot_boxes(img0, [[x1, y1, x2, y2]], labels=[f"person {conf:.2f}"], colors=[(0, 255, 0)], line_thickness=2)

                # YOLO-format normalized box
                x_center = ((x1 + x2) / 2) / w0
                y_center = ((y1 + y2) / 2) / h0
                bw = (x2 - x1) / w0
                bh = (y2 - y1) / h0
                label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")

                # Compare with ground truth for precision
                found_match = False
                for i, gt in enumerate(gt_boxes):
                    iou = compute_iou([x1, y1, x2, y2], gt)
                    if iou > 0.5 and i not in matched:
                        tp += 1
                        matched.add(i)
                        found_match = True
                        break
                if not found_match:
                    fp += 1
        else:
            if len(gt_boxes) > 0:
                fn += len(gt_boxes)

    # Save annotated image
    cv2.imwrite(os.path.join(output_img_dir, filename), img0)

    # Save YOLO label txt
    with open(os.path.join(output_txt_dir, base_name + ".txt"), 'w') as f:
        f.write('\n'.join(label_lines))

# Compute and print precision/recall
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\n✅ Inference complete. Results saved to 'inference_output/'")
print(f"🔍 Precision: {precision:.4f}")
print(f"📦 Recall:    {recall:.4f}")
print(f"TP: {tp}, FP: {fp}, FN: {fn}")

