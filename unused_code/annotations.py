import os
import cv2
import numpy as np
from scipy.io import loadmat
from glob import glob
from tqdm import tqdm

def parse_vbb(vbb_path):
    mat = loadmat(vbb_path)
    A = mat['A'][0][0]
    obj_lists = A[1][0]
    obj_labels = A[4][0]
    label_map = [str(l[0]) for l in obj_labels]
    frames = []

    for objs in obj_lists:
        bboxes = []
        if objs.size == 0:
            frames.append(bboxes)
            continue
        for obj in objs[0]:
            label = label_map[obj[0][0][0] - 1]
            if label != 'person':
                continue
            pos = obj[1][0]
            occluded = obj[3][0][0]
            if occluded:
                continue
            bboxes.append(pos)
        frames.append(bboxes)
    return frames

def convert_to_yolo(box, img_w, img_h):
    x, y, w, h = box
    x_center = (x + w/2) / img_w
    y_center = (y + h/2) / img_h
    return f"0 {x_center:.6f} {y_center:.6f} {w/img_w:.6f} {h/img_h:.6f}"

def write_labels(vbb_path, image_dir, label_dir):
    os.makedirs(label_dir, exist_ok=True)
    frame_boxes = parse_vbb(vbb_path)
    vbb_name = os.path.splitext(os.path.basename(vbb_path))[0]
    for i, boxes in enumerate(frame_boxes):
        img_path = os.path.join(image_dir, f"{vbb_name}_{i:05d}.jpg")
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        label_file = os.path.join(label_dir, f"{vbb_name}_{i:05d}.txt")
        with open(label_file, 'w') as f:
            for box in boxes:
                f.write(convert_to_yolo(box, w, h) + '\n')

base_path = "/Users/julialou/Desktop/project/data_and_labels"

vbb_files = glob(os.path.join(base_path, "annotations", "annotations", "set*", "V*.vbb"))

train_image_dir = os.path.join(base_path, "images", "Train")
test_image_dir = os.path.join(base_path, "images", "Test")
train_label_dir = os.path.join(base_path, "labels", "Train")
test_label_dir = os.path.join(base_path, "labels", "Test")

for vbb in tqdm(vbb_files, desc="Processing Annotations"):
    set_name = os.path.basename(os.path.dirname(vbb))
    if set_name in ["set00", "set01", "set02", "set03", "set04", "set05"]:
        write_labels(vbb, train_image_dir, train_label_dir)
    else:
        write_labels(vbb, test_image_dir, test_label_dir)
