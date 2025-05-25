# infer.py
# This script runs inference using the trained YOLOv5 model

import sys
sys.path.append('./yolov5')  

from yolov5.detect import run  # Import the detection function

if __name__ == '__main__':
    # Run detection on validation images
    run(
        weights='runs/train/pedestrian_yolov5/weights/best.pt',  # path to trained weights
        source='datasets/images/val',                            # folder of images to test on
        imgsz=640,                                               # image size
        conf_thres=0.25,                                         # confidence threshold
        save_txt=True,                                           # save results as text files
        save_conf=True,                                          # save confidence scores
        nosave=False                                             # save output images with bounding boxes
    )
