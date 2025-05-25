# train.py
# This script trains a YOLOv5 model using the pedestrian datasets 

import sys
sys.path.append('./yolov5')  

from yolov5.train import run  # Import the training function from YOLOv5

if __name__ == '__main__':
    # Run training with custom settings
    run(
        imgsz=640,  # image size (640x640)
        batch=16,   # batch size
        epochs=50,  # number of training epochs
        data='data/data.yaml',  # path to dataset config file
        weights='yolov5s.pt',   # use pretrained YOLOv5 small model
        name='pedestrian_yolov5',  # name of the training run
        project='runs/train',      # where to save the results
        exist_ok=True  # don't crash if the folder already exists
    )
