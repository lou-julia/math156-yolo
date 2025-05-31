# Machine Learning and Convulational Neural Networks using CalTech Pedestrians and YOLO v5 Model 🚗 🚷🚶‍♀️

## How to use our repository:
Our code runs on the CalTech Pedestrians Dataset, which is accessible at https://data.caltech.edu/records/f6rph-90m20. The files to download are data_and_labels.zip and CaltechPedestrians.zip.

### Preprocessing steps:
The data is provided in a pre-split format, with separate .seq files for training and testing, along with corresponding annotation files in .vbb format. For model training, we converted the .vbb files into YOLO-compatible .txt label files with "Parse vbb to txt" and extracted the video frames from the .seq files into .jpg images using "extract_frames_from_seq edited".

### Stuff you dont need to worry about:
YOLOv5 (All Caps): Source code from the yolov5 repo.
