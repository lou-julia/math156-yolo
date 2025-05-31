# Machine Learning and Convulational Neural Networks using CalTech Pedestrians and YOLO v5 Model 🚗 🚷🚶‍♀️

## How to use this repository:
Our code runs on the CalTech Pedestrians Dataset, which is accessible at https://data.caltech.edu/records/f6rph-90m20. The files to download are data_and_labels.zip and CaltechPedestrians.zip. Sections below correspond to sections of the final report. 

### Dataset and Preprocessing
The data is provided in a pre-split format, with separate .seq files for training and testing, along with corresponding annotation files in .vbb format.
1. "parse_vbb_to_txt.py": Convert the .vbb files into YOLO-compatible .txt label files
2. "extract_frames_from_seq.py": Extract the video frames from the .seq files into .jpg images with

### Model and Methodology - Simplified YOLOv5
1. "YOLOv1_to_v5_overview.py" in the unused_code folder: Contains the code for the original YOLOv1 implementation that we used. However, we later moved on to a version of YOLOv5. An explanation of how we built upon YOLOv1 to create YOLOv5 is contained in this file.
2. "yolov5" folder: the CNN Architecture we used to perform object detection.
   
### Other code contained in the repository that was not mentioned:
- YOLOv5 (All Caps) folder: Source code from the yolov5 repository. 
- unused_code folder: Code that did not end up being used in the final report, meant to show thought processes and conceptual input as well as highlight intellectual contributions that are not immediately demonstrated. 

