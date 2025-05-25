# Machine Learning and Convulational Neural Networks using CalTech Pedestrians and YOLO v5 Model

## Objective
The objective of our project is to analyze the mathematical foundations of CNNs and demonstrate their effectiveness in solving real-world object detection tasks, with a specific focus on identifying pedestrians for autonomous driving systems.

## Application
Detecting pedestrians is an important part of making self-driving cars safer. These vehicles need to quickly recognize when people are nearby and react in time. Our project simulates this real-world application by training YOLO to look at footage and draw boxes around people it sees. YOLO is already used in industry for things like security and traffic analysis, so this gives us the opportunity to explore how it works in a real scenario. This technology has diverse applications, even beyond autonomous vehicles. It could be used to detect pedestrian traffic flow in busy areas such as airports, malls, or busy intersections. These insights could be used to improve infrastructure and make these areas safer.


## Dataset
We chose from well-known pedestrian datasets that include video or image data with labeled pedestrians.
1) Caltech Pedestrian Dataset: large-scale dataset with over 350,000 bounding boxes from real urban driving footage
2) JAAD (Joint Attention in Autonomous Driving): dataset that shows pedestrian behavior such as walking, stopping, and looking.
3) City Person: datasets that vary in lighting, occlusion, and motion 


## Model
YOLO (You Only Look Once)
The model will be trained on a pedestrian-specific dataset, where the only class label is "person." During training, the model will learn to identify pedestrians in varying poses, lighting conditions, and environments. After training, it will be evaluated using standard object detection metrics such as precision, recall, mean Average Precision (mAP), and Intersection over Union (IoU).


## Project Steps
1. Downloading dataset
2. Setting up YOLO model in programming
3. Training model on datasets
4. Analyzing Results
5. Dataset and Preprocessing
6. Report Write-up 

## Project Distribution
All of us will download the dataset. Deeloc, Julia, and Samvit will set up the YOLO model in programming. Das, Melissa, and Qi will work on training model on datasets. All of us will analyze results, work on dataset and preprocessing, and writing up our final report. 

## References

Dollar, P., Wojek, C., Schiele, B., & Perona, P. (2009). Caltech Pedestrians [Data set]. IEEE Conference on Computer Vision and Pattern Recognition. https://doi.org/10.1109/CVPR.2009.5206631

Jiang, Z., Huang, S., & Li, M. (2024). A Pedestrian Detection Network Based on an Attention Mechanism and Pose Information. Applied Sciences, 14(18), 8214. https://doi.org/10.3390/app14188214

L. Chen et al., "Deep Neural Network Based Vehicle and Pedestrian Detection for Autonomous Driving: A Survey," in IEEE Transactions on Intelligent Transportation Systems, vol. 22, no. 6, pp. 3234-3246, June 2021, doi: 10.1109/TITS.2020.2993926.

Lan, W., Dang, J., Wang, Y., & Wang, S. (2018). Pedestrian detection based on YOLO network model. 2018 IEEE International Conference on Mechatronics and Automation (ICMA), 1547–1551. https://doi.org/10.1109/ICMA.2018.8484698

M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The Cityscapes Dataset for Semantic Urban Scene Understanding,” in Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

Rasouli, Amir, Kotseruba, Iuliia, Tsotsos, John K. (n.d.). Are they going to cross? A benchmark dataset and baseline for pedestrian crosswalk behavior. In IEEE Intelligent Vehicles Symposium (IV) (pp. 264–269). essay. Retrieved 2025, from https://data.nvision2.eecs.yorku.ca/JAAD$_$dataset/.
