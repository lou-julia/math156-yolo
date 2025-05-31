import cv2, os
from glob import glob
from tqdm import tqdm

def extract_frames_from_seq(seq_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(seq_file)
    count = 0
    base_name = os.path.splitext(os.path.basename(seq_file))[0]
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        filename = os.path.join(output_dir, f"{base_name}_{count:05d}.jpg")
        cv2.imwrite(filename, frame)
        count += 1
    cap.release()

base_path = "/Users/julialou/Desktop/project/data_and_labels"

train_seq_files = glob(os.path.join(base_path, "Train", "set*", "set*", "V*.seq"))
for seq_file in tqdm(train_seq_files, desc="Extracting Train"):
    extract_frames_from_seq(seq_file, os.path.join(base_path, "images/Train"))
    
test_seq_files = glob(os.path.join(base_path, "Test", "set*", "set*", "V*.seq"))
for seq_file in tqdm(test_seq_files, desc="Extracting Test"):
    extract_frames_from_seq(seq_file, os.path.join(base_path, "images/Test"))
