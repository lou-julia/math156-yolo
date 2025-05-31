def extract_frames_from_seq(seq_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(seq_file)
    count = 0
    base_name = os.path.splitext(os.path.basename(seq_file))[0]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # *********** Filtering invalid/small frames ***********
        if frame is None or frame.shape[0] < 100 or frame.shape[1] < 100:
            count += 1
            continue

        # *********** Resize image to YOLO-compatible size ***********
        frame = cv2.resize(frame, (640, 640))

        filename = os.path.join(output_dir, f"{base_name}_{count:05d}.jpg")
        cv2.imwrite(filename, frame)
        count += 1

    cap.release()
