import cv2

def parser_video(path, stride = 1):
    # return frame_list, fps, frame_cnt
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Can not open the video")
        exit()
    frame_count = 0
    frame_list = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break  
        if frame_count % stride == 0:
            frame_list.append(frame)
        frame_count += 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame_list, fps, frame_count

def gen_video(frame_list, path, fps):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_height, frame_width = frame_list[0].shape[:2]
    print(f"Video resolution: {frame_width}x{frame_height}")
    
    videoWrite = cv2.VideoWriter(path, fourcc, fps, (frame_width, frame_height))
    if not videoWrite.isOpened():
        print("Error: VideoWriter failed to open!")
        return
    
    for i, frame in enumerate(frame_list):
        if frame is not None:
            videoWrite.write(frame)
        else:
            print(f"Skipping invalid frame at index {i}")
    
    videoWrite.release()
    print(f"Video saved to {path}")
