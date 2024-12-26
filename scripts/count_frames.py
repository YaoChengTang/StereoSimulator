import cv2
import os
from tqdm import tqdm
def parser_video(path, readFps = 1):
    # return frame_list, fps, frame_cnt
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Failed to open video")
    else:
        NotImplemented
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if __name__ == "__main__":
    folder_path = "/data2/videos/youtube/video"
    sum = 0
    sumc = 0
    for i in range(11):
        floder = folder_path + str(i)
        List = os.listdir(floder)
        sum = 0
        for j in tqdm(List):
            sum += parser_video(os.path.join(floder, j))
        print(f"folder{i} frames are {sum}")
        sumc += sum
    print(sumc)
