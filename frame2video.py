import cv2
from utils.video_frame import gen_video
import os

List = os.listdir('/data4/lzd/iccv25/imgR_noFill1')
List.sort(key= lambda x : int(x.split('.')[0]))
frame_list = []
for i in List:
    frame = cv2.imread(f'/data4/lzd/iccv25/imgR_noFill1/{i}')
    frame_list.append(frame)


gen_video(frame_list,'./haha1.mp4' ,30)
