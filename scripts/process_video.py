import os
import re
import sys
import cv2
import glob
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.video_frame import parser_video


def save_frame(frame, frames_dir, frame_idx):
    """Save a single frame to disk."""
    frame_filename = os.path.join(frames_dir, f"frame_{frame_idx:04d}.png")
    cv2.imwrite(frame_filename, frame)


def save_frames(frames_root, video_rel_path, frame_list, max_workers=4):
    """Save frames using multiple threads."""
    sub_dir, ext = os.path.splitext(video_rel_path)
    sub_dir = process_string(sub_dir)
    frames_dir = os.path.join(frames_root, sub_dir)
    os.makedirs(frames_dir, exist_ok=True)

    # Save frames using multiple threads
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for frame_idx, frame in enumerate(frame_list):
            executor.submit(save_frame, frame, frames_dir, frame_idx)
    
    print(f"save all frames into {frames_dir}")


def process_string(input_string):
    video_name = input_string.split("/")[-1]
    prefix = input_string.split("/")[0]
    # Replace all special characters (except '_') with '-'
    video_name = re.sub(r'[^a-zA-Z0-9_]', '-', video_name)

    # Merge multiple consecutive underscores into one
    video_name = re.sub(r'_+', '_', video_name)

    # Remove the part before and including the first '_'
    video_name = video_name.split('_', 1)[-1]

    video_name = re.sub(r'-+', '-', video_name)

    video_name = video_name.replace("Y2metaapp", "").replace("Y2meta.app", "").replace("Y2meta-app", "")
    video_name = re.sub(r'^[^a-zA-Z0-9]+', '', video_name)

    result = os.path.join(prefix, video_name)
    return result


videos_root = "/data2/Fooling3D/videos"
frames_root = "/data2/Fooling3D/video_frame_sequence"
cache_root = "/data2/Fooling3D/cache"
video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
max_workers = 10

# # Collect video paths
# video_paths = []
# for ext in video_extensions:
#     video_paths += glob.glob(os.path.join(videos_root, '**', ext), recursive=True)

# print(len(video_paths), video_paths[0])


# video_paths_1 = video_paths[:2000]
# video_paths_2 = video_paths[2000:4000]
# video_paths_3 = video_paths[4000:]
# np.save(os.path.join(cache_root, "yao.npy"), video_paths_1)
# np.save(os.path.join(cache_root, "liu.npy"), video_paths_2)
# np.save(os.path.join(cache_root, "zeng.npy"), video_paths_3)


# user = input("input your name (yao, liu, zeng):")
# start_video_idx = input("start from video index:")
# start_video_idx = int(start_video_idx) if start_video_idx else 0
# print(f"Start from {start_video_idx} frame")

# video_paths = np.load(os.path.join(cache_root, "{}.npy".format(user)))
# video_paths.sort()
# for video_idx, video_path in enumerate(video_paths[start_video_idx:]):
#     print("\r\n", "-"*10, f"Processing {video_idx+start_video_idx}/{len(video_paths)} frame", "-"*10, "\r\n")
#     frame_list, frame_count = parser_video(video_path)
#     video_rel_path = "/".join(video_path.split("/")[-2:])
#     print(f"raw video path: {video_path}")
#     print(f"relative path: {video_rel_path}")
#     save_frames(frames_root, video_rel_path, frame_list, max_workers)
#     while True:
#         if input("enter n to process next video:")=='n':
#             break


abnormal_cnt = 0
video_rel_path_list = []
frames_rel_dir_list = []
for user, length in [("liu", 650), ("zeng", 300), ("yao", 970)]:
    video_paths = np.load(os.path.join(cache_root, "{}.npy".format(user)))
    video_paths.sort()
    for video_idx, video_path in enumerate(video_paths[:length]):
        video_rel_path = "/".join(video_path.split("/")[-2:])

        sub_dir, ext = os.path.splitext(video_rel_path)
        sub_dir = process_string(sub_dir)
        if len(sub_dir)==0:
            sub_dir = f"{abnormal_cnt}"
            abnormal_cnt += 1
        
        # if sub_dir.find("DIY_Textured_") != -1:
        #     print(f"raw video path: {video_path}")
        #     print(f"relative path: {video_rel_path}")
        #     print(f"frames_rel_dir: {sub_dir}")

        #     frames_dir = os.path.join(frames_root, sub_dir)
        #     print(frames_dir)
        #     print(os.path.exists(frames_dir))
        #     # print(len(os.listdir(frames_dir)))
        
        frames_dir = os.path.join(frames_root, sub_dir)
        if not os.path.exists(frames_dir) or len(os.listdir(frames_dir))==0:
            print(f"terrible video: {video_rel_path}   ->   {sub_dir}")
            continue
        
        if video_idx%100==0:
            print(f"raw video path: {video_path}")
            print(f"relative path: {video_rel_path}")
            print(f"frames_rel_dir: {sub_dir}")

        video_rel_path_list.append(video_rel_path)
        frames_rel_dir_list.append(sub_dir)


import pandas as pd
df = {"video_rel_path": video_rel_path_list,
      "frames_rel_dir": frames_rel_dir_list}
df = pd.DataFrame(df)
df.to_csv("/data2/Fooling3D/meta_data/frames_metadata.csv", index=False)
