import os
import sys
import cv2
import glob
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.video_frame import parser_video


def save_frame(frame, frames_dir, frame_idx):
    """Save a single frame to disk."""
    frame_filename = os.path.join(frames_dir, f"frame_{frame_idx:04d}.png")
    cv2.imwrite(frame_filename, frame)


def save_frames(frames_root, video_rel_path, frame_list, max_workers=4):
    """Save frames using multiple threads."""
    sub_dir, ext = os.path.splitext(video_rel_path)
    frames_dir = os.path.join(frames_root, sub_dir)
    # os.makedirs(frames_dir, exist_ok=True)
    print(f"save all frames into {frames_dir}")

    # # Save frames using multiple threads
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     for frame_idx, frame in enumerate(frame_list):
    #         executor.submit(save_frame, frame, frames_dir, frame_idx)



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

user = input("input your name (yao, liu, zeng):")
video_paths = np.load(os.path.join(cache_root, "{}.npy".format(user)))
for video_idx, video_path in enumerate(video_paths):
    print("\r\n", "-"*10, f"Processing {video_idx}/{len(video_paths)} frame", "-"*10, "\r\n")
    frame_list, frame_count = parser_video(video_path)
    video_rel_path = "/".join(video_path.split("/")[-2:])
    print(f"raw video path: {video_path}")
    print(f"relative path: {video_rel_path}")
    save_frames(frames_root, video_rel_path, frame_list, max_workers)
    while True:
        if input("enter n to process next video:")=='n':
            break


