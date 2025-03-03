import cv2
import torch
from tqdm import tqdm
import sys
sys.path.append('/data4/lzd/iccv25/code/Depth-Anything-V2')
from matplotlib import pyplot as plt
import os
import numpy as np
from fire import Fire
from PIL import Image
import csv
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


def check(List):
    for i in range(len(List)):
        expected_name = f'frame_{i}.png'
        # print(expected_name)
        if expected_name not in List:
            print(f"Expected {expected_name} not found in the list!")
            return False    
    return True
def rename_frames(save_path, frame_names):
    # 遍历文件夹中的所有文件
    for idx, frame_name in enumerate(frame_names):
        # 获取当前文件的路径
        old_path = os.path.join(save_path, f'frame_{idx}.png')
        
        # 获取新的文件名
        new_name = frame_name  # 使用 frame_names 中的名字作为新名称
        new_path = os.path.join(save_path, new_name)
        # print(old_path, new_path)
        # continue
        # 重命名文件
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            print(f"Renamed {old_path} to {new_path}")
        else:
            print(f"File {old_path} not found!")

def main(
    idx:int, 
):
    folder = f'/data2/Fooling3D/video_frame_sequence/video{idx}'
    video_list = os.listdir(folder)
    video_paths = [os.path.join(folder, i) for i in video_list]
    # video_paths = video_paths[95:]
    # 打开原文件并直接写回

    flag = False
    for frame_folder in video_paths:
        frame_names = [
            p for p in os.listdir(frame_folder)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
        ]
        save_path = os.path.join(
            f'/data5/fooling-depth/depth/video{idx}', os.path.basename(frame_folder)
        )
        depth_names = [
            p for p in os.listdir(save_path)
            if os.path.splitext(p)[-1] in [".png"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p.replace("frame_", ""))[0]))
        depth_names.sort(key=lambda p: int(os.path.splitext(p.replace("frame_", ""))[0]))
        # print(save_path)
        # print(frame_folder)
        try:
            assert len(depth_names) == len(frame_names)
        except:
            print(save_path)
            print(f"Expected {len(frame_names)} frames but got {len(depth_names)} frames")
            continue
        for depth,frame in zip(depth_names, frame_names):
            print(depth, frame)
            if depth != frame:
                print(save_path + f"Expected {frame} but got {depth}")
                break
        

            
if __name__ == "__main__":
    Fire(main)