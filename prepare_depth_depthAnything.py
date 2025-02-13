import cv2
import torch
from tqdm import tqdm
import sys
sys.path.append('/data4/lzd/iccv25/code/Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2
from matplotlib import pyplot as plt
from utils import video_frame as vf
from utils import image_process
import os
import numpy as np
from fire import Fire
from PIL import Image
import csv
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


def main(
    idx:int, 
):
    folder_path = "/data2/videos/youtube/video"
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()
    folder = f'/data2/Fooling3D/video_frame_sequence/video{idx}'
    video_list = os.listdir(folder)
    video_paths = [os.path.join(folder, i) for i in video_list]
    # print(video_paths[93])
    # print(video_paths[94])
    # print(video_paths[95])
    # exit(0)
    video_paths = video_paths[95:]
    # 打开原文件并直接写回
    for frame_folder in video_paths:
        frame_names = [
            p for p in os.listdir(frame_folder)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p.replace("frame_", ""))[0]))
        frame_list = [np.array(Image.open(os.path.join(frame_folder, frame_name))) for frame_name in frame_names]
        cnt = 0
        save_path = os.path.join(
            f'/data5/fooling-depth/depth/depth{idx}', os.path.basename(frame_folder)
        )
        os.makedirs(save_path, exist_ok=True)
        print(save_path)
        for frame in frame_list:
            depth = model.infer_image(frame)
            cv2.imwrite(os.path.join(save_path, f'frame_{cnt}.png'), np.round(depth * 16).astype(np.uint16))
            cnt += 1

            
if __name__ == "__main__":
    Fire(main)