import cv2
import torch
from tqdm import tqdm
import sys
sys.path.append('/data4/lzd/iccv25/code/Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2
from matplotlib import pyplot as plt
import video_frame as vf
import os
import numpy as np
from fire import Fire
from PIL import Image
import csv
from glob import glob
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
            
if __name__ == "__main__":
    mask_path = "/data2/Fooling3D/sam_mask_beta/"
    mask_list = sorted(glob(mask_path+"/*/*/*"))
    # print(mask_list)
    unique_paths = set()

    for file_path in mask_list:
        dir_part = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        frame_part = file_name.split("-")[0] 
        unique_path = os.path.join(dir_part, frame_part)
        # print(unique_path)
        # exit(0)
        unique_paths.add(unique_path)

    result = sorted(list(unique_paths))
    video_list = []
    for path in result:
        video_list.append(path.replace('sam_mask_beta', 'video_frame_sequence_beta/')+'.png')
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'../checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()
    for frame_name in video_list:
        save_path = frame_name.replace("/data2/Fooling3D/video_frame_sequence_beta","/data5/fooling-depth/depth")
        if os.path.exists(save_path):
            continue
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        frame = np.array(Image.open(frame_name))
        print(save_path)
        depth = model.infer_image(frame)
        cv2.imwrite(save_path, np.round(depth * 16).astype(np.uint16))