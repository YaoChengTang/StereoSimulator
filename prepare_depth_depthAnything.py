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
    folder = f'/data2/videos/youtube/video{idx}'
    video_list = os.listdir(folder)
    video_paths = [os.path.join(folder, i) for i in video_list]
    csv_path = folder + ".csv"
    with open(csv_path, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        rows = list(reader)  # 读取所有行

    # 打开原文件并直接写回
    infile.close()
    for row in rows:
        if row['depth'] == '0':  # depth 为 0 时进行处理
            vpath = os.path.join(folder, row['video_name'])
            try:
                frame_list, fps, frame_count = vf.parser_video(vpath, 15)
            except:
                print('CAN Not Open Video')
                continue
            cnt = 0
            save_path = os.path.join(
                f'/data2/videos/depth/depth{idx}', os.path.splitext(os.path.basename(vpath))[0]
            )
            os.makedirs(save_path, exist_ok=True)
            print(save_path)
            for frame in tqdm(frame_list):
                depth = model.infer_image(frame)
                cv2.imwrite(os.path.join(save_path, f'{cnt}.jpg'), np.round(depth * 16).astype(np.uint16))
                cnt += 1
            row['depth'] = 1
            with open(csv_path, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()  # 写入表头
                writer.writerows(rows)  # 写入所有修改后的行
            file.close()

            
if __name__ == "__main__":
    Fire(main)