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
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'





if __name__ == '__main__':
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
    vpath = '/data2/videos/youtube/video5/Y2metaapp3D_Trick_Art_Hole_On_Line_Paper_Traffic_signs_No_horn_honking.mp4'
    frame_list, fps, frame_count = vf.parser_video(vpath, 15)
    img_process = image_process.ImageProcess()
    # raw_img = cv2.imread('/data4/lzd/datasets/booster/train/balanced/Bedroom/camera_00/im0.png')
    cnt = 0
    videoName = vpath.split('/')[-1].split('.')[0]
    # os.mkdir('imgR/' + videoName)
    # os.mkdir('imgL/' + videoName)
    # os.mkdir('imgR_noFill/' + videoName)
    # os.mkdir('mask/' + videoName)
    # os.mkdir('depth/' + videoName)
    save_path = os.path.join('/data4/lzd/iccv25/vis/depth_anything/', videoName)
    save_path_L = os.path.join('/data4/lzd/iccv25/vis/imgL/', videoName)
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path_L, exist_ok=True)
    for i in tqdm(frame_list):
        # depth = model.infer_image(i) # HxW raw depth map in numpy
        # print(depth)
        # cv2.imwrite(os.path.join(save_path, f'{cnt}.png'), np.round(depth * 16).astype(np.uint16))

        # imgR, imgRFill, mask = img_process.project_image(i, depth/4)
        # imgRFill.save(f'imgR/{videoName}/{cnt}.png')
        # imgR.save(f'imgR_noFill/{videoName}/{cnt}.png')
        # mask.save(f'mask/{videoName}/{cnt}.png')
        cv2.imwrite(os.path.join(save_path_L, f'{cnt}.png'), i)
        # # cv2.imwrite(f'mask/{cnt}.png',mask)
        # plt.imsave(f'depth/{videoName}/{cnt}.png', depth/4, cmap='jet')
        # np.save(f'../vis/depth/{videoName}/{cnt}.npy', depth/4)
        cnt += 1
        # break




