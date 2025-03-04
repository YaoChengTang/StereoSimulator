import os
from glob import glob
import argparse
import shutil
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
from depth_post_process import rectify_depth_image
left_root = "/data2/Fooling3D/real_data/testing_new/left"
# sv_root = "/data2/Fooling3D/real_data/testing_new/left"

def load_mask(path_name, file_name):
    img_name = os.path.basename(file_name).replace('.png', '').replace("frame_","")
    pattern = os.path.join(path_name, f"{img_name}*-illusion.jpg")
    matching_files = glob(pattern)
    nocc_pix = None
    # print(matching_files)
    for file in matching_files:
        if nocc_pix is None:
            nocc_pix = (imageio.imread(file) == 255)
        else:
            nocc_pix = (nocc_pix | (imageio.imread(file) == 255))
        
    return nocc_pix
    # plt.imsave(f"{img_name}.jpg",nocc_pix, cmap='gray')
    # exit(0)







if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', help="Data name type")
    args = parser.parse_args()

    folders = []
    for root_, dirs, files in os.walk(left_root):
        for dir_name in dirs:
            if args.name in dir_name:
                folders.append(os.path.join(root_, dir_name))
    assert len(folders) > 0

    if args.name == "TransReflect":
        for folder in folders:
            files = os.listdir(folder)
            for file in files:
                file_path = os.path.join(folder, file)
                mask_path = file_path.replace('left', "mask").replace('png','jpg')
                path_name = os.path.join("/data2/Fooling3D/real_data/SAM",\
                            args.name, os.path.basename(folder).split('_')[-1], "zed_left_color_image")
                mask = load_mask(path_name, file)
                mask = mask.astype(np.uint8) * 255
                height, width = mask.shape[:2]
                mask[:, : width * 3 // 5] = 0
                # print(mask)
                cv2.imwrite(mask_path, mask)
                # exit(0)
                print(mask_path)
    elif args.name == "PaperOnFloor":

    else:
        raise ValueError("No this name")