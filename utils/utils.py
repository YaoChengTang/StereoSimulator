import os
import sys
import cv2
import glob
import pickle
import shutil
import multiprocessing

import numpy as np
import pandas as pd



def check_paths(path_list, video_idx=None):
    succ = True
    for path in path_list:
        if not os.path.exists(path):
            # print(f"No such path {video_idx}: {path}")
            succ = False
    return succ


def load_meta_data(path):
    if not os.path.exists(path):
        return {}

    prefix, suffix = os.path.splitext(os.path.basename(path))
    if suffix==".csv":
        data =  pd.read_csv(path)

    elif suffix==".pkl":
        with open(path, 'rb') as f:
            data = pickle.load(f)

    return data

def write_meta_data(data, path):
    prefix, suffix = os.path.splitext(os.path.basename(path))
    if suffix==".csv":
        if isinstance(data, pd.DataFrame):
            data.to_csv(path, index=False)
        else:
            data = pd.DataFrame(data)
            data.to_csv(path, index=False)

    elif suffix==".pkl":
        with open(path, 'wb') as f:
            pickle.dump(data, f)


def load_rgb_image(path):
    if not os.path.exists(path):
        return None
    return cv2.imread(path)

def load_depth_image(path):
    if not os.path.exists(path):
        return None
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def load_mask_image(path):
    if not os.path.exists(path):
        return None
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is not None:
        mask[mask>=128] = 255
        mask[mask<128] = 0
    return mask