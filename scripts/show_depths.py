import numpy as np
import os
import cv2
if __name__ == "__main__":
    path = '/data4/lzd/iccv25/data/depth_sets0/Y2metaappDIY_Wall_Painting__How_to_Create_a_Simple_Bulb_Design.npz'
    x = np.load(path)['depth']
    folder = os.path.join('/data4/lzd/iccv25/vis/depth', path.split('/')[-1][:-4])
    if not os.path.exists(folder):
        os.mkdir(folder)
    for i in range(x.shape[0]):
        cv2.imwrite(os.path.join(folder, f'{i}.png'), np.round(x[i] * 512).astype(np.uint16))