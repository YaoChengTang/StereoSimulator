import os
import sys
import cv2
import glob
import pickle
import shutil

import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from datetime import datetime
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import normalize
from matplotlib.colors import BoundaryNorm



def fit_plane_ransac(points, threshold=0.01, max_trials=1000, seed = 42):
    """
    使用 RANSAC 拟合三维点的平面。

    参数:
        points (np.ndarray): 三维点集，形状为 (N, 3)。
        threshold (float): 点到平面的距离阈值，用于定义内点。
        max_trials (int): RANSAC 的最大迭代次数。
    
    返回:
        plane_params (np.ndarray): 平面参数 (a, b, c, d)，满足 ax + by + cz + d = 0。
        inliers (np.ndarray): 内点索引数组。
    """
    # 提取三维点的坐标
    X = points[:, :2]  # 使用 x 和 y 作为特征
    y = points[:, 2]   # z 作为目标值

    # 使用 RANSAC 进行平面拟合
    ransac = RANSACRegressor(residual_threshold=threshold, max_trials=max_trials, random_state=seed)
    ransac.fit(X, y)

    # 提取 RANSAC 的平面参数
    inliers = ransac.inlier_mask_
    a, b = ransac.estimator_.coef_
    d = ransac.estimator_.intercept_

    # 平面参数 ax + by + cz + d = 0
    normal_vector = np.array([a, b, -1.0])
    # normal_vector = normalize(normal_vector.reshape(1, -1))[0]  # 法向量归一化
    # d = d / np.linalg.norm(normal_vector)
    plane_params = np.append(normal_vector, d)

    return plane_params, inliers
def guided_filter(input_image, guide_image, radius, eps):
    """
    Applies guided filtering to the input image.

    Parameters:
        input_image (np.ndarray): The input image to be filtered (e.g., mask region).
        guide_image (np.ndarray): The guide image (e.g., source image or mask itself).
        radius (int): Radius of the filter (window size).
        eps (float): Regularization parameter (controls the smoothness).

    Returns:
        np.ndarray: The filtered output image.
    """
    # Convert images to float32
    scale = input_image.max()
    input_image = input_image.astype(np.float32) / scale
    guide_image = guide_image.astype(np.float32) / 255.0
    
    # Apply guided filter
    filtered = cv2.ximgproc.guidedFilter(guide_image, input_image, radius, eps)
    
    # Scale back to 0-255 range and convert to uint8
    filtered = (filtered * scale).astype(np.uint16)
    return filtered


def load_image(fpath):
    mask = Image.open(fpath)
    return np.array(mask)
def fit_plane_svd(points_xyz: np.ndarray):
    """
    使用 SVD 对给定的 3D 点集进行平面拟合，返回 (a, b, c, d)，
    使得 a*x + b*y + c*z + d = 0.
    points_xyz: (N, 3) 的 numpy 数组，每行是 (x, y, z).
    """
    # 构造矩阵 M: shape = (N, 4)
    M = np.hstack([points_xyz, np.ones((points_xyz.shape[0], 1))])  # [x, y, z, 1]
    
    # 对 M 做 SVD 分解
    # M = U * S * V^T
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    
    # Vt 的最后一行（即转置前的最后一列）是最小奇异值对应的特征向量
    plane_params = Vt[-1, :]  # [a, b, c, d]
    
    # 为了后续使用方便，可以让 (a, b, c) 归一化
    a, b, c, d = plane_params
    norm_factor = np.sqrt(a*a + b*b + c*c)
    if norm_factor > 1e-12:
        plane_params /= norm_factor
    
    return plane_params  # (a, b, c, d)



def visualize_ransac(points, inliers, plane_params):
    """
    可视化 RANSAC 内点和外点以及拟合的平面。

    参数:
        points (np.ndarray): 原始三维点集，形状为 (N, 3)。
        inliers (np.ndarray): 内点的布尔索引数组，形状为 (N,)。
        plane_params (np.ndarray): 拟合平面的参数 (a, b, c, d)。
    """
    # 分离内点和外点
    inlier_points = points[inliers]
    outlier_points = points[~inliers]

    # 创建 3D 绘图对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制内点和外点
    ax.scatter(inlier_points[:, 0], inlier_points[:, 1], inlier_points[:, 2], 
               c='b', label='Inliers', s=5)
    ax.scatter(outlier_points[:, 0], outlier_points[:, 1], outlier_points[:, 2], 
               c='r', label='Outliers', s=5)

    x = np.linspace(points[:, 0].min(), points[:, 0].max(), 10)
    y = np.linspace(points[:, 1].min(), points[:, 1].max(), 10)
    x, y = np.meshgrid(x, y)
    z = (-plane_params[0] * x - plane_params[1] * y - plane_params[3]) / plane_params[2]
    ax.plot_surface(x, y, z, color='g', alpha=0.5, label="Fitted Plane")

    # save Image
    plt.savefig('/data4/lzd/iccv25/vis/haha.png')

def Process(Z, source_image, mask_small = None, mask_large = None):
    """
    Process Mono Image, correct illusion region's depth and filter out noise.

    Pramas:
        mask_small(np.array): illusion mask
        mask_large(np.array): support region
        Z(np.array): depth image
        source_image(np.array): source image, RGB image

    Out:
        processed image(np.array)
    """

    if mask_large is None:
        mask_large = mask_small.copy()
        coords_2d = np.argwhere(mask_small)
    else:
        mask_small = mask_small > 0
        mask_large = mask_large > 0
        coords_2d = np.argwhere(mask_large & ~mask_small)
    points_xyz = []
    for (i, j) in coords_2d:
        x_val = i
        y_val = j
        z_val = Z[i, j]
        points_xyz.append([x_val, y_val, z_val])
    points_xyz = np.array(points_xyz)  # (N, 3)
    # return mask_
    plane_params, inliers = fit_plane_ransac(points_xyz, threshold=1, max_trials=200)
    a,b,c,d = plane_params


    rows, cols = np.where(mask_small)
    for (i,j) in zip(rows, cols):
        Z[i,j] = (-d - (a * i + b * j)) / c

    radius = 8  # Window radius
    eps = 1e-2  # Regularization parameter
    filtered_image = guided_filter(Z, source_image, radius, eps)
    Z[mask_large & ~mask_small] = filtered_image[mask_large & ~mask_small]

    return Z


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


def update_meta_data(image_root, mask_root, depth_root, meta_root):
    meta_path  = os.path.join(meta_root, "frames_metadata.csv")
    df = load_meta_data(meta_path)
    video_name_list = df["frames_rel_dir"]

    # data = {}
    cnt = 0
    data = load_meta_data(os.path.join(meta_root, "data_dict.pkl"))
    # for video_name in video_name_list:
    # for video_idx, video_name in enumerate(tqdm(video_name_list[400:1000], desc="Processing Videos", unit="video")):
    for video_idx, video_name in enumerate(tqdm(video_name_list, desc="Processing Videos", unit="video")):
        # video_name  = video_name_list[0]
        # rel_path = "video0/a_Pretty_Switchboard_Mural_youtubeshorts_muralpainting"

        area_type_list = ["illusion", "nonillusion"]
        # print(f"using {video_idx}:{video_name}")

        # Skip this video if video name is invalid.
        if not os.path.basename(video_name):
            continue

        # Skip this video if no corresponding paths of frame, mask and depth.
        if not check_paths([os.path.join(image_root, video_name),
                            os.path.join(mask_root, video_name),
                            os.path.join(depth_root, video_name),], video_idx):
            continue
        
        # Skip this video if it has been processed before
        if video_name in data:
            continue

        # Read the name of all frames and masks
        frame_name_list = os.listdir( os.path.join(image_root, video_name) )
        mask_name_list  = os.listdir( os.path.join(mask_root, video_name) )

        # Skip this video if no corresponding frames or masks are available.
        if len(frame_name_list)==0 or len(mask_name_list)==0:
            continue
        
        frame_name_list.sort()
        mask_name_list.sort()
        data[video_name] = {}
        pre_idx, cur_idx = 0, 0
        # for frame_name in frame_name_list:
        for frame_name in tqdm(frame_name_list, desc=f"Processing Frames in {video_name}", unit="frame", leave=False):
            frame_path = os.path.join(image_root, video_name, frame_name)
            depth_path = os.path.join(depth_root, video_name, frame_name)

            # Obtain the masks for corresponding frame
            # Start from the mask index of last frame, avoiding searching from the scratch
            prefix, suffix = os.path.splitext(frame_name)
            frame_mask_name_list = []
            for mask_name in mask_name_list[pre_idx:]:
                if mask_name.startswith(prefix):
                    frame_mask_name_list.append(mask_name)
                    cur_idx += 1
                elif pre_idx != cur_idx:
                    pre_idx = cur_idx
                    break
            
            # Skip this frame if no corresponding masks or depth data are available.
            if len(frame_mask_name_list)==0 or not os.path.exists(depth_path):
                cnt += 1
                print(f"{video_name}: frame_mask_name_list: {len(frame_mask_name_list)}, ")
                continue

            mask_dict  = {}
            for frame_mask_name in frame_mask_name_list:
                # Parse the mask name: f"{raw_file_name}-{disp_objid:02d}-{out_obj_id:02d}-{area_type}.jpg"
                info = os.path.splitext(frame_mask_name)[0].split("-")
                try:
                    obj_id = info[1]
                    area_type = info[3]
                    if obj_id not in mask_dict:
                        mask_dict[obj_id] = {}
                    mask_dict[obj_id][area_type] = os.path.join(mask_root, video_name, mask_name)
                except Exception as err:
                    print(len(data.keys()))
                    raise Exception(err, frame_mask_name, info, video_name, frame_name, cnt)
            
            data[video_name][frame_name] = {}
            data[video_name][frame_name]["image"] = frame_path
            data[video_name][frame_name]["depth"] = depth_path
            data[video_name][frame_name]["mask"]  = mask_dict
        
        # Check if the value is empty or None
        if not data[video_name]:
            del data[video_name]

    # Save and update the meta data
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    write_meta_data(data, os.path.join(meta_root, f"data_dict_{timestamp}.pkl"))
    shutil.copy(os.path.join(meta_root, f"data_dict_{timestamp}.pkl"), 
                os.path.join(meta_root, f"data_dict.pkl"))

    print(f"saving videos: {len(data.keys())}")
    print(cnt)

# Test
if __name__ == '__main__':
    image_root = "/data2/Fooling3D/video_frame_sequence"
    mask_root  = "/data2/Fooling3D/sam_mask"
    depth_root = "/data5/fooling-depth/depth"
    meta_root  = "/data2/Fooling3D/meta_data"

    update_meta_data(image_root, mask_root, depth_root, meta_root)


    # mask_small_path = "/data4/lzd/iccv25/vis/mask/Y2metaapp3D_Trick_Art_Hole_On_Line_Paper_Traffic_signs_No_horn_honking/326.jpg"
    # mask_large_path = "/data4/lzd/iccv25/vis/mask/Y2metaapp3D_Trick_Art_Hole_On_Line_Paper_Traffic_signs_No_horn_honking/326_all.jpg"
    # depth_path = "/data4/lzd/iccv25/vis/depth_anything/Y2metaapp3D_Trick_Art_Hole_On_Line_Paper_Traffic_signs_No_horn_honking/326.png"
    # source_path = "/data4/lzd/iccv25/vis/imgL/Y2metaapp3D_Trick_Art_Hole_On_Line_Paper_Traffic_signs_No_horn_honking/326.png"
    # mask_small = load_image(mask_small_path)
    # mask_large = load_image(mask_large_path)
    # source_image = load_image(source_path)
    # Z = Process(load_image(depth_path), source_image, mask_small, None)
    # cv2.imwrite('/data4/lzd/iccv25/vis/test_post/1_f_2.png', Z.astype(np.uint16))
    # # mask_small = mask_small > 0
    # # mask_large = mask_large > 0
    # # Z = load_image(depth_path)
    # # plt.imsave('/data4/lzd/iccv25/vis/test_post/1_o.png',Z, cmap='jet')
    # # coords_2d = np.argwhere(mask_large & ~mask_small)
    # # points_xyz = []
    # # for (i, j) in coords_2d:
    # #     x_val = i
    # #     y_val = j
    # #     z_val = Z[i, j]
    # #     points_xyz.append([x_val, y_val, z_val])
    # # points_xyz = np.array(points_xyz)  # (N, 3)
    # # # return mask_
    # # plane_params, inliers = fit_plane_ransac(points_xyz, threshold=1, max_trials=1000)
    # # a,b,c,d = plane_params
    # # print("拟合的平面参数: ax + by + cz + d = 0")
    # # print(f"a, b, c, d = {plane_params}")
    # # print(f"内点数量: {np.sum(inliers)}")
    # # cnt = 0
    # # rows, cols = np.where(mask_small)
    # # for (i,j) in zip(rows, cols):
    # #     Z[i,j] = (-d - (a * i + b * j)) / c



    # # Z = Z.astype(np.uint16)
    # # # plt.imsave('/data4/lzd/iccv25/vis/test_post/1.png',Z, cmap='jet')
    # # cv2.imwrite('/data4/lzd/iccv25/vis/test_post/1_.png', Z.astype(np.uint16))
    # # # Load the source image and mask
    # # radius = 8  # Window radius
    # # eps = 1e-2  # Regularization parameter
    # # filtered_image = guided_filter(Z, source_image, radius, eps)
    # # Z[mask_large & ~mask_small] = filtered_image[mask_large & ~mask_small]
    # # # plt.imsave('/data4/lzd/iccv25/vis/test_post/1_f.png',Z, cmap='jet')
    # # cv2.imwrite('/data4/lzd/iccv25/vis/test_post/1_f_.png', Z.astype(np.uint16))
    # # # img = Image.fromarray(filtered_image)
    # # # img.save('/data4/lzd/iccv25/vis/test_post')