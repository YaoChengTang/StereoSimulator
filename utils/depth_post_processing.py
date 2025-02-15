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
import open3d as o3d
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

    cnt = 0
    data = {}
    # data = load_meta_data(os.path.join(meta_root, "data_dict.pkl"))
    # for video_name in video_name_list:
    # for video_idx, video_name in enumerate(tqdm(video_name_list[400:1000], desc="Processing Videos", unit="video")):
    for video_idx, video_name in enumerate(tqdm(video_name_list, desc="Processing Videos", unit="video")):
        # video_name  = video_name_list[0]
        # rel_path = "video0/a_Pretty_Switchboard_Mural_youtubeshorts_muralpainting"
        
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
                # print(f"{video_name}: {os.path.join(mask_root, video_name)}, frame_mask_name_list: {len(frame_mask_name_list)}, ")
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
                    mask_dict[obj_id][area_type] = os.path.join(mask_root, video_name, frame_mask_name)
                except Exception as err:
                    print(len(data.keys()))
                    raise Exception(err, frame_mask_name, info, video_name, frame_name, cnt)
            
            data[video_name][frame_name] = {}
            data[video_name][frame_name]["image"] = frame_path
            data[video_name][frame_name]["depth"] = depth_path
            data[video_name][frame_name]["mask"]  = mask_dict

            # print(f"{frame_path}\r\n{depth_path}\r\n{frame_mask_name_list}")
            # print(data)

            # break
        
        # Check if the value is empty or None
        if not data[video_name]:
            del data[video_name]

        # break

    # Save and update the meta data
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    write_meta_data(data, os.path.join(meta_root, f"data_dict_{timestamp}.pkl"))
    shutil.copy(os.path.join(meta_root, f"data_dict_{timestamp}.pkl"), 
                os.path.join(meta_root, f"data_dict.pkl"))

    print(f"saving videos: {len(data.keys())}")
    print(cnt)

def load_rgb_image(path):
    return cv2.imread(path)

def load_depth_image(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def load_mask_image(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def is_support_unreliable(ill_mask, sup_mask, threshold=0.5):
    """
    Determine whether the support region is unreliable due to excessive overlap with the illusion region.

    Parameters:
    ill_mask (np.ndarray): Binary mask (255 for valid, 0 for invalid) for the illusion region.
    sup_mask (np.ndarray): Binary mask (255 for valid, 0 for invalid) for the support region.
    threshold (float): Overlap ratio threshold; default is 0.5 (more overlap than non-overlap).

    Returns:
    bool: True if support region is unreliable, False otherwise.
    """
    # Compute the overlap area
    overlap_mask = (sup_mask == 255) & (ill_mask == 255)
    
    # Count the number of pixels in each region
    sup_area = np.count_nonzero(sup_mask == 255)
    overlap_area = np.count_nonzero(overlap_mask)

    # Determine if the support region is unreliable
    return overlap_area > sup_area * threshold



def clean_dirty_obj_ids(mask_image_dict, area_types, overlap_threshold=0.5):
    """
    Removes obj_id whose illusion region is largely overlapped with other obj_id's illusion regions.

    Parameters:
    mask_image_dict (dict): Dictionary containing obj_id as keys and illusion masks as values.
    area_types (list): List containing the key for illusion masks (e.g., area_types[0] is "illusion").
    overlap_threshold (float): If the overlap ratio exceeds this threshold, the obj_id is removed.

    Returns:
    dict: A cleaned version of mask_image_dict without unreliable obj_id.
    """
    obj_ids = list(mask_image_dict.keys())

    # Get the shape of the masks
    first_mask = mask_image_dict[obj_ids[0]][area_types[0]]
    H, W = first_mask.shape

    # Step 1: Compute total_ill_mask (counts obj_id affecting each pixel)
    total_ill_mask = np.zeros((H, W), dtype=np.int32)
    for obj_id in obj_ids:
        ill_mask = (mask_image_dict[obj_id][area_types[0]] == 255).astype(np.int32)
        total_ill_mask += ill_mask  # Accumulate how many obj_id affect each pixel

    # Step 2: Check each obj_id for excessive overlap
    obj_to_remove = set()
    for obj_id in obj_ids:
        ill_mask = (mask_image_dict[obj_id][area_types[0]] == 255).astype(np.int32)
        obj_ill_pixels = np.count_nonzero(ill_mask)  # Total pixels in this obj_id's illusion area

        if obj_ill_pixels == 0:
            continue  # Skip empty masks

        # Compute overlap pixels (pixels where total_ill_mask > 1)
        overlap_pixels = np.count_nonzero((total_ill_mask > 1) & (ill_mask == 1))

        # Compute overlap ratio
        overlap_ratio = overlap_pixels / obj_ill_pixels

        # Mark obj_id for removal if overlap is too high
        if overlap_ratio > overlap_threshold:
            obj_to_remove.add(obj_id)

    # Step 3: Remove dirty obj_ids
    cleaned_mask_image_dict = {
        obj_id: mask_image_dict[obj_id] for obj_id in obj_ids if obj_id not in obj_to_remove
    }

    return cleaned_mask_image_dict


def generate_sup_mask_from_ill(ill_mask, min_area=100, thickness=5):
    """
    Generate sup_mask based on the edges of large connected components in ill_mask.

    Parameters:
    ill_mask (np.ndarray): Binary mask (255 for valid, 0 for invalid) representing the illusion region.
    min_area (int): Minimum area threshold for selecting connected regions.
    thickness (int): Thickness for contours.

    Returns:
    np.ndarray: The updated sup_mask.
    """
    # Step 1: Compute connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ill_mask, connectivity=8)

    # Step 2: Identify valid large regions (vectorized)
    areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background (index 0)
    valid_labels = np.where(areas > min_area)[0] + 1  # Shift index to match labels

    # Step 3: Create a mask with only large regions (vectorized)
    valid_mask = np.isin(labels, valid_labels).astype(np.uint8) * 255

    if np.count_nonzero(valid_mask) == 0:
        return np.zeros_like(ill_mask)  # No valid regions, return empty mask

    # Step 4: Find all contours in valid_mask
    contours, _ = cv2.findContours(valid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 5: Draw contours on a new support mask
    sup_mask = np.zeros_like(ill_mask)
    cv2.drawContours(sup_mask, contours, -1, 255, thickness=thickness)

    return sup_mask


def fit_uvd_plane_open3d(sup_mask, depth_image, dis_thold=0.01, ransac_n=3, n_itr=1000):
    """
    Fit a plane equation (a*u + b*v + c*d + d0 = 0) using Open3D's RANSAC plane segmentation.

    Parameters:
    sup_mask (np.ndarray): Binary mask where 255 indicates valid points.
    depth_image (np.ndarray): Depth map with corresponding depth values.

    Returns:
    tuple: (a, b, c, d0) plane coefficients
    """
    v_coords, u_coords = np.nonzero(sup_mask)
    d_coords = depth_image[v_coords, u_coords]

    # Build (u, v, d) data
    points = np.column_stack((u_coords, v_coords, d_coords))

    # Build Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Fit a plane via RANSAC
    plane_model, inliers = pcd.segment_plane(distance_threshold=dis_thold, ransac_n=ransac_n, num_iterations=n_itr)

    return tuple(plane_model)  # (a, b, c, d0)


def correct_depth_with_plane(plane_model, depth_image, ill_mask, sup_mask):
    """
    Correct depth values in ill_mask and sup_mask using the fitted plane equation.

    Parameters:
    plane_model (tuple): (a, b, c, d) plane coefficients.
    depth_image (np.ndarray): Original depth map.
    ill_mask (np.ndarray): Binary mask for the illusion region (255 = valid).
    sup_mask (np.ndarray): Binary mask for the support region (255 = valid).

    Returns:
    np.ndarray: Corrected depth map.
    """
    # Unpack plane coefficients
    a, b, c, d_0 = plane_model

    # Create a copy of the depth image to store corrected values
    corrected_depth = depth_image.copy()

    # Get valid (u, v) coordinates from both masks
    mask = (ill_mask == 255) | (sup_mask == 255)
    v_coords, u_coords = np.nonzero(mask)

    # Avoid division by zero if c is too small
    if np.abs(c) < 1e-6:
        print("Warning: Plane normal component c is too small, skipping correction.")
        return corrected_depth

    # Compute corrected depth values d' = - (a*u + b*v + d0) / c
    corrected_depth[v_coords, u_coords] = - (a * u_coords + b * v_coords + d_0) / c

    return corrected_depth


def get_mask_boundaries(mask, kernel_size=7):
    """
    Extract the boundary of a binary mask using morphological operations.

    Parameters:
    mask (np.ndarray): Binary mask (255 for valid, 0 for invalid).

    Returns:
    np.ndarray: Binary mask where 255 represents boundary pixels.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=1)
    eroded = cv2.erode(mask, kernel, iterations=1)
    boundary = dilated - eroded  # Pixels that expanded but not eroded = boundary

    return boundary

def selective_guided_filter(depth_image, left_image, ill_mask, sup_mask, bound_size=7, radius=8, eps=0.01):
    """
    Apply guided filter only to the boundary regions of ill_mask and sup_mask.

    Parameters:
    depth_image (np.ndarray): Depth map to be smoothed.
    left_image (np.ndarray): RGB guidance image.
    ill_mask (np.ndarray): Binary mask for illusion region (255=valid, 0=invalid).
    sup_mask (np.ndarray): Binary mask for support region (255=valid, 0=invalid).
    radius (int): Filter window size.
    eps (float): Regularization term.

    Returns:
    np.ndarray: Depth map with selective guided filtering applied.
    """
    # Compute mask boundaries (only process edge pixels)
    mask = (ill_mask == 255) | (sup_mask == 255)
    edge_mask = get_mask_boundaries(mask.astype(np.uint8), kernel_size=bound_size)
    edge_mask = (edge_mask > 0).astype(np.uint8)

    # Normalize depth map using image width
    image_width = depth_image.shape[1]
    depth_normalized = np.clip(depth_image, 0, image_width).astype(np.float32) / image_width

    # Convert left_image to grayscale for guided filter
    guide_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    if depth_normalized is None or depth_normalized.size == 0:
        raise ValueError("Error: depth_normalized is empty or None. Check depth image processing.")


    # Apply guided filter to the entire depth map
    smoothed_depth = cv2.ximgproc.guidedFilter(guide_image, depth_normalized, radius, eps)

    # Selectively update only the edge regions
    smooth_depth_image = depth_image.copy()
    smooth_depth_image[edge_mask == 1] = smoothed_depth[edge_mask == 1] * image_width  # Restore original scale

    return smooth_depth_image


def save_rectified_depth(depth_root, depth_path, depth_image, debug_info=""):
    """
    Save rectified depth image in a structured directory inside depth_root.

    Parameters:
    depth_root (str): The root directory containing depth files.
    depth_path (str): The full path of the depth file inside depth_root.
    depth_image (np.ndarray): The rectified depth image to be saved.
    """
    # Ensure absolute paths
    depth_root = os.path.abspath(depth_root)
    depth_path = os.path.abspath(depth_path)

    # Extract last folder name from depth_root
    last_folder = os.path.basename(depth_root)
    last_folder_parent = os.path.dirname(depth_root)

    # Define new root directory with "_rect" suffix
    new_root = os.path.join(last_folder_parent, f"{last_folder}_rect")

    # Compute relative path inside depth_root
    relative_subpath = os.path.relpath(depth_path, depth_root)

    # Construct the final save path
    sv_path = os.path.join(new_root, relative_subpath)
    sv_dir = os.path.dirname(sv_path)
    if debug_info is not None and len(debug_info)>0:
        prefix, suffix = os.path.splitext(sv_path)
        sv_path = f"{prefix}-{debug_info}{suffix}"

    # Ensure the save directory exists
    os.makedirs(sv_dir, exist_ok=True)

    # Ensure depth_image is valid before saving
    if depth_image is None or depth_image.size == 0:
        print(f"Error: depth_image is empty. Skipping {sv_path}")
        return

    # Save depth image as uint16 PNG
    cv2.imwrite(sv_path, depth_image.astype(np.uint16))
    print(f"Saved rectified depth: {sv_path}")


def rectify_depth_image(left_image, depth_image, mask_image_dict, area_types, 
                        depth_root, depth_path,
                        min_area=100, thickness=5, 
                        dis_thold=0.01, ransac_n=3, n_itr=1000, 
                        bound_size=7, radius=8, eps=0.01):
    """
    Rectify the depth image using planar fitting and guided filtering for illusion and support regions.

    Parameters:
    - left_image (np.ndarray): RGB guidance image.
    - depth_image (np.ndarray): Input depth map.
    - mask_image_dict (dict): Dictionary mapping object IDs to area masks.
    - area_types (list): List containing two keys ['illusion', 'support'] to extract masks.
    - depth_root (str): Root directory for depth images.
    - depth_path (str): Full path to the original depth image.
    - min_area (int): Minimum area threshold for connected components in ill_mask.
    - thickness (int): Thickness of boundaries for sup_mask generation.
    - dis_thold (float): Distance threshold for RANSAC plane fitting.
    - ransac_n (int): Number of points used per RANSAC iteration.
    - n_itr (int): Maximum number of RANSAC iterations.
    - bound_size (int): Boundary size for selective smoothing.
    - radius (int): Filter window size for guided filtering.
    - eps (float): Regularization parameter for guided filtering.

    Returns:
    - np.ndarray: Final smoothed depth map.
    """
    for obj_id, obj_masks_dict in mask_image_dict.items():
        ill_mask = obj_masks_dict.get(area_types[0], None)
        sup_mask = obj_masks_dict.get(area_types[1], None)

        if ill_mask is None:
            continue  # Skip if no illusion mask is available

        # Generate sup_mask if missing or empty
        if sup_mask is None or sup_mask.size == 0:
            sup_mask = generate_sup_mask_from_ill(ill_mask, min_area=min_area, thickness=thickness)
            save_rectified_depth(depth_root, depth_path, sup_mask, debug_info="maskFromIll")

        # Fit a plane (a*u + b*v + c*d + d0 = 0)
        plane_model = fit_uvd_plane_open3d(sup_mask, depth_image, dis_thold=dis_thold, ransac_n=ransac_n, n_itr=n_itr)

        # Ensure a valid plane model was found
        if plane_model is None:
            print(f"Skipping obj_id {obj_id}: No valid plane model found.")
            continue

        # Correct depth values in ill_mask and sup_mask
        corrected_depth_image = correct_depth_with_plane(plane_model, depth_image, ill_mask, sup_mask)
        save_rectified_depth(depth_root, depth_path, corrected_depth_image, debug_info="plane_fit")

        # Apply guided filtering only to boundary regions
        smooth_depth_image = selective_guided_filter(corrected_depth_image, left_image, ill_mask, sup_mask, radius=radius, eps=eps)

        # Save the final depth map
        save_rectified_depth(depth_root, depth_path, smooth_depth_image)


# Test
if __name__ == '__main__':
    image_root = "/data2/Fooling3D/video_frame_sequence"
    mask_root  = "/data2/Fooling3D/sam_mask"
    depth_root = "/data5/fooling-depth/depth"
    meta_root  = "/data2/Fooling3D/meta_data"

    # update_meta_data(image_root, mask_root, depth_root, meta_root)

    area_types = ["illusion", "nonillusion"]
    data = load_meta_data(os.path.join(meta_root, "data_dict.pkl"))
    for video_name, video_dict in data.items():
        for frame_name, frame_dict in video_dict.items():
            frame_path = frame_dict["image"]
            depth_path = frame_dict["depth"]
            mask_dict  = frame_dict["mask"]

            left_image = load_rgb_image(frame_path)
            depth_image = load_depth_image(depth_path)

            mask_image_dict = {}
            obj_id_list = list(mask_dict.keys())
            for obj_id in obj_id_list:
                mask_image_dict[obj_id] = {}
                
                ill_mask_path = mask_dict[obj_id].get(area_types[0])
                sup_mask_path = mask_dict[obj_id].get(area_types[1])
                
                # Skip this illusion mask if mask doe not exist
                if ill_mask_path is not None and not os.path.exists(ill_mask_path):
                    continue

                ill_mask = load_mask_image(ill_mask_path)
                # Skip this illusion mask if too few postive values in the mask
                if ill_mask is not None and (ill_mask>128).sum()<100:
                    print(f"Too few postive values in the illusion mask: {sup_mask_path}")
                    continue

                sup_mask = None
                if sup_mask_path  is not None and os.path.exists(sup_mask_path):
                    sup_mask = load_mask_image(sup_mask_path)
                # The sup_mask is invalid, if too few postive values in the mask
                if sup_mask is not None and (sup_mask>128).sum()<100:
                    print(f"Too few postive values in the support mask: {sup_mask_path}")
                    sup_mask = None
                # The sup_mask is unreliable when excessive overlap with the illusion region
                if sup_mask is not None and is_support_unreliable(ill_mask, sup_mask, threshold=0.5):
                    print(f"The sup_mask is unreliable: {sup_mask_path}")
                    sup_mask = None
                
                mask_image_dict[obj_id][area_types[0]] = ill_mask
                mask_image_dict[obj_id][area_types[1]] = sup_mask
            
            # Removes obj_id whose illusion region is largely overlapped with other obj_id's illusion regions.
            mask_image_dict = clean_dirty_obj_ids(mask_image_dict, area_types, overlap_threshold=0.5)

            print(f"{frame_path}\r\n{depth_path}\r\n{mask_dict}\r\n\r\n")

            # Rectify the depth image using planar fitting and guided filtering for illusion and support regions.
            # And then save the rectified depth image
            rectify_depth_image(left_image, depth_image, mask_image_dict, area_types, 
                                depth_root, depth_path,
                                min_area=100, thickness=5, 
                                dis_thold=0.01, ransac_n=3, n_itr=1000, 
                                bound_size=7, radius=8, eps=0.01)
            
            print(f"depth_image max: {depth_image.max()}, min: {depth_image.min()}")
            print(f"ill_mask max: {ill_mask.max()}, min: {ill_mask.min()}")
            print(f"mask: {obj_id_list}")

            break
        break


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