import os
import sys
import cv2
import glob
import pickle
import shutil
import multiprocessing

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

from plane_utils import fit_plane_ransac_cpu2gpu
from utils import load_rgb_image, load_depth_image, load_mask_image
from calibration import ZedCalibration, L515Calibration, WarpCalibration



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
    overlap_mask = (sup_mask > 128) & (ill_mask > 128)
    
    # Count the number of pixels in each region
    sup_area = np.count_nonzero(sup_mask > 128)
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
        ill_mask = (mask_image_dict[obj_id][area_types[0]] > 128).astype(np.int32)
        total_ill_mask += ill_mask  # Accumulate how many obj_id affect each pixel

    # Step 2: Check each obj_id for excessive overlap
    obj_to_remove = set()
    for obj_id in obj_ids:
        ill_mask = (mask_image_dict[obj_id][area_types[0]] > 128).astype(np.int32)
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


def fit_xyd_plane(sup_mask, depth_image, calib_info, device=0, dis_thold=0.01, ransac_n=3, n_itr=1000, batch=100):
    """
    Fit a plane equation (a*u + b*v + c*d + d0 = 0) using Open3D's RANSAC plane segmentation.

    Parameters:
    sup_mask (np.ndarray): Binary mask where 255 indicates valid points.
    depth_image (np.ndarray): Depth map with corresponding depth values.
    calib_info (L515Calibration, ZedCalibration): Camera calibration information.
    device (int): GPU device ID for RANSAC plane fitting.
    dis_thold (float): Distance threshold for plane fitting.
    ransac_n (int): Number of points used per RANSAC iteration.
    n_itr (int): Maximum number of RANSAC iterations. It becomes n_itr//batch in GPU mode.
    batch (int): Number of points considered in parallel during each batched RANSAC iteration.

    Returns:
    tuple: (a, b, c, d0) plane coefficients
    """
    # Get camera intrinsic matrix 
    K = calib_info.get_raw_intrinsic_matrix()

    # Get valid (u, v) coordinates from support mask
    v_coords, u_coords = np.nonzero(sup_mask)
    d_coords = depth_image[v_coords, u_coords]

    # Flatten the pixel coordinates
    x_flat = u_coords.flatten()
    y_flat = v_coords.flatten()
    depth_flat = d_coords.flatten()

    # Project Depth Points to 3D
    # Convert pixel coordinates to normalized camera coordinates (in L515 camera frame)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    X = (x_flat - cx) * depth_flat / fx
    Y = (y_flat - cy) * depth_flat / fy
    Z = depth_flat

    # Build (x, y, z) data
    points = np.column_stack((X, Y, Z))

    if device>=0:
        # print(f"resolution:{depth_image.shape}, points: {points.shape}")
        # print("-"*30)
        batch = 1000
        plane_model, inliers = fit_plane_ransac_cpu2gpu(points, device=device, inlierthresh=dis_thold, 
                                                        batch=batch, numbatchiter=n_itr//batch, verbose=False)
        # print("-"*10, plane_model.shape, inliers.shape)
    else:
        # Build Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # print(f"resolution:{depth_image.shape}, points: {points.shape}")

        # Fit a plane via RANSAC
        plane_model, inliers = pcd.segment_plane(distance_threshold=dis_thold, ransac_n=ransac_n, num_iterations=n_itr)
        # print("-"*10, inliers.shape)

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
    mask = (ill_mask > 128) | (sup_mask > 128)
    v_coords, u_coords = np.nonzero(mask)

    # Avoid division by zero if c is too small
    if np.abs(c) < 1e-6:
        print("Warning: Plane normal component c is too small, skipping correction.")
        return corrected_depth

    # Compute corrected depth values d' = - (a*u + b*v + d0) / c
    corrected_depth[v_coords, u_coords] = - (a * u_coords + b * v_coords + d_0) / c

    # Limit the depth value into 0~1
    corrected_depth = np.clip(corrected_depth, 0, 1)

    return corrected_depth


def correct_depth_with_plane_xyd(plane_model, depth_image, calib_info, mask, 
                                 dis_thold=0.01, eps=np.finfo(float).eps):
    """
    Correct depth values in ill_mask from points in sup_mask using the fitted plane equation.

    Parameters:
    plane_model (tuple): (a, b, c, d) plane coefficients.
    depth_image (np.ndarray): Original depth map.
    mask (np.ndarray): Binary mask for the region (1 = valid).
    dis_thold (float): Distance threshold for plane fitting.
    eps (float): Small value to avoid division by zero.

    Returns:
    np.ndarray: Corrected depth map.
    """
    # Get camera intrinsic matrix 
    K = calib_info.get_raw_intrinsic_matrix()

    # Unpack plane coefficients
    a, b, c, d_0 = plane_model

    # Create a copy of the depth image to store corrected values
    corrected_depth = depth_image.copy()
    max_depth = depth_image.max()

    # Get valid (u, v) coordinates from both masks
    v_coords, u_coords = np.nonzero(mask)
    d_coords = depth_image[v_coords, u_coords]

    # Flatten the pixel coordinates
    x_flat = u_coords.flatten()
    y_flat = v_coords.flatten()
    depth_flat = d_coords.flatten()

    # Project Depth Points to 3D
    # Convert pixel coordinates to normalized camera coordinates (in L515 camera frame)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    
    X = (x_flat - cx) * depth_flat / fx
    Y = (y_flat - cy) * depth_flat / fy
    Z = depth_flat

    # Compute the depth of point on the plane, which is the intersection of the plane and the line from camera center
    Z_new = -Z * d_0 / (a*X + b*Y + c*Z + eps)
    # Update the depth value if the diference between original depth and new depth is larger than the threshold
    Z_update = np.where(np.abs(Z-Z_new) < dis_thold, Z, Z_new)
    # Z_update = Z_new

    # Update the depth value
    corrected_depth[v_coords, u_coords] = Z_update
    # Avoid invalid depth values
    corrected_depth[corrected_depth > max_depth] = 0

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


def detect_foreground_background_boundaries(left_image, ksize=7, threshold=None, dilation_iter=1):
    """
    Detect boundaries where foreground and background meet using edge detection.

    Parameters:
    left_image (np.ndarray): RGB image used for edge detection.
    ksize (int): Sobel kernel size for edge detection.
    threshold (int or None): Gradient magnitude threshold to define edges. If None, adaptive thresholding is used.
    dilation_iter (int): Number of dilation iterations to enlarge the detected edges.

    Returns:
    np.ndarray: Binary mask where edges between foreground and background are detected.
    """
    # Convert to grayscale for edge detection
    gray_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Sobel filter to detect edges
    grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=ksize)
    
    # Compute the gradient magnitude
    grad_mag = cv2.magnitude(grad_x, grad_y)
    
    # If no threshold is provided, use Otsu's thresholding to dynamically calculate the threshold
    if threshold is None:
        _, threshold = cv2.threshold(grad_mag.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Create a binary mask based on the gradient magnitude and threshold
    edge_mask = np.uint8(grad_mag > threshold)  # Detect strong edges
    
    # Enlarge the edge by dilation to capture the boundary areas more effectively
    kernel = np.ones((2*ksize, 2*ksize), np.uint8)  # Square kernel for dilation
    edge_mask = cv2.dilate(edge_mask, kernel, iterations=dilation_iter)
    
    return edge_mask

def selective_guided_filter(depth_image, left_image, ill_mask, sup_mask, 
                            bound_size=7, radius=8, eps=0.01, 
                            edge_ksize=7, edge_threshold=100):
    """
    Apply guided filter only to the boundary regions of ill_mask and sup_mask, avoiding front-to-background crossings.

    Parameters:
    depth_image (np.ndarray): Depth map to be smoothed.
    left_image (np.ndarray): RGB guidance image.
    ill_mask (np.ndarray): Binary mask for illusion region (255=valid, 0=invalid).
    sup_mask (np.ndarray): Binary mask for support region (255=valid, 0=invalid).
    bound_size (int): Filter window size.
    radius (int): Filter radius.
    eps (float): Regularization term.
    edge_ksize (int): Kernel size for detecting foreground-background boundaries.
    edge_threshold (int): Threshold for detecting foreground-background boundaries.

    Returns:
    np.ndarray: Depth map with selective guided filtering applied.
    """
    # Compute mask boundaries (only process edge pixels)
    mask = (ill_mask > 128) | (sup_mask > 128)
    edge_mask = get_mask_boundaries(mask.astype(np.uint8), kernel_size=bound_size)
    edge_mask = (edge_mask > 0).astype(np.uint8)

    # Detect foreground-background boundaries in the left image
    fg_bg_edge_mask = detect_foreground_background_boundaries(left_image, ksize=edge_ksize, threshold=edge_threshold)
    
    # Combine edge masks: exclude foreground-background crossing regions from update
    edge_mask = edge_mask & (fg_bg_edge_mask == 0)

    # Normalize depth map using image width
    scale = depth_image.max()
    depth_normalized = depth_image / scale

    # Convert left_image to grayscale for guided filter
    guide_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY) / 255.0

    if depth_normalized is None or depth_normalized.size == 0:
        raise ValueError("Error: depth_normalized is empty or None. Check depth image processing.")

    # Apply guided filter to the entire depth map
    smoothed_depth = cv2.ximgproc.guidedFilter(guide_image.astype(np.float32), depth_normalized.astype(np.float32), radius, eps)

    # Limit the depth value into 0~1
    smoothed_depth = np.clip(smoothed_depth, 0, 1)

    # Selectively update only the edge regions
    smooth_depth_image = depth_image.copy()
    smooth_depth_image[edge_mask == 1] = smoothed_depth[edge_mask == 1] * scale  # Restore original scale

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


def rectify_depth_image(left_image, depth_image, mask_image_dict, calib_info,
                        area_types, depth_root, depth_path,
                        device=0, min_area=100, thickness=5, 
                        dis_thold=0.01, ransac_n=3, n_itr=1000, 
                        bound_size=7, radius=8, eps=0.01,
                        edge_ksize=7, edge_threshold=100):
    """
    Rectify the depth image using planar fitting and guided filtering for illusion and support regions.

    Parameters:
    - left_image (np.ndarray): RGB guidance image.
    - depth_image (np.ndarray): Input depth map.
    - mask_image_dict (dict): Dictionary mapping object IDs to area masks.
    - calib_info (L515Calibration, ZedCalibration): camera parameters for depth image rectification.
    - area_types (list): List containing two keys ['illusion', 'support'] to extract masks.
    - depth_root (str): Root directory for depth images.
    - depth_path (str): Full path to the original depth image.
    - device (int): GPU device ID for RANSAC plane fitting.
    - min_area (int): Minimum area threshold for connected components in ill_mask.
    - thickness (int): Thickness of boundaries for sup_mask generation.
    - dis_thold (float): Distance threshold for RANSAC plane fitting.
    - ransac_n (int): Number of points used per RANSAC iteration.
    - n_itr (int): Maximum number of RANSAC iterations.
    - bound_size (int): Boundary size for selective smoothing.
    - radius (int): Filter window size for guided filtering.
    - eps (float): Regularization parameter for guided filtering.
    - edge_ksize (int): Kernel size for detecting foreground-background boundaries.
    - edge_threshold (int): Threshold for detecting foreground-background boundaries.

    Returns:
    - np.ndarray: Final smoothed depth map.
    """
    # Normalization to avoid bad plane model due to too large depth value
    rectified_depth_image = depth_image * calib_info.get_depth_scale()
    # rectified_depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
    # rectified_depth_image = depth_image.copy()

    for obj_id, obj_masks_dict in mask_image_dict.items():
        ill_mask = obj_masks_dict.get(area_types[0], None)
        sup_mask = obj_masks_dict.get(area_types[1], None)

        if ill_mask is None:
            continue  # Skip if no illusion mask is available

        # Generate sup_mask if missing or empty
        if sup_mask is None or sup_mask.size == 0:
            # sup_mask = generate_sup_mask_from_ill(ill_mask, min_area=min_area, thickness=thickness)
            sup_mask = ill_mask.copy()
            # save_rectified_depth(depth_root, depth_path, sup_mask, debug_info="maskFromIll")
        
        # print(f"Processing {depth_image.shape} obj_id {obj_id} in {depth_path} with " + \
        #       f"{(ill_mask>128).sum() / ill_mask.size * 100:.3f}% in ill_mask " + \
        #       f"{(sup_mask>128).sum() / ill_mask.size * 100:.3f}% in sup_mask")

        # Fit a plane (a*u + b*v + c*d + d0 = 0)
        # plane_model = fit_uvd_plane_open3d(sup_mask, rectified_depth_image, dis_thold=dis_thold, ransac_n=ransac_n, n_itr=n_itr)
        plane_model = fit_xyd_plane(sup_mask, rectified_depth_image, calib_info, 
                                    device=device, dis_thold=dis_thold, ransac_n=ransac_n, n_itr=n_itr)

        # Ensure a valid plane model was found
        if plane_model is None:
            print(f"Skipping obj_id {obj_id}: No valid plane model found.")
            continue

        # Correct depth values in ill_mask and sup_mask
        # corrected_depth_image = correct_depth_with_plane(plane_model, rectified_depth_image, ill_mask, sup_mask)
        corrected_depth_image = correct_depth_with_plane_xyd(plane_model, rectified_depth_image, calib_info, ill_mask, 
                                                             dis_thold=dis_thold)
        # save_rectified_depth(depth_root, depth_path, 
        #                      corrected_depth_image * (depth_image.max() - depth_image.min()) + depth_image.min(), 
        #                      debug_info="plane_fit")
        rectified_depth_image = corrected_depth_image

        # # Apply guided filtering only to boundary regions
        # smooth_depth_image = selective_guided_filter(corrected_depth_image, left_image, ill_mask, sup_mask, 
        #                                              bound_size=bound_size, radius=radius, eps=eps,
        #                                              edge_ksize=edge_ksize, edge_threshold=edge_threshold)

        # rectified_depth_image = smooth_depth_image

    # Recover the scale of depth
    # rectified_depth_image = rectified_depth_image * (depth_image.max() - depth_image.min()) + depth_image.min()
    rectified_depth_image = rectified_depth_image / calib_info.get_depth_scale()

    # Save the final depth map
    save_rectified_depth(depth_root, depth_path, rectified_depth_image)



def process_frame(video_name, frame_name, frame_dict, calib_dict, area_types, depth_root, device):
    """
    Process a single frame.
    """
    try:
        frame_path = frame_dict["image"]
        depth_path = frame_dict["depth"]
        mask_dict  = frame_dict["mask"]

        l515_calib = L515Calibration(calib_dict["L515"])
        
        left_image = load_rgb_image(frame_path)
        depth_image = load_depth_image(depth_path)
        valid = depth_image > 0

        if left_image is None:
            print(f"No RGB image: {frame_path}")
            return
        
        if depth_image is None:
            print(f"No depth image: {depth_path}")
            return

        mask_image_dict = {}
        obj_id_list = list(mask_dict.keys())
        for obj_id in obj_id_list:
            ill_mask_path = mask_dict[obj_id].get(area_types[0])
            sup_mask_path = mask_dict[obj_id].get(area_types[1])

            # Skip this illusion mask if mask does not exist
            if ill_mask_path is None or not os.path.exists(ill_mask_path):
                continue

            ill_mask = load_mask_image(ill_mask_path)
            # Skip this illusion mask if too few positive values in the mask
            if ill_mask is None or (ill_mask > 128).sum() < 100:
                print(f"Too few positive values in the illusion mask: {ill_mask_path}")
                continue
            
            # Set invalid pixels in the illusion mask to 0
            ill_mask[~valid] = 0

            sup_mask = None
            if sup_mask_path is not None and os.path.exists(sup_mask_path):
                sup_mask = load_mask_image(sup_mask_path)
            # The sup_mask is invalid if too few positive values in the mask
            if sup_mask is not None and (sup_mask > 128).sum() < 100:
                print(f"Too few positive values in the support mask: {sup_mask_path}")
                sup_mask = None
            # # The sup_mask is unreliable when excessive overlap with the illusion region
            # if sup_mask is not None and is_support_unreliable(ill_mask, sup_mask, threshold=0.5):
            #     print(f"The sup_mask is unreliable: {sup_mask_path}")
            #     sup_mask = None
            # Set invalid pixels in the support mask to 0
            if sup_mask is not None:
                sup_mask[~valid] = 0

            mask_image_dict[obj_id] = {}
            mask_image_dict[obj_id][area_types[0]] = ill_mask
            mask_image_dict[obj_id][area_types[1]] = sup_mask

        # Skip this frame if no valid illusion mask
        if len(mask_image_dict.keys()) == 0:
            return

        # # Removes obj_id whose illusion region is largely overlapped with other obj_id's illusion regions.
        # mask_image_dict = clean_dirty_obj_ids(mask_image_dict, area_types, overlap_threshold=0.5)

        # Rectify the depth image using planar fitting and guided filtering for illusion and support regions.
        rectify_depth_image(left_image, depth_image, mask_image_dict, l515_calib, 
                            area_types, depth_root, depth_path,
                            device=device, min_area=100, thickness=5, 
                            dis_thold=0.01, ransac_n=3, n_itr=1000, 
                            bound_size=7, radius=8, eps=0.01)
    
    except Exception as err:
        raise Exception(err, f"video_name: {video_name}   frame_name: {frame_name}   " + \
                             f"frame_dict: {frame_dict}   area_types:{area_types}   depth_root: {depth_root}")

# Function to initialize each process with a specific GPU ID
def init_process(gpu_id):
    if gpu_id != -1:
        torch.cuda.set_device(gpu_id)
        print(f"Process {os.getpid()} is using GPU {gpu_id}")
        
def process_video(video_name, video_dict, calib_dict, area_types, depth_root):
    """
    Process all frames in a single video in parallel.
    """
    # Calculate the number of processes (CPU cores - 10)
    # num_processes = max(1, os.cpu_count() - 10)  # Ensure at least 1 process is used
    num_processes = 3

    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus>0:
        print(f"Using {num_gpus} GPUs for RANSAC plane fitting.")
    else:
        print("Using CPU for RANSAC plane fitting.")

    # Assign a fixed GPU ID to each process
    gpu_ids = [i % num_gpus for i in range(num_processes)]

    # Use multiprocessing to process frames concurrently
    multiprocessing.set_start_method('spawn', force=True)
    with multiprocessing.Pool(processes=num_processes) as pool:
    # with multiprocessing.Pool(processes=num_processes, initializer=init_process, initargs=(gpu_ids,)) as pool:
        # Prepare tasks as a list of arguments for process_frame
        tasks = [
            (video_name, frame_name, frame_dict, calib_dict, area_types, depth_root, i%num_gpus if num_gpus>0 else -1)
            for i, (frame_name, frame_dict) in enumerate(video_dict.items())
        ]

        # Initialize processes with fixed GPU allocation
        for gpu_id in gpu_ids:
            pool.apply_async(init_process, args=(gpu_id,))  # Initialize the process on each GPU

        # Process frames in parallel
        pool.starmap(process_frame, tasks)
        # pool.starmap(process_frame, tasks[:1])


def get_directories_with_image(root_dir):
    dirs_with_image = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # if dirpath.find("TransReflect")!=-1 or dirpath.find("StreetView")!=-1 or dirpath.find("Painting")!=-1:
        #     continue
        # if dirpath.find("PaperOnWall/Window4")==-1 and dirpath.find("TransReflect/Showcase3")==-1 and dirpath.find("Monitor/Cup2")==-1:
        #     continue
        # if dirpath.find("Video/Registration")==-1:
        #     continue
        if dirpath.find("L515_color_image")!=-1 and any(file.lower().endswith(('.png', '.jpg', '.jpeg')) for file in filenames):
            dirs_with_image.append(dirpath)

    return dirs_with_image

def build_meta_data(video_dir_list, image_root, depth_root, mask_root):
    data = {}
    for video_dirpath in video_dir_list:
        video_dict  = {}

        video_name  = os.path.relpath(video_dirpath, image_root)
        object_path = os.path.dirname(video_name)

        for frame_file in os.listdir(video_dirpath):
            frame_id    = os.path.splitext(frame_file)[0]

            # image_path = os.path.join(depth_root, object_path, frame_id, "L515_color_image.png")
            image_path = os.path.join(video_dirpath, frame_file)
            depth_path = os.path.join(depth_root, object_path, frame_id, "L515_depth_image.png")
            mask_dir  = os.path.dirname( image_path.replace("Dataset_new_structure", "SAM") )
            # print("-"*10, image_path, depth_path, mask_dir)
            mask_dict = {}
            for mask_file in glob.glob(os.path.join(mask_dir, f"{frame_id}*.jpg")):
                _, obj_id, _, area_type = os.path.splitext(mask_file)[0].split("-")
                if obj_id not in mask_dict:
                    mask_dict[obj_id] = {}
                mask_dict[obj_id][area_type] = os.path.join(mask_dir, mask_file)

            frame_dict = {
                "image": image_path,
                "depth": depth_path,
                "mask":  mask_dict,
            }
            
            video_dict[frame_id] = frame_dict
        
        calib_dict = {
            "L515": os.path.join(depth_root, object_path, "L515_calib.yaml"),
            "ZED":  os.path.join(depth_root, object_path, "ZED_calib.yaml"),
        }

        data[video_name] = (video_dict, calib_dict)

    return data


if __name__ == '__main__':
    image_root = "/data2/Fooling3D/real_data/Dataset_new_structure"
    mask_root  = "/data2/Fooling3D/real_data/SAM"
    depth_root = "/data2/Fooling3D/real_data/Dataset"

    area_types = ["illusion", "nonillusion"]
    
    video_dir_list = get_directories_with_image(image_root)
    # video_dir_list = video_dir_list[:3]
    data = build_meta_data(video_dir_list, image_root, depth_root, mask_root)
    # import pprint
    # pprint.pprint(data)
    # for key, val in data.items():
    #     print(key, end=": ")
    #     pprint.pprint(val)

    # Process each video in parallel
    start_from_video_name = None
    # start_from_video_name = "Video/Downhill/L515_color_image"
    # start_from_video_name = "Video/Objects/L515_color_image"
    # start_from_video_name = "Video/Driving1/L515_color_image"
    started = False
    for video_name, (video_dict, calib_dict) in tqdm(data.items(), desc="Processing videos"):
        # Start from last failure video
        if start_from_video_name is not None:
            if video_name==start_from_video_name:
                started = True
            if not started:
                continue
        process_video(video_name, video_dict, calib_dict, area_types, depth_root)
        # break


