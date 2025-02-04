import os
import cv2
import yaml
import argparse
import numpy as np

from datetime import datetime



def init_paths(args):
    # Create scene directory if it doesn't exist
    scene_name = args.scene_name  # Get the scene name from arguments
    if not os.path.exists(scene_name):
        os.makedirs(scene_name)


def stack_images_hor(stacked_image, images):
    h1, w1 = stacked_image.shape[:2]
    h2, w2 = images.shape[:2]
    
    max_height = max(h1, h2)
    total_width = w1 + w2
    
    if len(stacked_image.shape)==3 or len(images.shape)==3:
        channel = stacked_image.shape[2] if len(stacked_image.shape)==3 else images.shape[2]
        result = np.zeros((max_height, total_width, channel), dtype=np.uint8)
    else:
        result = np.zeros((max_height, total_width), dtype=np.uint8)

    if len(stacked_image.shape)==3 and len(images.shape)==2:
        images = images[..., np.newaxis]
    elif len(stacked_image.shape)==2 and len(images.shape)==3:
        stacked_image = stacked_image[..., np.newaxis]
    
    result[(max_height-h1)//2:(max_height-h1)//2 + h1, :w1] = stacked_image
    result[(max_height-h2)//2:(max_height-h2)//2 + h2, w1:w1+w2] = images
    
    return result


def stack_images_ver(stacked_image, images):
    h1, w1 = stacked_image.shape[:2]
    h2, w2 = images.shape[:2]
    
    max_width = max(w1, w2)
    total_height = h1 + h2
    
    if len(stacked_image.shape)==3 or len(images.shape)==3:
        channel = stacked_image.shape[2] if len(stacked_image.shape)==3 else images.shape[2]
        result = np.zeros((total_height, max_width, channel), dtype=np.uint8)
    else:
        result = np.zeros((total_height, max_width), dtype=np.uint8)

    if len(stacked_image.shape)==3 and len(images.shape)==2:
        images = images[..., np.newaxis]
    elif len(stacked_image.shape)==2 and len(images.shape)==3:
        stacked_image = stacked_image[..., np.newaxis]
    
    result[:h1, (max_width-w1)//2:(max_width-w1)//2 + w1] = stacked_image
    result[h1:h1+h2, (max_width-w2)//2:(max_width-w2)//2 + w2] = images
    
    return result



def save_data(args, data, idx):
    scene_name = args.scene_name  # Get the scene name from arguments
    root = args.root
    if type(idx) is int:
        sv_dir = os.path.join(root, scene_name, f"{idx:04d}")
    elif type(idx) is str:
        sv_dir = os.path.join(root, scene_name, f"{idx}")
    else:
        raise Exception(f"No support for such type of idx: {type(idx)}")
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)
    
    keys = []
    for key, value in data.items():
        filename = os.path.join(sv_dir, f"{key}.png")
        cv2.imwrite(filename, value)
        keys.append(key)
    print(f"Saved {keys} into {sv_dir}.")



def get_image_paths(root="./Dataset", scene_name="calib"):
    calib_dir = os.path.join(root, scene_name)

    # Initialize lists to store image paths
    left_img_paths = []
    right_img_paths = []
    rgb_img_paths = []
    depth_img_paths = []

    # Traverse all directories and files under calib_dir
    for frame_dir in os.listdir(calib_dir):
        if not os.path.isdir( os.path.join(calib_dir, frame_dir) ):
            continue
        left_img_paths.append( os.path.join(calib_dir, frame_dir, "zed_left_color_image.png") )
        right_img_paths.append( os.path.join(calib_dir, frame_dir, "zed_right_color_image.png") )
        rgb_img_paths.append( os.path.join(calib_dir, frame_dir, "L515_color_image.png") )
        depth_img_paths.append( os.path.join(calib_dir, frame_dir, "L515_depth_image.png") )

    return left_img_paths, right_img_paths, rgb_img_paths, depth_img_paths


def colorize_depth_rgb(depth_img, rgb_img):
    # Normalize depth to 0-255 for better visualization
    depth_img_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)

    # Apply a color map to the depth map for visualization
    depth_img_colored = cv2.applyColorMap(depth_img_normalized.astype(np.uint8), cv2.COLORMAP_JET)

    print("-"*10, rgb_img.shape, rgb_img.dtype, depth_img_colored.shape, depth_img_colored.dtype)

    # Blend the depth map (as a color map) with the RGB image
    alpha = 0.6  # Transparency for blending
    overlay = cv2.addWeighted(rgb_img, 1 - alpha, depth_img_colored, alpha, 0)

    return overlay


def remap_depth_to_zed(img_depth, img_ZED, K_L515, dist_L515, K_ZED, dist_ZED, R, T):
    """
    Remap the L515 depth map (img_depth) to the ZED left image.
    
    Parameters:
        img_depth (np.array): The depth map from the L515 (height x width).
        K_L515 (np.array): Intrinsic matrix for L515.
        dist_L515 (np.array): Distortion coefficients for L515.
        K_ZED (np.array): Intrinsic matrix for ZED's left camera.
        dist_ZED (np.array): Distortion coefficients for ZED's left camera.
        R (np.array): Rotation matrix from L515 to ZED.
        T (np.array): Translation vector from L515 to ZED.
        
    Returns:
        np.array: Depth map on ZED's left image.
    """
    h, w = img_depth.shape
    H, W, _ = img_ZED.shape
    
    # Create mesh grid for pixel coordinates
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Flatten the pixel coordinates
    x_flat = x.flatten()
    y_flat = y.flatten()
    depth_flat = img_depth.flatten()

    # 1. Project Depth Points to 3D (in L515 camera coordinate system)
    # Convert pixel coordinates to normalized camera coordinates (in L515 camera frame)
    fx_L515 = K_L515[0, 0]
    fy_L515 = K_L515[1, 1]
    cx_L515 = K_L515[0, 2]
    cy_L515 = K_L515[1, 2]
    
    X_L515 = (x_flat - cx_L515) * depth_flat / fx_L515
    Y_L515 = (y_flat - cy_L515) * depth_flat / fy_L515
    Z_L515 = depth_flat  # Depth values are in meters

    # Stack the 3D points (in L515 camera coordinate system)
    points_3D_L515 = np.vstack((X_L515, Y_L515, Z_L515))

    # 2. Transform 3D Points to ZED's Coordinate System
    # Apply the extrinsic transformation: ZED = R * L515 + T
    if len(T.shape)==1:
        T = T[:, np.newaxis]
    points_3D_ZED = R @ points_3D_L515 + T

    # 3. Project 3D Points onto ZED's Left Image
    fx_ZED = K_ZED[0, 0]
    fy_ZED = K_ZED[1, 1]
    cx_ZED = K_ZED[0, 2]
    cy_ZED = K_ZED[1, 2]
    
    # Project the 3D points back onto the ZED image plane
    x_projected = (fx_ZED * points_3D_ZED[0] / points_3D_ZED[2]) + cx_ZED
    y_projected = (fy_ZED * points_3D_ZED[1] / points_3D_ZED[2]) + cy_ZED
    depth_projected = depth_flat

    # 4. Reconstruct the Depth Map on ZED's Left Image
    depth_map_ZED, invalid_mask = construct_geometry_target(x_projected, 
                                                          y_projected, 
                                                          depth_projected, 
                                                          H, W)
    return depth_map_ZED, invalid_mask


def construct_geometry_target(x_projected, y_projected, depth_projected, H, W):
    # Round the projected coordinates to nearest integer pixels
    x_projected_int = np.round(x_projected).astype(int)
    y_projected_int = np.round(y_projected).astype(int)

    # Assign depth map with minimum depth value
    depth_map_ZED = assign_minimum_depth(x_projected_int, y_projected_int, depth_projected, H, W)

    x_projected_int = np.concatenate([np.ceil(x_projected).astype(int), np.floor(x_projected).astype(int)])
    y_projected_int = np.concatenate([np.ceil(y_projected).astype(int), np.floor(y_projected).astype(int)])
    depth_projected_cat = np.concatenate([depth_projected, depth_projected])
    depth_map_hole = assign_minimum_depth(x_projected_int, y_projected_int, depth_projected_cat, H, W)

    depth_map_ZED[np.isinf(depth_map_ZED)] = 0
    depth_map_hole[np.isinf(depth_map_hole)] = 0
    depth_map_ZED = depth_map_ZED + (np.abs(depth_map_ZED)<np.finfo(np.float32).eps) * depth_map_hole

    invalid_mask = np.abs(depth_map_ZED)<np.finfo(np.float32).eps

    return depth_map_ZED, invalid_mask


def assign_minimum_depth(x_projected_int, y_projected_int, depth_projected, H, W):
    depth_map_ZED = np.zeros((H,W))

    # Filter out points that project outside the image bounds (negative or too large values)
    valid = (x_projected_int >= 0) & (x_projected_int < W) & (y_projected_int >= 0) & (y_projected_int < H)
    x_projected_int = x_projected_int[valid]
    y_projected_int = y_projected_int[valid]
    depth_projected = depth_projected[valid]

    # print(f"img_ZED: {img_ZED.shape}, depth_map_ZED: {depth_map_ZED.shape} " + \
    #       f"x_projected_int: {x_projected_int.shape}, y_projected_int: {y_projected_int.shape}, depth_projected: {depth_projected.shape}")
    # print(f"x_projected_int min: {x_projected_int.min()}, x_projected_int.max: {x_projected_int.max()} " +\
    #       f"y_projected_int min: {y_projected_int.min()}, y_projected_int.max: {y_projected_int.max()}")

    # Assign depth map with the minimum depth for each pixel using NumPy operations (no loops)
    # Flatten the pixel coordinates to map them into a single index
    pixel_indices = np.ravel_multi_index((y_projected_int, x_projected_int), (H, W))

    # Create a temporary depth map with infinity (for minimum depth calculation)
    depth_map_temp = np.inf * np.ones(H * W)

    # Use np.minimum to apply the minimum depth value for each pixel
    np.minimum.at(depth_map_temp, pixel_indices, depth_projected)

    # Reshape depth_map_temp back to the image shape
    depth_map_ZED = depth_map_temp.reshape((H, W))

    # depth_map_ZED[depth_map_ZED == np.inf] = 0

    return depth_map_ZED


def repair_depth(img_depth_remap, invalid_mask_remap, img_ZED, args=None, frame_idx=None):
    img_depth_remap = img_depth_remap.astype(np.float32)
    # img_depth_remap_dense = cv2.ximgproc.guidedFilter(guide=img_ZED, src=img_depth_remap, radius=16, eps=1000)
    # img_depth_remap_dense = cv2.inpaint(img_depth_remap, invalid_mask_remap.astype(np.uint8), inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(invalid_mask_remap.astype(np.uint8) * 255, connectivity=8)
    area_threshold = 100
    areas = stats[1:, cv2.CC_STAT_AREA]   # Extract areas from stats (skip the first row, which is the background)
    small_labels = np.where(areas <= area_threshold)[0] + 1  # Create a mask for labels with areas <= threshold. Adjust to 1-based labels
    small_mask = np.isin(labels, small_labels).astype(np.uint8) * 255
    img_depth_remap_dense = cv2.inpaint(img_depth_remap, small_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    invalid_mask_dense = np.abs(img_depth_remap_dense)<np.finfo(np.float32).eps
    img_depth_remap_dense_smoothed = cv2.ximgproc.guidedFilter(
        guide=img_ZED,
        src=cv2.inpaint(img_depth_remap, invalid_mask_remap.astype(np.uint8), inpaintRadius=3, flags=cv2.INPAINT_TELEA).astype(np.float32),
        radius=3,
        eps=1e-3,
    )
    img_depth_remap_dense_smoothed = img_depth_remap_dense_smoothed * (~invalid_mask_dense)
    img_depth_remap_repair = img_depth_remap * (~invalid_mask_remap) + img_depth_remap_dense_smoothed * invalid_mask_remap

    if hasattr(args, 'vis_debug') and args.vis_debug:
        data = {
            "img_depth_remap": img_depth_remap.astype(np.uint16),
            "img_depth_remap_dense": img_depth_remap_dense.astype(np.uint16),
            "img_depth_remap_dense_smoothed": img_depth_remap_dense_smoothed.astype(np.uint16),
            "img_depth_remap_repair": img_depth_remap_repair.astype(np.uint16),
        }
        save_data(args, data, frame_idx)

    invalid_mask = np.abs(img_depth_remap_repair)<np.finfo(np.float32).eps

    return img_depth_remap_repair, invalid_mask