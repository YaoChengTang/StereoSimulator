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


def stack_images(image_list, placeholder_size=None, orientation='horizontal'):
    """
    Stacks a list of images either horizontally or vertically, centering them along the
    perpendicular axis. For any None in the list, a placeholder image (filled with zeros)
    is created.
    
    If placeholder_size is None, then:
      - For horizontal stacking, the placeholder width is set to the maximum valid width.
      - For vertical stacking, the placeholder height is set to the maximum valid height.
    
    Parameters:
        image_list (list of np.ndarray or None): List of images to stack. Each image can be
            grayscale (2D array) or color (3D array).
        placeholder_size (int or None): For horizontal stacking, this value is used as the placeholder's width;
            for vertical stacking, it is used as the placeholder's height. If None, the value defaults to
            max(valid_widths) for horizontal or max(valid_heights) for vertical stacking.
        orientation (str): Either 'horizontal' or 'vertical', controlling the stacking direction.
    
    Returns:
        np.ndarray: The resulting stacked image.
                    Returns None if image_list is empty.
    """
    if not image_list:
        return None

    # Determine if any valid image is color and collect valid dimensions.
    is_color = False
    valid_heights = []
    valid_widths = []
    for img in image_list:
        if img is not None:
            h, w = img.shape[:2]
            valid_heights.append(h)
            valid_widths.append(w)
            if len(img.shape) == 3:
                is_color = True

    # Define default dimensions based on valid images.
    default_height = max(valid_heights) if valid_heights else 100
    default_width = max(valid_widths) if valid_widths else 100

    # Use default placeholder_size if not provided.
    if placeholder_size is None:
        if orientation == 'horizontal':
            placeholder_size = default_width
        elif orientation == 'vertical':
            placeholder_size = default_height
        else:
            raise ValueError("Orientation must be either 'horizontal' or 'vertical'")

    # Compute canvas dimensions.
    if orientation == 'horizontal':
        canvas_height = default_height
        # For each image, if None, use placeholder_size as width.
        widths = [img.shape[1] if img is not None else placeholder_size for img in image_list]
        canvas_width = sum(widths)
    elif orientation == 'vertical':
        canvas_width = default_width
        # For each image, if None, use placeholder_size as height.
        heights = [img.shape[0] if img is not None else placeholder_size for img in image_list]
        canvas_height = sum(heights)
    else:
        raise ValueError("Orientation must be either 'horizontal' or 'vertical'")

    # Determine number of channels for the output.
    channels = 1
    if is_color:
        for img in image_list:
            if img is not None and len(img.shape) == 3:
                channels = img.shape[2]
                break

    # Process the image list by replacing None with a placeholder image.
    processed_images = []
    for img in image_list:
        if img is None:
            # Create a placeholder image with all zeros.
            if orientation == 'horizontal':
                # Placeholder shape: (canvas_height, placeholder_size, channels) for color,
                # or (canvas_height, placeholder_size) for grayscale.
                if is_color:
                    placeholder = np.zeros((canvas_height, placeholder_size, channels), dtype=np.uint8)
                else:
                    placeholder = np.zeros((canvas_height, placeholder_size), dtype=np.uint8)
            else:  # vertical orientation
                # Placeholder shape: (placeholder_size, canvas_width, channels) for color,
                # or (placeholder_size, canvas_width) for grayscale.
                if is_color:
                    placeholder = np.zeros((placeholder_size, canvas_width, channels), dtype=np.uint8)
                else:
                    placeholder = np.zeros((placeholder_size, canvas_width), dtype=np.uint8)
            processed_images.append(placeholder)
        else:
            # If the overall output is color but the current image is grayscale, convert it.
            if is_color and len(img.shape) == 2:
                img = np.repeat(img[..., np.newaxis], channels, axis=2)
            processed_images.append(img)

    # Create a blank canvas for the result.
    if is_color:
        result = np.zeros((canvas_height, canvas_width, channels), dtype=np.uint8)
    else:
        result = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

    # Place each image onto the canvas.
    if orientation == 'horizontal':
        current_x = 0
        for img in processed_images:
            h, w = img.shape[:2]
            # Compute vertical offset to center the image.
            y_offset = (canvas_height - h) // 2
            result[y_offset:y_offset+h, current_x:current_x+w] = img
            current_x += w
    else:  # vertical stacking
        current_y = 0
        for img in processed_images:
            h, w = img.shape[:2]
            # Compute horizontal offset to center the image.
            x_offset = (canvas_width - w) // 2
            result[current_y:current_y+h, x_offset:x_offset+w] = img
            current_y += h

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


def colorize_depth_rgb(depth_img, rgb_img=None, color_type="mix"):
    if color_type in ["single", "both"]:
        vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)

    if color_type in ["mix", "both"]:
        # Normalize depth to 0-255 for better visualization
        depth_img_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)

        # Apply a color map to the depth map for visualization
        depth_img_colored = cv2.applyColorMap(depth_img_normalized.astype(np.uint8), cv2.COLORMAP_JET)

        # print("-"*10, rgb_img.shape, rgb_img.dtype, depth_img_colored.shape, depth_img_colored.dtype)

        # Blend the depth map (as a color map) with the RGB image
        alpha = 0.6  # Transparency for blending
        overlay = cv2.addWeighted(rgb_img, 1 - alpha, depth_img_colored, alpha, 0)

        if color_type=="mix":
            vis = overlay
        else:
            vis = stack_images_ver(overlay, vis)

    return vis


def remap_depth_to_zed(img_depth, img_ZED, K_L515, dist_L515, K_ZED, dist_ZED, R, T, depth_scale, 
                       args=None, frame_idx=None):
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
    # Rescale to meters
    img_depth = img_depth * depth_scale

    # Upsample the source depth map to avoid misassignment of depth from BG to FG due to the sparsity after the warping
    up_scale = 3
    img_depth_raw = img_depth
    img_depth = cv2.resize(img_depth, (0, 0), fx=up_scale, fy=up_scale, interpolation=cv2.INTER_NEAREST)
    K_L515 = K_L515 * up_scale

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
    depth_projected = points_3D_ZED[2]

    # 4. Reconstruct the Depth Map on ZED's Left Image
    depth_map_ZED, invalid_mask = construct_geometry_target(x_projected, 
                                                          y_projected, 
                                                          depth_projected, 
                                                          H, W)

    # Rescale to original scale
    depth_map_ZED = depth_map_ZED / depth_scale
    
    if args is not None and hasattr(args, 'vis_debug') and args.vis_debug:
        data = {
            "img_depth_raw": (img_depth_raw / depth_scale).astype(np.uint16),
            "img_depth_up": (img_depth / depth_scale).astype(np.uint16),
        }
        save_data(args, data, frame_idx)

    return depth_map_ZED, invalid_mask


def construct_geometry_target(x_projected, y_projected, depth_projected, H, W):
    # Round the projected coordinates to nearest integer pixels
    x_projected_int = np.round(x_projected).astype(int)
    y_projected_int = np.round(y_projected).astype(int)

    # Assign depth map with minimum depth value, Z-Buffer
    depth_map_ZED = assign_minimum_depth(x_projected_int, y_projected_int, depth_projected, H, W)

    x_projected_int = np.concatenate([np.ceil(x_projected).astype(int), np.floor(x_projected).astype(int)])
    y_projected_int = np.concatenate([np.ceil(y_projected).astype(int), np.floor(y_projected).astype(int)])
    depth_projected_cat = np.concatenate([depth_projected, depth_projected])
    depth_map_hole = assign_minimum_depth(x_projected_int, y_projected_int, depth_projected_cat, H, W)

    depth_map_ZED[np.isinf(depth_map_ZED)] = 0
    depth_map_hole[np.isinf(depth_map_hole)] = 0
    depth_map_ZED = depth_map_ZED + (np.abs(depth_map_ZED) < np.finfo(np.float32).eps) * depth_map_hole

    invalid_mask = np.abs(depth_map_ZED) < np.finfo(np.float32).eps

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
    invalid_mask_dense = np.abs(img_depth_remap_dense) < np.finfo(np.float32).eps
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



def build_invalid_areas(depth_ZED, depth_L515, img_ZED, K_L515, dist_L515, K_ZED, dist_ZED, R, T, depth_scale,
                         args=None, frame_idx=None):
    """
    Remap the depth map from ZED left image to L515 image, and detect the points located at the invalid areas in L515.
    
    Parameters:
        depth_L515 (np.array): The depth map from the L515 (height x width).
        depth_ZED (np.array): The depth map from the ZED (Height x Width).
        K_L515 (np.array): Intrinsic matrix for L515.
        dist_L515 (np.array): Distortion coefficients for L515.
        K_ZED (np.array): Intrinsic matrix for ZED's left camera.
        dist_ZED (np.array): Distortion coefficients for ZED's left camera.
        R (np.array): Rotation matrix from L515 to ZED.
        T (np.array): Translation vector from L515 to ZED.
        
    Returns:
        np.array: Invalid mask on ZED's left image.
    """
    # Rescale to meters
    depth_ZED = depth_ZED * depth_scale
    depth_L515 = depth_L515 * depth_scale

    h, w = depth_L515.shape
    H, W = depth_ZED.shape
    
    # Create mesh grid for pixel coordinates
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    
    # Flatten the pixel coordinates
    x_flat = x.flatten()
    y_flat = y.flatten()
    depth_ZED_flat = depth_ZED.flatten()

    # 1. Project Depth Points to 3D (in ZED camera coordinate system)
    # Convert pixel coordinates to normalized camera coordinates (in ZED camera frame)
    fx_ZED, fy_ZED = K_ZED[0, 0], K_ZED[1, 1]
    cx_ZED, cy_ZED = K_ZED[0, 2], K_ZED[1, 2]
    
    X_ZED = (x_flat - cx_ZED) * depth_ZED_flat / fx_ZED
    Y_ZED = (y_flat - cy_ZED) * depth_ZED_flat / fy_ZED
    Z_ZED = depth_ZED_flat  # Depth values are in meters

    # Stack the 3D points (in ZED camera coordinate system)
    points_3D_ZED = np.vstack((X_ZED, Y_ZED, Z_ZED))

    # 2. Transform 3D Points to L515's Coordinate System
    # Apply the extrinsic transformation: ZED = R * L515 + T -> L515 = R^{-1}(ZED - T)
    if len(T.shape)==1:
        T = T[:, np.newaxis]
    points_3D_L515 = np.linalg.inv(R) @ (points_3D_ZED - T)

    # 3. Project 3D Points onto L515's Left Image
    fx_L515, fy_L515 = K_L515[0, 0], K_L515[1, 1]
    cx_L515, cy_L515 = K_L515[0, 2], K_L515[1, 2]
    
    # Project the 3D points back onto the L515 image plane
    x_projected = (fx_L515 * points_3D_L515[0] / points_3D_L515[2]) + cx_L515
    y_projected = (fy_L515 * points_3D_L515[1] / points_3D_L515[2]) + cy_L515

    # 4. Detect the points located at the invalid areas in L515
    x_projected_int = np.round(x_projected).astype(int)
    y_projected_int = np.round(y_projected).astype(int)
    inside_img_mask = (x_projected_int >= 0) & (x_projected_int < w) & (y_projected_int >= 0) & (y_projected_int < h)

    chs_x_projected_int, chs_y_projected_int = x_projected_int[inside_img_mask], y_projected_int[inside_img_mask]
    chs_x_flat, chs_y_flat = x_flat[inside_img_mask], y_flat[inside_img_mask]

    depth_ZED_final = depth_ZED.copy()
    invalid_mask = np.abs(depth_L515[chs_y_projected_int, chs_x_projected_int]) < np.finfo(np.float32).eps
    depth_ZED_final[chs_y_flat, chs_x_flat] = depth_ZED_final[chs_y_flat, chs_x_flat] * (~invalid_mask)
    invalid_mask = np.abs(depth_ZED_final) < np.finfo(np.float32).eps

    # Rescale to original scale
    depth_ZED_final = depth_ZED_final / depth_scale

    return depth_ZED_final, invalid_mask



def save_ply(rgb_image, depth_image, depth_scale, intrinsics, file_name, args=None, frame_idx=None):
    # Get the height and width of the image
    H, W, _ = rgb_image.shape
    
    # Convert depth image to real-world depth values (in meters)
    depth = depth_image * depth_scale  # Depth in meters
    
    # Create meshgrid for pixel coordinates (u, v)
    u, v = np.meshgrid(np.arange(W), np.arange(H))  # u is the column, v is the row
    
    # Create mask for valid depth values (non-zero)
    valid_mask = depth > 0
    
    # Compute 3D coordinates (X, Y, Z) using the intrinsic matrix
    # Intrinsics matrix: K = [f_x, 0, c_x; 0, f_y, c_y; 0, 0, 1]
    f_x, f_y, c_x, c_y = intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]
    
    # Transform pixel coordinates (u, v) into normalized camera coordinates
    x = (u - c_x) * depth / f_x
    y = (v - c_y) * depth / f_y
    z = depth
    
    # Filter out invalid depth values
    x = x[valid_mask]
    y = y[valid_mask]
    z = z[valid_mask]
    
    # Extract RGB values and filter based on valid depth mask
    R = rgb_image[:, :, 0][valid_mask]  # Red channel
    G = rgb_image[:, :, 1][valid_mask]  # Green channel
    B = rgb_image[:, :, 2][valid_mask]  # Blue channel
    
    # Combine XYZ coordinates and RGB colors into a single array
    points = np.vstack([x, y, z, R, G, B]).T
    
    # Build the save path
    scene_name = args.scene_name  # Get the scene name from arguments
    root = args.root
    if type(frame_idx) is int:
        sv_dir = os.path.join(root, scene_name, f"{frame_idx:04d}")
    elif type(frame_idx) is str:
        sv_dir = os.path.join(root, scene_name, f"{frame_idx}")
    else:
        raise Exception(f"No support for such type of frame_idx: {type(frame_idx)}")
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)
    sv_path = os.path.join(sv_dir, file_name)

    # Write point cloud to .ply format using numpy.savetxt (faster than looping)
    header = (
        f"ply\n"
        f"format ascii 1.0\n"
        f"element vertex {points.shape[0]}\n"
        f"property float x\n"
        f"property float y\n"
        f"property float z\n"
        f"property uchar red\n"
        f"property uchar green\n"
        f"property uchar blue\n"
        f"end_header\n"
    )
    
    # Open the file and write the header
    with open(sv_path, 'w') as f:
        f.write(header)
        # Write all points at once using numpy.savetxt
        np.savetxt(f, points, fmt='%f %f %f %d %d %d')
    print(f"Saved {sv_path}")