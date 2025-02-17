import os
import sys
import cv2
import glob
import time
import pickle
import shutil
import multiprocessing

import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd

from PIL import Image
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from simple_lama_inpainting import SimpleLama

from utils import check_paths, load_meta_data, write_meta_data
from utils import load_rgb_image, load_depth_image, load_mask_image


class ImageProcess():
    def __init__(self):
        self.simple_lama = SimpleLama()

    def morphological_min(self, input_mask, kernel_size=3):
        input_mask_neg = -input_mask.float() 
        eroded_mask = -F.max_pool2d(input_mask_neg.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        return (eroded_mask > 0).squeeze().int() 

    def morphological_max(self, input_mask, kernel_size=3):
        dilated_mask = F.max_pool2d(input_mask.unsqueeze(0).unsqueeze(0).float(), kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        return (dilated_mask > 0).squeeze().int()

    def sum_pooling(self, input_mask, kernel_size=3):
        summed_mask = F.avg_pool2d(input_mask.unsqueeze(0).unsqueeze(0).float(), kernel_size=kernel_size, stride=1, padding=kernel_size//2) * kernel_size * kernel_size
        return (summed_mask > (kernel_size * kernel_size // 2)).squeeze().int() 
    
    def get_occlusion_mask(self, shifted):

        mask_up = shifted > 0
        mask_down = shifted > 0

        shifted_up = np.ceil(shifted)
        shifted_down = np.floor(shifted)

        for col in range(shifted.shape[1] - 2):
            loc = shifted[:, col:col + 1]  # keepdims
            loc_up = np.ceil(loc)
            loc_down = np.floor(loc)

            _mask_down = ((shifted_down[:, col + 2:] != loc_down) * 
                          (shifted_up[:, col + 2:] != loc_down)).min(-1)
            _mask_up   = ((shifted_down[:, col + 2:] != loc_up) * 
                          (shifted_up[:, col + 2:] != loc_up)).min(-1)

            mask_up[:, col]   = mask_up[:, col] * _mask_up
            mask_down[:, col] = mask_down[:, col] * _mask_down

        mask = mask_up + mask_down
        return mask

    # def get_occlusion_mask(self, shifted):
    #     H, W = shifted.shape

    #     # Acquire valid shift
    #     valid = shifted > 0

    #     # Build valid position
    #     shifted_up   = np.ceil(shifted).astype(np.int32)
    #     shifted_down = np.floor(shifted).astype(np.int32)
    #     print(f"shifted_up: {shifted_up.shape}, shifted_down: {shifted_down.shape}")

    #     # Compute similarity for each position
    #     shifted_up_extend   = shifted_up[..., np.newaxis]
    #     shifted_down_extend = shifted_down[..., np.newaxis]
    #     loc_up_extend       = shifted_up[:, np.newaxis, :]
    #     loc_down_extend     = shifted_down[:, np.newaxis, :]

    #     sim_down = np.all([np.equal(loc_down_extend, shifted_down_extend), 
    #                        np.equal(loc_down_extend, shifted_up_extend)], 
    #                       axis=0)  # (H, W, W)
    #     sim_up = np.all([np.equal(loc_up_extend, shifted_down_extend), 
    #                      np.equal(loc_up_extend, shifted_up_extend)], 
    #                     axis=0)  # (H, W, W)
    #     # sim_down = (loc_down_extend == shifted_down_extend) & (loc_down_extend == shifted_up_extend)  # (H,W,W)
    #     # sim_up   = (loc_up_extend == shifted_down_extend) & (loc_up_extend == shifted_up_extend)  # (H,W,W)
    #     print(f"sim_down: {sim_down.shape}, sim_up: {sim_up.shape}")

    #     # Create a mask where mask[i, j] is True if and only if j >= i + 2
    #     i_indices = np.arange(W)
    #     j_indices = np.arange(W)
    #     triangle_mask = j_indices >= (i_indices[:, None] + 2)  # (W,W)
    #     # print(f"triangle_mask: {triangle_mask.shape}")

    #     # Compute the occlusion mask
    #     mask_down = (sim_down & triangle_mask[np.newaxis]).any(axis=2) * valid # (H,W)
    #     mask_up   = (sim_up & triangle_mask[np.newaxis]).any(axis=2) * valid # (H,W)
    #     # print(f"mask_down: {mask_down.shape}, mask_up: {mask_up.shape}")

    #     # # Create the triangle mask (j >= i + 2)
    #     # i_indices = np.arange(W)
    #     # j_indices = np.arange(W)
    #     # triangle_mask = j_indices >= (i_indices[:, None])

    #     # # Compute the occlusion mask using broadcasting
    #     # # Avoid creating full H x W x W array, instead directly perform pairwise comparisons
    #     # mask_down = np.any(
    #     #     (shifted_down[:, :, np.newaxis] == shifted_down[:, np.newaxis, :]) & 
    #     #     (shifted_down[:, :, np.newaxis] == shifted_up[:, np.newaxis, :]) &
    #     #     triangle_mask[np.newaxis], axis=2
    #     # ) & valid

    #     # mask_up = np.any(
    #     #     (shifted_up[:, :, np.newaxis] == shifted_down[:, np.newaxis, :]) & 
    #     #     (shifted_up[:, :, np.newaxis] == shifted_up[:, np.newaxis, :]) &
    #     #     triangle_mask[np.newaxis], axis=2
    #     # ) & valid

    #     mask = mask_up + mask_down

    #     return mask

    # def get_occlusion_mask(self, warped_pos, disp):

    #     mask_up = shifted > 0
    #     mask_down = shifted > 0

    #     shifted_up = np.ceil(shifted)
    #     shifted_down = np.floor(shifted)

    #     for col in range(shifted.shape[1] - 2):
    #         loc = shifted[:, col:col + 1]  # keepdims
    #         loc_up = np.ceil(loc)
    #         loc_down = np.floor(loc)

    #         _mask_down = ((shifted_down[:, col + 2:] != loc_down) * 
    #                       (shifted_up[:, col + 2:] != loc_down)).min(-1)
    #         _mask_up   = ((shifted_down[:, col + 2:] != loc_up) * 
    #                       (shifted_up[:, col + 2:] != loc_up)).min(-1)

    #         mask_up[:, col]   = mask_up[:, col] * _mask_up
    #         mask_down[:, col] = mask_down[:, col] * _mask_down

    #     mask = mask_up + mask_down
    #     return mask

    
    def warp(self, left_image, disparity_map):
        H, W, C = left_image.shape
        # print(left_image)
        right_image_np = np.zeros_like(left_image)
        # mask = np.ones((H, W), dtype=np.uint8) * 255
        mask = np.zeros((H, W), dtype=np.uint8)
        xs, ys = np.meshgrid(np.arange(W), np.arange(H))
        shifted = xs - disparity_map
        mask_ = self.get_occlusion_mask(shifted)
        mask[mask_] = 255
        # for i in range(H):
        #     for j in range(W):
        #         if j - int(disparity_map[i,j]) < 0 :
        #             continue
        #         right_image_np[i,j - int(disparity_map[i,j])] = left_image[i,j]
        #         mask[i,j - int(disparity_map[i,j])] = 0
        right_image_np = cv2.cvtColor(right_image_np, cv2.COLOR_BGR2RGB)
        right_image = Image.fromarray(right_image_np)
        # right_image.save('imgR/imgR.png')
        print(mask)
        cv2.imwrite('mask/mask1.png', mask)
        exit(0)
        # mask = Image.fromarray(mask).convert('L')
        result = self.simple_lama(right_image, mask)
        # result.save('imgR/imgR_.png')
        return right_image
    
    def project_image(self, image, disp_map):
        feed_height, process_width, C = image.shape
        image = np.array(image)
        # background_image = np.array(background_image)

        # set up for projection
        warped_image = np.zeros_like(image).astype(float)
        warped_image = np.stack([warped_image] * 2, 0)
        xs, ys = np.meshgrid(np.arange(process_width), np.arange(feed_height))
        pix_locations = xs - disp_map

        # find where occlusions are, and remove from disparity map
        start_time = time.time()
        mask = self.get_occlusion_mask(pix_locations)
        occ_mask = mask.copy()
        masked_pix_locations = pix_locations * mask - process_width * (1 - mask)
        cost_time = time.time() - start_time
        print("-"*10, f"get_occlusion_mask cost time: {cost_time:.3f}, mask: {mask.shape}, {mask.min()}, {mask.max()}")

        # do projection - linear interpolate up to 1 pixel away
        weights = np.ones((2, feed_height, process_width)) * 10000

        start_time = time.time()
        for col in range(process_width - 1, -1, -1):
            loc = masked_pix_locations[:, col]
            loc_up = np.ceil(loc).astype(int)
            loc_down = np.floor(loc).astype(int)
            weight_up = loc_up - loc
            weight_down = 1 - weight_up

            mask = loc_up >= 0
            mask[mask] = \
                weights[0, np.arange(feed_height)[mask], loc_up[mask]] > weight_up[mask]
            weights[0, np.arange(feed_height)[mask], loc_up[mask]] = \
                weight_up[mask]
            warped_image[0, np.arange(feed_height)[mask], loc_up[mask]] = \
                image[:, col][mask] / 255.

            mask = loc_down >= 0
            mask[mask] = \
                weights[1, np.arange(feed_height)[mask], loc_down[mask]] > weight_down[mask]
            weights[1, np.arange(feed_height)[mask], loc_down[mask]] = weight_down[mask]
            warped_image[1, np.arange(feed_height)[mask], loc_down[mask]] = \
                image[:, col][mask] / 255.

        weights /= weights.sum(0, keepdims=True) + 1e-7  # normalise
        weights = np.expand_dims(weights, -1)
        warped_image = warped_image[0] * weights[1] + warped_image[1] * weights[0]
        warped_image *= 255.
        cost_time = time.time() - start_time
        print("-"*10, f"linear interpolate cost time: {cost_time:.3f}")

        # return warped_image, (occ_mask*255).astype(np.uint8)

        # # now fill occluded regions with random background
        # if not disable_background:
        #     warped_image[warped_image.max(-1) == 0] = background_image[warped_image.max(-1) == 0]
        
        warped_image = warped_image.astype(np.uint8)
        mask_ =  np.zeros((feed_height, process_width), dtype=np.uint8)
        mask_[warped_image.max(-1) == 0] = 1
        mask_t = torch.from_numpy(mask_).squeeze()
        
        start_time = time.time()
        mask_max = self.morphological_max(mask_t)
        mask_min = self.morphological_min(mask_max)
        mask_re = self.sum_pooling(mask_min)
        mask_re = mask_re.numpy() * 255
        warped_image[mask_re > 0] = 0
        right_image_np = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)
        right_image_np[mask_re > 0] = 0
        right_image = Image.fromarray(right_image_np)
        mask = Image.fromarray(mask_re).convert('L')
        result = self.simple_lama(right_image, mask)
        cost_time = time.time() - start_time
        print("-"*10, f"simple_lama cost time: {cost_time:.3f}", result.size)
        
        return right_image, result, mask, (occ_mask*255).astype(np.uint8)


def save_right_image(image_root, frame_path, right_image, debug_info=""):
    """
    Save generated right image in a structured directory inside image_root.

    Parameters:
    image_root (str): The root directory containing depth files.
    frame_path (str): The full path of the iamge file inside image_root.
    right_image (np.ndarray): The generated right image to be saved.
    """
    # Ensure absolute paths
    image_root = os.path.abspath(image_root)
    frame_path = os.path.abspath(frame_path)

    # Extract last folder name from image_root
    last_folder = os.path.basename(image_root)
    last_folder_parent = os.path.dirname(image_root)

    # Define new root directory with "_rect" suffix
    new_root = os.path.join(last_folder_parent, f"{last_folder}_right")

    # Compute relative path inside image_root
    relative_subpath = os.path.relpath(frame_path, image_root)

    # Construct the final save path
    sv_path = os.path.join(new_root, relative_subpath)
    sv_dir = os.path.dirname(sv_path)
    if debug_info is not None and len(debug_info)>0:
        prefix, suffix = os.path.splitext(sv_path)
        sv_path = f"{prefix}-{debug_info}{suffix}"

    # Ensure the save directory exists
    os.makedirs(sv_dir, exist_ok=True)

    # Ensure right_image is valid before saving
    if right_image is None or right_image.size == 0:
        print(f"Error: right_image is empty. Skipping {sv_path}")
        return

    # Save depth image as uint16 PNG
    if isinstance(right_image, np.ndarray):
        cv2.imwrite(sv_path, right_image)
    elif isinstance(right_image, Image.Image):
        right_image.save(sv_path)
    else:
        raise Exception(f"No support for type: {type(right_image)}")
    print(f"Saved generated right image: {sv_path}")


def determine_scaling_factor(disparity_map, image_width, threshold=0.9, epsilon=1e-6, max_iterations=100):
    """
    Determine the appropriate scaling factor to ensure that at least a certain
    percentage of points (default 90%) stay within the image width when warping.
    Uses binary search to efficiently find the scaling factor, with a limit on max iterations.
    
    Args:
        disparity_map (numpy.ndarray): The input disparity map.
        image_width (int): The width of the RGB image.
        threshold (float): The desired percentage of valid points (0 to 1, default 0.9).
        epsilon (float): The tolerance for binary search convergence.
        max_iterations (int): The maximum number of iterations for the binary search.
        
    Returns:
        scaling_factor (float): The computed scaling factor for the disparity map.
    """
    # Initialize the search range for scaling factor
    scaling_factor = image_width / disparity_map.max() / 4
    left, right = scaling_factor/3, scaling_factor
    iteration = 0
    
    while right - left > epsilon and iteration < max_iterations:
        iteration += 1
        
        # Midpoint scaling factor
        scaling_factor = (left + right) / 2.0
        
        # Scale disparity map
        scaled_disparity_map = disparity_map * scaling_factor
        
        # Create a mesh grid of pixel indices for the left image
        h, w = scaled_disparity_map.shape
        x_left, y_left = np.meshgrid(np.arange(w), np.arange(h))
        
        # Calculate the warped coordinates in the right image
        x_right = x_left - scaled_disparity_map
        
        # Count how many x_right coordinates are within valid image boundaries
        valid_points = np.sum((x_right >= 0) & (x_right < image_width))
        total_points = x_right.size
        
        # Check if the ratio of valid points meets the threshold
        valid_ratio = valid_points / total_points
        
        # Binary search decision: adjust search range based on the validity
        if valid_ratio >= threshold:
            left = scaling_factor  # Move towards the higher scaling factor
        else:
            right = scaling_factor  # Move towards the lower scaling factor

    # Return the midpoint as the optimal scaling factor
    return (left + right) / 2.0


def project_image(left_image, disp_map):
    H, W, C = left_image.shape
    right_image = np.zeros((H,W,C))

    # Compute the warped position
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u_projected = (u - disp_map).flatten()
    v_projected = v.flatten()
    u_choose    = u.flatten()

    # Filter out points that project outside the image bounds (negative or too large values)
    valid = u_projected >= 0
    u_projected = u_projected[valid]
    v_projected = v_projected[valid]
    u_choose    = u_choose[valid]

    # Compute the quantization position
    u_projected_int = np.concatenate([np.ceil(u_projected).astype(int), np.floor(u_projected).astype(int)])
    v_projected_int = np.concatenate([v_projected, v_projected])
    u_choose_int    = np.concatenate([u_choose, u_choose])

    # Filter out points that project outside the image bounds (negative or too large values)
    valid = (u_projected_int >= 0) & (u_projected_int < W)
    u_projected_int = u_projected_int[valid]
    v_projected_int = v_projected_int[valid]
    u_choose_int    = u_choose_int[valid]

    # Assign value with the maximum disp/coordinate value for each pixel using NumPy operations (no loops)
    # Create a temporary depth map with infinity (for minimum depth calculation)
    tar = -100 * np.ones((H,W))
    src = u_choose_int

    print("*"*10, u_projected_int.max(), v_projected_int.max())

    # Use np.maximum to apply the maximum disp/coordinate value for each pixel
    np.maximum.at(tar, (v_projected_int, u_projected_int), src)

    valid_mask = tar >= 0
    rows, cols = np.where(valid_mask)
    used_u = tar[(rows, cols)].astype(int)
    right_image = np.zeros_like(left_image)
    right_image[(rows, cols)] = left_image[(rows, used_u)]
    occ_mask = np.zeros((H,W))
    occ_mask[(rows, used_u)] = 1

    return right_image, ~valid_mask, (occ_mask*255).astype(np.uint8)
    

def repair_image(right_image, invalid_mask, inpainter=None):
    right_image = Image.fromarray(right_image)
    invalid_mask = Image.fromarray(invalid_mask).convert('L')
    right_image_repair = inpainter(right_image, invalid_mask)
    return np.array(right_image_repair)

# Define globally shared variables and initialization function
def init_process():
    global img_processor
    img_processor = ImageProcess()


def generate_right_image(left_image, disparity_map, image_root, frame_path, threshold=0.9, epsilon=1e-6, max_iterations=-1):
    # Determine the appropriate scaling factor to ensure that at least a certain
    # percentage of points (default 90%) stay within the image width when warping.
    start_time = time.time()
    scale_factor = determine_scaling_factor(disparity_map, left_image.shape[1], 
                                            threshold=threshold, epsilon=epsilon, max_iterations=max_iterations)
    print(f"scale_factor: {scale_factor}")
    cost_time = time.time() - start_time
    print("-"*10, f"determine_scaling_factor cost time: {cost_time:.3f}")

    # Generate the right image
    # img_processor = ImageProcess()
    global img_processor
    right_image1, right_image_fill, mask, occ_mask1 = img_processor.project_image(left_image, disparity_map * scale_factor)
    # right_image1, occ_mask1 = img_processor.project_image(left_image, disparity_map * scale_factor)

    start_time = time.time()
    right_image2, invalid_mask2, occ_mask2 = project_image(left_image, disparity_map * scale_factor)
    cost_time = time.time() - start_time
    print("-"*10, f"project_image cost time: {cost_time:.3f}")
    
    start_time = time.time()
    right_image_repair = repair_image(right_image2, invalid_mask2, inpainter=img_processor.simple_lama)
    cost_time = time.time() - start_time
    print("-"*10, f"repair_image cost time: {cost_time:.3f}", right_image_repair.shape)
    
    # Save the gnerated right image
    save_right_image(image_root, frame_path, right_image1, debug_info="raw_right")
    save_right_image(image_root, frame_path, right_image_fill, debug_info="raw_fill")
    save_right_image(image_root, frame_path, occ_mask1, debug_info="raw_occ")
    save_right_image(image_root, frame_path, right_image2, debug_info="new_right")
    save_right_image(image_root, frame_path, occ_mask2, debug_info="new_occ")
    save_right_image(image_root, frame_path, right_image_repair, debug_info="new_fill")


def process_frame(video_name, frame_name, frame_dict, area_types, image_root):
    """
    Process a single frame.
    """
    print(f"Process ID: {os.getpid()}")
    try:
        frame_path = frame_dict["image"]
        depth_path = frame_dict["depth"]
        depth_path = depth_path.replace("/depth/", "/depth_rect/")

        left_image = load_rgb_image(frame_path)
        depth_image = load_depth_image(depth_path)

        if left_image is None:
            print(f"No RGB image: {frame_path}")
            return
        
        if depth_image is None:
            print(f"No depth image: {depth_path}")
            return
        
        generate_right_image(left_image, depth_image, image_root, frame_path, threshold=0.9, epsilon=1e-6, max_iterations=1)

    
    except Exception as err:
        raise Exception(err, f"video_name: {video_name}   frame_name: {frame_name}   " + \
                             f"frame_dict: {frame_dict}   area_types:{area_types}   image_root: {image_root}")


def process_video(video_name, video_dict, area_types, image_root):
    """
    Process all frames in a single video in parallel.
    """
    # Calculate the number of processes (CPU cores - 10)
    # num_processes = max(1, os.cpu_count() - 10)  # Ensure at least 1 process is used
    num_processes = 1

    multiprocessing.set_start_method('spawn', force=True)

    # Use multiprocessing to process frames concurrently
    with multiprocessing.Pool(processes=num_processes, initializer=init_process) as pool:
        # Prepare tasks as a list of arguments for process_frame
        tasks = [
            (video_name, frame_name, frame_dict, area_types, image_root)
            for frame_name, frame_dict in video_dict.items()
        ]
        # Process frames in parallel
        # pool.starmap(process_frame, tasks)
        pool.starmap(process_frame, tasks[:3])



if __name__ == '__main__':
    image_root = "/data2/Fooling3D/video_frame_sequence"
    mask_root  = "/data2/Fooling3D/sam_mask"
    depth_root = "/data5/fooling-depth/depth"
    meta_root  = "/data2/Fooling3D/meta_data"

    # Load metadata
    area_types = ["illusion", "nonillusion"]
    data = load_meta_data(os.path.join(meta_root, "data_dict.pkl"))

    # Process each video in parallel
    start_from_video_name = None
    # start_from_video_name = "video0/the_cake_studio_shorts"
    # start_from_video_name = "video5/In_Indian_Bike_Driving_3d_Game_Nitin_Patel_shorts"
    # start_from_video_name = "video0/wall_painting_new_creative_design"
    # start_from_video_name = "video2/drawing_easiest_trick_art_easytrick_drawing"
    started = False
    for video_name, video_dict in tqdm(data.items(), desc="Processing videos"):
        # Start from last failure video
        if start_from_video_name is not None:
            if video_name==start_from_video_name:
                started = True
            if not started:
                continue
        process_video(video_name, video_dict, area_types, image_root)
        break
