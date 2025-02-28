import os
import cv2
import yaml
import argparse
import numpy as np
import pyzed.sl as sl
import pyrealsense2 as rs

from PIL import Image
from datetime import datetime
from utils import get_image_paths, stack_images_hor, stack_images_ver, stack_images, colorize_depth_rgb
from utils import save_data, save_ply
from utils import remap_depth_to_zed, repair_depth, build_invalid_areas, detect_noise_in_depth
from calibration import ZedCalibration, L515Calibration, WarpCalibration



def main(args):
    # Load calibration for both cameras (Realsense and ZED)
    zed_calib = ZedCalibration(os.path.join(args.root, args.scene_name, "Zed_calib.yaml"))
    l515_calib = L515Calibration(os.path.join(args.root, args.scene_name, "L515_calib.yaml"))
    warp_calib = WarpCalibration(os.path.join(args.root, "calib", "L515_ZEDleft.yaml"))

    K_L515 = l515_calib.get_raw_intrinsic_matrix()
    K_ZED = zed_calib.get_rectified_calib()['left']['intrinsic']

    dist_L515 = l515_calib.get_raw_distortion_coefficients()
    dist_ZED = zed_calib.get_rectified_calib()['left']['dist']

    depth_scale = l515_calib.get_depth_scale()

    R = warp_calib.get_rotation()
    T = warp_calib.get_translation()

    left_img_paths, right_img_paths, rgb_img_paths, depth_img_paths = get_image_paths(root=args.root, scene_name=args.scene_name)
    L515_images = rgb_img_paths  # List of Realsense L515 images
    ZED_images = left_img_paths  # List of ZED camera images
    for L515_img_path, ZED_img_path, depth_img_path in zip(L515_images, ZED_images, depth_img_paths):
        frame_idx = os.path.basename(os.path.dirname(depth_img_path))
        scenename = os.path.basename(os.path.dirname(os.path.dirname(depth_img_path)))
        img_L515 = cv2.imread(L515_img_path)
        img_ZED = cv2.imread(ZED_img_path)
        img_depth = cv2.imread(depth_img_path, cv2.IMREAD_UNCHANGED)

        # Align depth map to ZED left camera
        img_depth_remap, invalid_mask_remap = remap_depth_to_zed(img_depth, img_ZED, 
                                                                 K_L515, dist_L515, K_ZED, dist_ZED, R, T, depth_scale,  
                                                                 args=args, frame_idx=frame_idx)
        # print(f"{scenename}-{frame_idx}\r\n" + \
        #       f"img_L515: {img_L515.shape}, img_ZED: {img_ZED.shape}, " + \
        #       f"img_depth: {img_depth.shape}, img_depth_remap: {img_depth_remap.shape} \r\n" + \
        #       f"img_depth max: {img_depth.max()}, img_depth min: {img_depth.min()} ")

        # Desitify the aligned depth map
        img_depth_remap_repair, invalid_mask_remap_repair = repair_depth(img_depth_remap, invalid_mask_remap, img_ZED, args=args, frame_idx=frame_idx)
        # img_depth_remap_repair = img_depth_remap_repair.astype(np.uint16)

        # Recheck invalid areas by remapping depth from ZED to L515
        img_depth_ivlarea, invalid_mask_ivlarea = build_invalid_areas(img_depth_remap_repair, img_depth, img_ZED, 
                                                              K_L515, dist_L515, K_ZED, dist_ZED, R, T, depth_scale, 
                                                              args=args, frame_idx=frame_idx)

        # Rreduce the noise points
        img_depth_lownoise, invalid_mask_lownoise = detect_noise_in_depth(img_depth_ivlarea, img_ZED, invalid_mask_ivlarea, depth_scale,
                                                                          noise_threshold=0.03, kernel_size=5, 
                                                                          edge_threshold=3, min_region_area=100, 
                                                                          args=args, frame_idx=frame_idx)

        img_depth_zed, img_depth_invalid_zed = img_depth_lownoise, invalid_mask_lownoise

        # Visualization
        if args.vis:
            vis_scale = 0.25
            img_L515_resized = cv2.resize(img_L515, (0, 0), fx=vis_scale, fy=vis_scale)
            depth_colormap = colorize_depth_rgb(img_depth, img_L515, color_type="both")
            depth_colormap_resized = cv2.resize(depth_colormap, (0, 0), fx=vis_scale, fy=vis_scale)
            invalid_mask_resized = cv2.resize((np.abs(img_depth)<np.finfo(np.float32).eps)*255., (0, 0), fx=vis_scale, fy=vis_scale).astype(np.uint8)
            vis_L515 = stack_images([img_L515_resized, depth_colormap_resized, invalid_mask_resized], 
                                    orientation='vertical')

            img_ZED_resized = cv2.resize(img_ZED, (0, 0), fx=vis_scale*2/3, fy=vis_scale*2/3)
            depth_colormap = colorize_depth_rgb(img_depth_remap, img_ZED, color_type="both")
            depth_colormap_resized = cv2.resize(depth_colormap, (0, 0), fx=vis_scale*2/3, fy=vis_scale*2/3)
            invalid_mask_remap_resized = cv2.resize(invalid_mask_remap*255., (0, 0), fx=vis_scale*2/3, fy=vis_scale*2/3).astype(np.uint8)
            vis_ZED_remap = stack_images([img_ZED_resized, depth_colormap_resized, invalid_mask_remap_resized], 
                                         orientation='vertical')

            depth_colormap_repair = colorize_depth_rgb(img_depth_remap_repair, img_ZED, color_type="both")
            depth_colormap_repair_resized = cv2.resize(depth_colormap_repair, (0, 0), fx=vis_scale*2/3, fy=vis_scale*2/3)
            invalid_mask_repair_resized = cv2.resize(invalid_mask_remap_repair*255., (0, 0), fx=vis_scale*2/3, fy=vis_scale*2/3).astype(np.uint8)
            vis_ZED_repair = stack_images([None, depth_colormap_repair_resized, invalid_mask_repair_resized], 
                                          orientation='vertical', placeholder_size=img_ZED_resized.shape[0])

            depth_colormap_ivlarea = colorize_depth_rgb(img_depth_ivlarea, img_ZED, color_type="both")
            depth_colormap_ivlarea_resized = cv2.resize(depth_colormap_ivlarea, (0, 0), fx=vis_scale*2/3, fy=vis_scale*2/3)
            invalid_mask_ivlarea_resized = cv2.resize(invalid_mask_ivlarea*255., (0, 0), fx=vis_scale*2/3, fy=vis_scale*2/3).astype(np.uint8)
            vis_ZED_ivlarea = stack_images([None, depth_colormap_ivlarea_resized, invalid_mask_ivlarea_resized], 
                                           orientation='vertical', placeholder_size=img_ZED_resized.shape[0])

            vis_img = stack_images([vis_L515, vis_ZED_remap, vis_ZED_repair, vis_ZED_ivlarea], orientation='horizontal')

            # print("-"*10, invalid_mask_remap.shape, img_depth_remap.shape)

            cv2.imshow(f"{scenename}-{frame_idx}", vis_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        # Save the final depth map for ZED left camera
        data = {
            "zed_depth_image": img_depth_zed.astype(np.uint16),
            "zed_depth_invalid_image": (img_depth_invalid_zed*255).astype(np.uint8),
        }
        save_data(args, data, frame_idx)


        # Save point cloud with RGB
        if args.sv_ply:
            save_ply(img_L515, img_depth, depth_scale, K_L515, 
                     "L515.ply", args=args, frame_idx=frame_idx)
            save_ply(img_ZED, img_depth_remap_repair, depth_scale, K_ZED, 
                     "ZED.ply", args=args, frame_idx=frame_idx)

        # break



if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RealSense Stream")
    parser.add_argument('--root', type=str, default="./Dataset", help='Dataset root')
    parser.add_argument('--scene_name', type=str, default="", help='Scene name for saving images')
    parser.add_argument('--vis_debug', action='store_true', help='Save visualization for debug')
    parser.add_argument('--sv_ply', action='store_true', help='Save point cloud with RGB')
    parser.add_argument('--vis', action='store_true', help='Visualize intermediate results')
    args = parser.parse_args()

    args.root = "./Dataset"
    # args.scene_name = "demo-20250203_211254"
    # args.scene_name = "demo-20250205_001631"
    # args.scene_name = "20250224_191550"

    if len(args.scene_name)>0:
        main(args)
    else:
        for scene_name in os.listdir(args.root):
            if scene_name!="calib" and os.path.exists( os.path.join(args.root,scene_name,"L515_calib.yaml") ):
                args.scene_name = scene_name
                print(f"Processing {scene_name}")
                main(args)
                # break

    