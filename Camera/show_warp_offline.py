import os
import cv2
import yaml
import argparse
import numpy as np
import pyzed.sl as sl
import pyrealsense2 as rs

from PIL import Image
from datetime import datetime
from utils import get_image_paths, stack_images_hor, stack_images_ver, colorize_depth_rgb
from utils import save_data, save_ply
from utils import remap_depth_to_zed, repair_depth
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
        img_depth_remap, invalid_mask_remap = remap_depth_to_zed(img_depth*depth_scale, img_ZED, 
                                                                 K_L515, dist_L515, K_ZED, dist_ZED, R, T, 
                                                                 args=args, frame_idx=frame_idx)
        img_depth_remap = img_depth_remap / depth_scale
        # print(f"{scenename}-{frame_idx}\r\n" + \
        #       f"img_L515: {img_L515.shape}, img_ZED: {img_ZED.shape}, " + \
        #       f"img_depth: {img_depth.shape}, img_depth_remap: {img_depth_remap.shape} \r\n" + \
        #       f"img_depth max: {img_depth.max()}, img_depth min: {img_depth.min()} ")

        # Desitify the aligned depth map
        img_depth_remap_repair, invalid_mask_remap_mask = repair_depth(img_depth_remap, invalid_mask_remap, img_ZED, args=args, frame_idx=frame_idx)
        img_depth_remap_repair = img_depth_remap_repair.astype(np.uint16)


        # Visualization
        img_L515_resized = cv2.resize(img_L515, (0, 0), fx=0.25, fy=0.25)
        img_ZED_resized = cv2.resize(img_ZED, (0, 0), fx=0.25, fy=0.25)
        vis_img = stack_images_hor(img_L515_resized, img_ZED_resized)

        depth_colormap = colorize_depth_rgb(img_depth, img_L515)
        depth_remap_colormap = colorize_depth_rgb(img_depth_remap, img_ZED)
        depth_colormap_resized = cv2.resize(depth_colormap, (0, 0), fx=0.25, fy=0.25)
        depth_remap_colormap_resized = cv2.resize(depth_remap_colormap, (0, 0), fx=0.25, fy=0.25)
        vis_depth = stack_images_hor(depth_colormap_resized, depth_remap_colormap_resized)
        vis_img = stack_images_ver(vis_img, vis_depth)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(img_depth, alpha=0.03), cv2.COLORMAP_JET)
        depth_remap_colormap = cv2.applyColorMap(cv2.convertScaleAbs(img_depth_remap, alpha=0.03), cv2.COLORMAP_JET)
        depth_colormap_resized = cv2.resize(depth_colormap, (0, 0), fx=0.25, fy=0.25)
        depth_remap_colormap_resized = cv2.resize(depth_remap_colormap, (0, 0), fx=0.25, fy=0.25)
        vis_depth = stack_images_hor(depth_colormap_resized, depth_remap_colormap_resized)
        vis_img = stack_images_ver(vis_img, vis_depth)

        invalid_mask_resized = cv2.resize((np.abs(img_depth)<np.finfo(np.float32).eps)*1.0, (0, 0), fx=0.25/2, fy=0.25/2) > 0
        invalid_mask_remap_resized = cv2.resize(invalid_mask_remap*1.0, (0, 0), fx=0.25/2, fy=0.25/2) > 0
        invalid_mask_resized = (invalid_mask_resized * 255).astype(np.uint8)
        invalid_mask_remap_resized = (invalid_mask_remap_resized * 255).astype(np.uint8)
        vis_mask = stack_images_ver(invalid_mask_resized, invalid_mask_remap_resized)
        # vis_img = stack_images_hor(vis_img, vis_mask)

        img_depth_remap_repair_colormap1 = colorize_depth_rgb(img_depth_remap_repair, img_ZED)
        img_depth_remap_repair_colormap1_resized = cv2.resize(img_depth_remap_repair_colormap1, (0, 0), fx=0.25, fy=0.25)
        img_depth_remap_repair_colormap2 = cv2.applyColorMap(cv2.convertScaleAbs(img_depth_remap_repair, alpha=0.03), cv2.COLORMAP_JET)
        img_depth_remap_repair_colormap2_resized = cv2.resize(img_depth_remap_repair_colormap2, (0, 0), fx=0.25, fy=0.25)

        vis_depth = stack_images_ver(img_depth_remap_repair_colormap1_resized, img_depth_remap_repair_colormap2_resized)
        vis_depth = stack_images_ver(vis_mask, vis_depth)
        vis_img = stack_images_hor(vis_img, vis_depth)


        cv2.imshow(f"{scenename}-{frame_idx}", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # break



        # Save the final depth map for ZED left camera
        data = {
            "zed_depth_image": img_depth_remap_repair.astype(np.uint16),
        }
        save_data(args, data, frame_idx)

        # Save point cloud with RGB
        if args.sv_ply:
            save_ply(img_L515, img_depth, depth_scale, K_L515, 
                     "L515.ply", args=args, frame_idx=frame_idx)
            save_ply(img_ZED, img_depth_remap_repair, depth_scale, K_ZED, 
                     "ZED.ply", args=args, frame_idx=frame_idx)




if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RealSense Stream")
    parser.add_argument('--root', type=str, default="./Dataset", help='Dataset root')
    parser.add_argument('--scene_name', type=str, default=datetime.now().strftime('%Y%m%d_%H%M%S'), help='Scene name for saving images')
    parser.add_argument('--vis_debug', action='store_true', help='Save visualization for debug')
    parser.add_argument('--sv_ply', action='store_true', help='Save point cloud with RGB')
    args = parser.parse_args()

    args.root = "./Dataset"
    # args.scene_name = "demo-20250203_211254"
    args.scene_name = "demo-20250205_001631"

    main(args)