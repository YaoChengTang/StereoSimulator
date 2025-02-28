import os
import sys
import cv2
import shutil
import numpy as np

from calibration import ZedCalibration, L515Calibration, WarpCalibration
from utils import load_rgb_image, load_depth_image, load_mask_image
from utils import writePFM, write_meta_data

if __name__ == "__main__":
    otherdata_root = "/data2/Fooling3D/real_data/Dataset"
    sam_root = "/data2/Fooling3D/real_data/SAM"
    tar_root = "/data2/Fooling3D/real_data/testing"
    meta_root = "/data2/Fooling3D/real_data/meta"

    data = {}
    for scene in os.listdir(sam_root):
        scene_path = os.path.join(sam_root, scene)
        if not os.path.exists(scene_path) or not os.path.isdir(scene_path):
            print("Invalid scene path:", scene_path)
            continue
        
        for obj in os.listdir(scene_path):
            obj_path = os.path.join(scene_path, obj)
            if not os.path.exists(obj_path) or not os.path.isdir(obj_path):
                print("Invalid object path:", obj_path)
                continue

            # Load calibration for both cameras (Realsense and ZED)
            if not os.path.exists(os.path.join(otherdata_root, scene, obj, "ZED_calib.yaml")):
                print("Invalid ZED calibration file:", os.path.join(otherdata_root, scene, obj, "ZED_calib.yaml"))
                continue
            if not os.path.exists(os.path.join(otherdata_root, scene, obj, "L515_calib.yaml")):
                print("Invalid L515 calibration file:", os.path.join(otherdata_root, scene, obj, "L515_calib.yaml"))
                continue
            zed_calib = ZedCalibration(os.path.join(otherdata_root, scene, obj, "ZED_calib.yaml"))
            l515_calib = L515Calibration(os.path.join(otherdata_root, scene, obj, "L515_calib.yaml"))
            scale_factor = l515_calib.get_depth_scale()
            F = zed_calib.get_folcal_length()
            B = zed_calib.get_baseline()

            print("Processing:", scene, obj)
            
            camera = "zed_left_color_image"
            camera_path = os.path.join(obj_path, camera)
            if not os.path.exists(camera_path) or not os.path.isdir(camera_path):
                print("Invalid camera path:", camera_path)
                continue
            
            # Store the mask files in a dictionary
            # Key: frame_id, Value: list of mask files
            mask_dict = {}
            for mask_file in os.listdir(camera_path):
                # Only collect illusion masks
                if mask_file.find("nonillusion")!=-1:
                    continue
                
                mask_name, suffix = os.path.splitext(mask_file)
                frame_id = mask_name.split('-')[0]
                if frame_id not in mask_dict:
                    mask_dict[frame_id] = []
                mask_dict[frame_id].append(mask_file)

            cnt = 0
            for frame_id, mask_file_list in mask_dict.items():
                src_left_path  = os.path.join(otherdata_root, scene, obj, frame_id, "zed_left_color_image.png")
                src_right_path = os.path.join(otherdata_root, scene, obj, frame_id, "zed_right_color_image.png")
                src_depth_path = os.path.join(otherdata_root, scene, obj, frame_id, "zed_depth_image.png")
                src_invalid_path = os.path.join(otherdata_root, scene, obj, frame_id, "zed_depth_invalid_image.png")

                # Compute disparity image (unit: m, pixel)
                depth_image = load_depth_image(src_depth_path)
                disp_image  = B * F / (depth_image * scale_factor)

                # Load and merge all masks
                try:
                    mask = None
                    invalid_image = load_mask_image(src_invalid_path)
                    for mask_file in mask_file_list:
                        mask_image = load_mask_image(os.path.join(camera_path, mask_file))
                        mask_image = np.logical_and(invalid_image<128, mask_image>128)
                        if mask_image.sum() < 10:
                            continue

                        if mask is None:
                            mask = mask_image.copy()
                        else:
                            mask = np.logical_or(mask, mask_image)
                except Exception as e:
                    raise Exception("Error in loading and merging masks:", e, 
                                    f"scene={scene}, obj={obj}, frame_id={frame_id}",
                                    f"{camera_path}, {mask_file_list}, {invalid_image.shape}, {mask_image.shape}")
                
                if mask is None:
                    continue

                # Save the disparity image and mask image
                tar_left_dir  = os.path.join(tar_root, "left", scene, obj)
                os.makedirs(tar_left_dir, exist_ok=True)
                shutil.copy(src_left_path, os.path.join(tar_left_dir, f"frame_{frame_id}.png"))

                tar_right_dir = os.path.join(tar_root, "right", scene, obj)
                os.makedirs(tar_right_dir, exist_ok=True)
                shutil.copy(src_right_path, os.path.join(tar_right_dir, f"frame_{frame_id}.png"))

                tar_disp_dir  = os.path.join(tar_root, "disp", scene, obj)
                os.makedirs(tar_disp_dir, exist_ok=True)
                writePFM(os.path.join(tar_disp_dir, f"frame_{frame_id}.pfm"), disp_image)
                
                tar_mask_dir  = os.path.join(tar_root, "mask", scene, obj)
                os.makedirs(tar_mask_dir, exist_ok=True)
                cv2.imwrite(os.path.join(tar_mask_dir, f"frame_{frame_id}.jpg"), mask.astype(np.uint8)*255)

                data[(scene, obj, frame_id)] = {
                    "left" : os.path.join("left", scene, obj, f"frame_{frame_id}.png"),
                    "right": os.path.join("right", scene, obj, f"frame_{frame_id}.png"),
                    "disp" : os.path.join("disp", scene, obj, f"frame_{frame_id}.pfm"),
                    "mask" : os.path.join("mask", scene, obj, f"frame_{frame_id}.jpg"),
                }

                cnt +=1
            print(f"Saving to {os.path.join(tar_root, "xxx", scene, obj)} with {cnt} frames")

    write_meta_data(data, os.path.join(meta_root, "testing_enter.pkl"))
