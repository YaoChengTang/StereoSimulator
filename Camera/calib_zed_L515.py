import os
import cv2
import yaml
import argparse
import numpy as np
import pyzed.sl as sl
import pyrealsense2 as rs

from PIL import Image
from datetime import datetime
from utils import get_image_paths, stack_images_hor, stack_images_ver
from utils import remap_depth_to_zed, colorize_depth_rgb
from calibration import ZedCalibration, L515Calibration



def main(args):
    # Chessboard dimensions (adjust according to your checkerboard)
    chessboard_size = (9, 6)
    square_size = 0.021085  # For example, 21.085mm per square, adjust based on your setup

    # Prepare world coordinates for the chessboard
    obj_points = []  # 3D points in world coordinates
    img_points_L515 = []  # 2D points from left camera images
    img_points_ZED = []  # 2D points from right camera images

    # Define world coordinates for the checkerboard
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size  # in meters

    # Read the images (from Realsense and ZED cameras)
    left_img_paths, right_img_paths, rgb_img_paths, depth_img_paths = get_image_paths(root="./Dataset", scene_name="calib")
    L515_images = rgb_img_paths  # List of Realsense L515 images
    ZED_images = left_img_paths  # List of ZED camera images
    print(L515_images, ZED_images, sep="\r\n")

    # Iterate over images and find the corners
    for L515_img_path, ZED_img_path in zip(L515_images, ZED_images):
        img_L515 = cv2.imread(L515_img_path)
        img_ZED = cv2.imread(ZED_img_path)

        gray_L515 = cv2.cvtColor(img_L515, cv2.COLOR_BGR2GRAY)
        gray_ZED = cv2.cvtColor(img_ZED, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret_L515, corners_L515 = cv2.findChessboardCorners(gray_L515, chessboard_size, None)
        ret_ZED, corners_ZED = cv2.findChessboardCorners(gray_ZED, chessboard_size, None)

        if ret_L515 and ret_ZED:
            # Refine the corner positions to subpixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_L515_subpixel = cv2.cornerSubPix(gray_L515, corners_L515, (5, 5), (-1, -1), criteria)
            corners_ZED_subpixel = cv2.cornerSubPix(gray_ZED, corners_ZED, (5, 5), (-1, -1), criteria)

            # If corners are found, add to the lists
            img_points_L515.append(corners_L515_subpixel)
            img_points_ZED.append(corners_ZED_subpixel)
            obj_points.append(objp)  # Add corresponding 3D points

            # Optional: Draw corners on images and display them
            cv2.drawChessboardCorners(img_L515, chessboard_size, corners_L515, ret_L515)
            cv2.drawChessboardCorners(img_ZED, chessboard_size, corners_ZED, ret_ZED)
            img_L515_resized = cv2.resize(img_L515, (0, 0), fx=0.25, fy=0.25)
            img_ZED_resized = cv2.resize(img_ZED, (0, 0), fx=0.25, fy=0.25)
            vis_img = stack_images_hor(img_L515_resized, img_ZED_resized)
            cv2.imshow('Left & Right Image', vis_img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # _, mtx_L515, dist_L515, _, _ = cv2.calibrateCamera(obj_points, img_points_L515, gray_L515.shape[::-1], None, None)
    # _, mtx_ZED, dist_ZED, _, _ = cv2.calibrateCamera(obj_points, img_points_ZED, gray_ZED.shape[::-1], None, None)
    # print(f"mtx_L515: {mtx_L515}" + "\r\n" + f"dist_L515: {dist_L515}")
    # print(f"mtx_ZED: {mtx_ZED}" + "\r\n" + f"dist_ZED: {dist_ZED}")


    # Camera calibration (using StereoCalibrate)
    # Assuming you have the intrinsic parameters of both cameras (from Realsense and ZED)
    zed_calib = ZedCalibration(os.path.join(args.root, "calib", "Zed_calib.yaml"))
    l515_calib = L515Calibration(os.path.join(args.root, "calib", "L515_calib.yaml"))

    K_L515 = l515_calib.get_raw_intrinsic_matrix()
    K_ZED = zed_calib.get_rectified_calib()['left']['intrinsic']

    dist_L515 = l515_calib.get_raw_distortion_coefficients()
    dist_ZED = zed_calib.get_rectified_calib()['left']['dist']

    depth_scale = l515_calib.get_depth_scale()

    # print("K_L515: ", K_L515, sep="\r\n")
    # print("dist_L515: ", dist_L515, sep="\r\n")
    # print("K_ZED: ", K_ZED, sep="\r\n")
    # print("dist_ZED: ", dist_ZED, sep="\r\n")
    

    # Perform stereo calibration
    ret, K_L515_new, dist_L515_new, K_ZED_new, dist_ZED_new, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_L515, img_points_ZED, K_L515, dist_L515, K_ZED, dist_ZED,
        gray_L515.shape[::-1], criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-6),
        flags=cv2.CALIB_FIX_INTRINSIC)

    print("Rotation Matrix:", R, sep="\r\n")
    print("Translation Vector:", T, sep="\r\n")
    # print("K_L515_new: ", K_L515_new, sep="\r\n")
    # print("dist_L515_new: ", dist_L515_new, sep="\r\n")
    # print("K_ZED_new: ", K_ZED_new, sep="\r\n")
    # print("dist_ZED_new: ", dist_ZED_new, sep="\r\n")

    data = {
        'R': R.tolist(),
        'T': T.tolist(),
        'E': E.tolist(),
        'F': F.tolist(),
        
    }

    # Save the data to YAML
    with open(os.path.join("./Dataset", "./calib", 'L515_ZEDleft.yaml'), 'w') as file:
        yaml.dump(data, file, default_flow_style=False)



    # # Use the stereo calibration results to rectify images
    # R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(K_L515_new, dist_L515_new, K_ZED_new, dist_ZED_new, gray_L515.shape[::-1], R, T)
    # print("R1:", R1, sep="\r\n")
    # print("R2:", R2, sep="\r\n")
    # print("P1:", P1, sep="\r\n")
    # print("P2:", P2, sep="\r\n")

    # # Compute the remapping matrices for rectification
    # map1_L515, map2_L515 = cv2.initUndistortRectifyMap(K_L515_new, dist_L515_new, R1, P1, gray_L515.shape[::-1], cv2.CV_32FC1)
    # map1_ZED, map2_ZED = cv2.initUndistortRectifyMap(K_ZED_new, dist_ZED_new, R2, P2, gray_L515.shape[::-1], cv2.CV_32FC1)




if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RealSense Stream")
    parser.add_argument('--root', type=str, default="./Dataset", help='Dataset root')
    parser.add_argument('--scene_name', type=str, default=datetime.now().strftime('%Y%m%d_%H%M%S'), help='Scene name for saving images')
    args = parser.parse_args()

    main(args)