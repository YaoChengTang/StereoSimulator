import os
import cv2
import argparse
import numpy as np


from utils import save_ply
from calibration import ZedCalibration, L515Calibration, WarpCalibration



def load_images(warped_depth_path, zed_left_path, zed_right_path):
    warped_depth = cv2.imread(warped_depth_path, cv2.IMREAD_UNCHANGED)
    zed_left = cv2.imread(zed_left_path)
    zed_right = cv2.imread(zed_right_path)
    return warped_depth, zed_left, zed_right

def depth_to_disparity(depth, focal_length, baseline):
    disparity = (focal_length * baseline) / (depth + 1e-6)  # Avoid division by zero
    disparity[depth == 0] = 0  # Set disparity to 0 where depth is 0
    return disparity

def distance_to_disparity(distance, baseline, f_x, f_y, c_x, c_y, c_z=1):
    h, w = distance.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - c_x) / f_x
    y = (v - c_y) / f_y
    z = distance * c_z / np.sqrt(x**2 + y**2 + c_z**2)

    stat = (c_z / np.sqrt(x**2 + y**2 + c_z**2))[distance!=0].flatten()
    print("-"*10, stat.mean(), stat.min(), stat.max(), np.median(stat))
    stat_bins = np.linspace(stat.min(), stat.max(), 11)
    stat_hist, _ = np.histogram(stat, bins=stat_bins)
    for i in range(len(stat_bins) - 1):
        print(f"Range {stat_bins[i]:.6f} - {stat_bins[i+1]:.6f}: {stat_hist[i]} points")
    
    # colored_stat = cv2.applyColorMap(((c_z / np.sqrt(x**2 + y**2 + c_z**2)) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    # cv2.imshow('c_z / sqrt(x^2 + y^2 + c_z^2)', colored_stat)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    focal_length = (f_x + f_y) / 2
    disparity = (focal_length * baseline) / (z + 1e-6)  # Avoid division by zero
    disparity[z == 0] = 0  # Set disparity to 0 where depth is 0

    return disparity

def show_correspondence(left_image, right_image, disparity, num_points=5, font_size=5, resize_scale=1/2.5):
    h, w = left_image.shape[:2]
    left_image_raw = left_image.copy()
    right_image_raw = right_image.copy()

    while True:
        gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray_left, maxCorners=1000, qualityLevel=0.1, minDistance=10)
        corners = np.int0(corners)

        valid_corners = [corner.ravel() for corner in corners if disparity[corner.ravel()[1], corner.ravel()[0]] != 0]
        if len(valid_corners) < num_points:
            print("Not enough valid points with non-zero disparity.")
            break

        chosen_points = np.random.choice(len(valid_corners), size=num_points, replace=False)
        chosen_points = [valid_corners[i] for i in chosen_points]
        chosen_points += [(1017, 490), (1015, 543), (1208, 528), (1089, 689)]

        color_list = []
        for (x, y) in chosen_points:
            color = tuple(np.random.randint(0, 256, 3).tolist())
            color_list.append(color)
            cv2.circle(left_image, (x, y), font_size, color, -1)  # Larger red points in left image
            corresponding_x = x - int(disparity[y, x])
            print(f"corresponding_x: {corresponding_x}, x: {x}, disparity: {disparity[y, x]}")
            if 0 <= corresponding_x < w:
                cv2.circle(right_image, (corresponding_x, y), font_size, color, -1)  # Larger blue points in right image

        cv2.imwrite(f'./Dataset/{video_name}/0000/left_image_with_points.png', left_image)
        cv2.imwrite(f'./Dataset/{video_name}/0000/right_image_with_points.png', right_image)
        combined_image = np.vstack((left_image, right_image))
        combined_image = cv2.resize(combined_image, (0, 0), fx=resize_scale, fy=resize_scale)

        # for (x, y), color in zip(chosen_points, color_list):
        #     corresponding_x = x - int(disparity[y, x])
        #     if 0 <= corresponding_x < w:
        #         cv2.line(combined_image, fx=resize_scale, fy=resize_scale, color, 1)  # Green line connecting points

        cv2.imshow('Left Image and Right Image with Correspondence', combined_image)

        key = cv2.waitKey(0)
        if key == 27:  # ESC key to exit
            break
        left_image = left_image_raw.copy()
        right_image = right_image_raw.copy()
    cv2.destroyAllWindows()

video_name = "./Dataset/real_pose1-20250303_225329"
def main(args):
    warped_depth_path = f'{video_name}/0000/zed_depth_image.png'
    zed_left_path = f'{video_name}/0000/zed_left_color_image.png'
    zed_right_path = f'{video_name}/0000/zed_right_color_image.png'
    zed_calib = ZedCalibration(os.path.join(video_name, "ZED_calib.yaml"))
    K_ZED = zed_calib.get_rectified_calib()['left']['intrinsic']

    focal_length = 1517.052734375  # Example value, adjust accordingly
    baseline = 62.97140121459961 / 1000  # Example value in meters, adjust accordingly
    depth_scale = 0.00025
    c_x, c_y = 939.1302490234375, 550.5545654296875

    warped_depth, zed_left, zed_right = load_images(warped_depth_path, zed_left_path, zed_right_path)
    warped_depth = warped_depth * depth_scale
    disparity = depth_to_disparity(warped_depth, focal_length, baseline)
    # disparity = distance_to_disparity(warped_depth, baseline, focal_length, focal_length, c_x, c_y, c_z=1)
    show_correspondence(zed_left, zed_right, disparity)
    disparity_scaled = (disparity * 32).astype(np.uint16)
    cv2.imwrite(f'{video_name}/0000/disparity_image.png', disparity_scaled)

    print("-"*10, video_name.split("/"))
    args.root = os.path.join(*(video_name.split("/")[:2]))
    args.scene_name = os.path.join(*(video_name.split("/")[2:]))
    frame_idx = zed_left_path.split("/")[-2]
    save_ply(zed_left, warped_depth/depth_scale, depth_scale, K_ZED, 
             f'zed.ply', args=args, frame_idx=frame_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RealSense Stream")
    parser.add_argument('--root', type=str, default="./Dataset", help='Dataset root')
    parser.add_argument('--scene_name', type=str, default="", help='Scene name for saving images')
    parser.add_argument('--auto_save', action='store_true', help='Save data automatically')
    args = parser.parse_args()

    main(args)
