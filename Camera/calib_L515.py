import cv2
import os
import yaml
import argparse

import numpy as np
import pyrealsense2 as rs

from datetime import datetime

from utils import get_image_paths, stack_images_hor, stack_images_ver, stack_images, colorize_depth_rgb



def estimate_extrinsics(object_points, image_points, K, dist_coeffs=None):
    """
    Estimate the camera extrinsic parameters (rotation matrix R and translation vector T)
    using the PnP algorithm.

    Parameters:
        object_points (numpy.ndarray): (N, 3) array of 3D points in world coordinates.
        image_points (numpy.ndarray): (N, 2) array of corresponding 2D points in the image.
        K (numpy.ndarray): (3, 3) camera intrinsic matrix.
        dist_coeffs (numpy.ndarray or None): (4, 1) distortion coefficients. 
                                             If None, distortion is assumed to be zero.

    Returns:
        R (numpy.ndarray): (3, 3) rotation matrix.
        T (numpy.ndarray): (3, 1) translation vector.
        success (bool): True if the PnP solution is found, False otherwise.
    """
    # Convert inputs to the appropriate type
    object_points = np.array(object_points, dtype=np.float32)
    image_points = np.array(image_points, dtype=np.float32)
    
    # Default to zero distortion if not provided
    if dist_coeffs is None:
        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
    
    # Solve PnP to get rotation and translation vectors
    success, rotation_vector, translation_vector = cv2.solvePnP(
        object_points, image_points, K, dist_coeffs
    )

    if success:
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rotation_vector)
        T = translation_vector.reshape(3, 1)  # Ensure T is (3,1)
        return R, T, True
    else:
        return None, None, False



data_path = "./calib_points.txt"
# Function to parse the calibration points from the file
def parse_calibration_points(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    src_points = []
    tar_points = []
    
    for i in range(0, len(lines), 2):
        src_line = lines[i].strip().split()[1:]  # Skip the ID
        tar_line = lines[i + 1].strip().split()[1:]  # Skip the ID
        
        try:
            src_points.append([(float(src_line[j]), float(src_line[j + 1])) for j in range(0, len(src_line), 2)])
            tar_points.append([(float(tar_line[j]), float(tar_line[j + 1]), float(tar_line[j + 2])) for j in range(0, len(tar_line), 3)])
        except Exception as err:
            print(src_line)
            print(tar_line)
            raise Exception(f"Error parsing line {i} and {i+1}: {err}")
    
    src_points_flat = [point for sublist in src_points for point in sublist]
    tar_points_flat = [point for sublist in tar_points for point in sublist]
    return np.array(src_points_flat), np.array(tar_points_flat)

# Parse the calibration points
src_points, tar_points = parse_calibration_points(data_path)
for src_point, tar_point in zip(src_points, tar_points):
    print(f"Source -> Target points: {src_point} -> {tar_point}")


R_raw = np.array([[0.999928,         0.009272,        -0.00761407],
                 [-0.00919691,       0.999909,         0.0098382],
                 [0.0077046,       -0.00976746,       0.999923]])
T_raw = np.array([[-0.000402700097765774,  0.0140136135742068,  -0.00529463961720467]]).T
H, W = 1024, 768
K = np.array([[901.75146484375, 0.0, 650.59228515625], [0.0, 902.520141601563, 367.704345703125], [0.0, 0.0, 1.0]], dtype=np.float32)
scale_factor = 0.0002500000118743628

# Convert tar_points from (u, v, z) to (x, y, z) using the intrinsic matrix K
def pixel_to_camera_coordinates(points, K, scale_factor=0.00025):
    """
    Convert pixel coordinates to camera coordinates using the intrinsic matrix.

    Parameters:
        points (numpy.ndarray): (N, 3) array of points in (u, v, z) format.
        K (numpy.ndarray): (3, 3) camera intrinsic matrix.

    Returns:
        numpy.ndarray: (N, 3) array of points in (x, y, z) format.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u, v, z = points[:, 0], points[:, 1], points[:, 2]*scale_factor

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    z = z

    return np.vstack((x, y, z)).T

# Convert tar_points to 3D space
tar_points_3d = pixel_to_camera_coordinates(tar_points, K)


R, T, succ = estimate_extrinsics(tar_points_3d, src_points, K, dist_coeffs=None)
print("RAW", R_raw, T_raw, sep="\r\n")
print("Updated", succ, R, T, sep="\r\n")


# Save H, W, K, R, T into a file
calibration_data = {
    'H': H,
    'W': W,
    'K': K.tolist(),
    'R': R.tolist(),
    'T': T.tolist()
}

with open('calibration_data.yaml', 'w') as file:
    yaml.dump(calibration_data, file)

# Load H, W, K, R, T from the file
with open('calibration_data.yaml', 'r') as file:
    loaded_data = yaml.safe_load(file)

H_loaded = loaded_data['H']
W_loaded = loaded_data['W']
K_loaded = np.array(loaded_data['K'])
R_loaded = np.array(loaded_data['R'])
T_loaded = np.array(loaded_data['T'])

print("Loaded data:")
print(f"H: {H_loaded}, W: {W_loaded}")
print(f"K: {K_loaded}")
print(f"R: {R_loaded}")
print(f"T: {T_loaded}")



depth_path = "./Dataset/Monitor/Apple/0000/L515_depth_image.png"
# Load the depth image
depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
print(depth_image[depth_image>0][:10])

# Convert depth image to 3D space
def depth_to_3d(depth_image, K, scale_factor=0.00025):
    """
    Convert a depth image to 3D space using the intrinsic matrix.

    Parameters:
        depth_image (numpy.ndarray): Depth image.
        K (numpy.ndarray): (3, 3) camera intrinsic matrix.
        scale_factor (float): Scale factor for depth values.

    Returns:
        numpy.ndarray: (H, W, 3) array of 3D points.
    """
    H, W = depth_image.shape
    i, j = np.indices((H, W))
    z = depth_image * scale_factor
    x = (j - K[0, 2]) * z / K[0, 0]
    y = (i - K[1, 2]) * z / K[1, 1]
    points_3d = np.stack((x, y, z), axis=-1)
    
    # Exclude points with depth value of 0
    valid_mask = depth_image > 0
    points_3d = points_3d[valid_mask]
    
    return points_3d

points_3d = depth_to_3d(depth_image, K, scale_factor)
print(points_3d[:10])

# Apply the extrinsic parameters to rectify the points
def rectify_points(points_3d, R, T):
    """
    Rectify 3D points using the extrinsic parameters.

    Parameters:
        points_3d (numpy.ndarray): (N, 3) array of 3D points.
        R (numpy.ndarray): (3, 3) rotation matrix.
        T (numpy.ndarray): (3, 1) translation vector.

    Returns:
        numpy.ndarray: (N, 3) array of rectified 3D points.
    """
    rectified_points = R @ points_3d.T + T
    return rectified_points.T

rectified_points = rectify_points(points_3d, R_loaded, T_loaded)
print(rectified_points[:10])

# Project the rectified points onto the image plane
def project_points(points_3d, K):
    """
    Project 3D points onto the image plane using the intrinsic matrix.

    Parameters:
        points_3d (numpy.ndarray): (N, 3) array of 3D points.
        K (numpy.ndarray): (3, 3) camera intrinsic matrix.

    Returns:
        numpy.ndarray: (N, 2) array of 2D points in the image plane.
    """
    # Normalize the points
    points_2d = points_3d[:, :2] / points_3d[:, 2:]
    # Add the homogeneous coordinate
    points_2d_homogeneous = np.hstack((points_2d, np.ones((points_2d.shape[0], 1))))
    # Project the points using the intrinsic matrix
    points_2d_projected = (K @ points_2d_homogeneous.T).T
    return points_2d_projected[:, :2]

projected_points = project_points(rectified_points, K_loaded)

# Create a new depth map from the projected points and rectified points
def create_depth_map(projected_points, rectified_points, image_shape, scale_factor=0.00025):
    """
    Create a depth map from the projected points and rectified points.

    Parameters:
        projected_points (numpy.ndarray): (N, 2) array of 2D points in the image plane.
        rectified_points (numpy.ndarray): (N, 3) array of rectified 3D points.
        image_shape (tuple): Shape of the output depth map (H, W).

    Returns:
        numpy.ndarray: Depth map.
    """
    depth_map = np.zeros(image_shape, dtype=np.float32)
    for (u, v), (x, y, z) in zip(projected_points, rectified_points):
        u, v = int(round(u)), int(round(v))
        if 0 <= u < image_shape[1] and 0 <= v < image_shape[0]:
            depth_map[v, u] = z / scale_factor
    return depth_map

# Create the new depth map
new_depth_map = create_depth_map(projected_points, rectified_points, depth_image.shape, scale_factor)
print(new_depth_map[new_depth_map>0][:10])

# Save the new depth map
new_depth_map_path = "./new_depth_map.png"
cv2.imwrite(new_depth_map_path, new_depth_map.astype(np.uint16))

# Display the new depth map
cv2.imshow('New Depth Map', new_depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
