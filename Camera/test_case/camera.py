import os
import cv2
import yaml
import argparse
import numpy as np
import pyzed.sl as sl
import pyrealsense2 as rs

from datetime import datetime


def init_L515(args, enable_rgb=True, enable_depth=True): 
    # Get the scene name from arguments
    scene_name = args.scene_name
    root = args.root

    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    if enable_depth: 
        config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    if enable_rgb: 
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Get intrinsics of RGB camera
    color_stream = pipeline.get_active_profile().get_stream(rs.stream.color)
    intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy
    height = intrinsics.height
    width  = intrinsics.width

    print("L515 RGB Camera Focal Length (fx, fy):", fx, fy)
    print("L515 RGB Camera Principal Point (cx, cy):", cx, cy)
    print("L515 RGB Camera Resolution (height, width):", height, width)

    # Get distortion parameters
    distortion = intrinsics.coeffs
    print("L515 RGB Camera Distortion Coefficients:", distortion)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("L515 Depth Scale is: ", depth_scale)

    data = {
        'RGB Camera': {
            'Focal Length (fx, fy)': [fx, fy],
            'Principal Point (cx, cy)': [cx, cy],
            'Resolution (height, width)': [height, width],
            'Distortion Coefficients': distortion,
        },
        'Depth Sensor': {
            'Depth Scale': depth_scale
        }
    }

    # Save the data to YAML
    with open(os.path.join(root, scene_name, 'L515_calib.yaml'), 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    return pipeline, align, depth_scale


def init_zed(args):
    # Get the scene name from arguments
    scene_name = args.scene_name
    root = args.root

    # Create a Camera object
    zed = sl.Camera()

    # Create InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.camera_fps = 30  # Set fps at 30
    # init_params.sdk_verbose = False
    # init_params.coordinate_units = sl.UNIT.MILLIMETER

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera.")
        exit(1)

    # Get camera information (ZED serial number)
    cam_info = zed.get_camera_information()
    # print("Camera Serial Number:", cam_info.serial_number)

    raw_calib_info = cam_info.camera_configuration.calibration_parameters_raw
    rect_calib_info = cam_info.camera_configuration.calibration_parameters
    
    data = {
        'RAW': {
            'Left RGB Camera': {
                'Focal Length (fx, fy)': [raw_calib_info.left_cam.fx, raw_calib_info.left_cam.fy],
                'Principal Point (cx, cy)': [raw_calib_info.left_cam.cx, raw_calib_info.left_cam.cy],
                'Resolution (height, width)': [raw_calib_info.left_cam.image_size.height, raw_calib_info.left_cam.image_size.width],
                'Distortion Coefficients': raw_calib_info.left_cam.disto.tolist(),   # [ k1, k2, p1, p2, k3 ]
            },
            'Right RGB Camera': {
                'Focal Length (fx, fy)': [raw_calib_info.right_cam.fx, raw_calib_info.right_cam.fy],
                'Principal Point (cx, cy)': [raw_calib_info.right_cam.cx, raw_calib_info.right_cam.cy],
                'Resolution (height, width)': [raw_calib_info.right_cam.image_size.height, raw_calib_info.right_cam.image_size.width],
                'Distortion Coefficients': raw_calib_info.right_cam.disto.tolist(),   # [ k1, k2, p1, p2, k3 ]
            },
        },
        'RECTIFIED': {
            'Left RGB Camera': {
                'Focal Length (fx, fy)': [rect_calib_info.left_cam.fx, rect_calib_info.left_cam.fy],
                'Principal Point (cx, cy)': [rect_calib_info.left_cam.cx, rect_calib_info.left_cam.cy],
                'Resolution (height, width)': [rect_calib_info.left_cam.image_size.height, rect_calib_info.left_cam.image_size.width],
                'Distortion Coefficients': rect_calib_info.left_cam.disto.tolist(),   # [ k1, k2, p1, p2, k3 ]
            },
            'Right RGB Camera': {
                'Focal Length (fx, fy)': [rect_calib_info.right_cam.fx, rect_calib_info.right_cam.fy],
                'Principal Point (cx, cy)': [rect_calib_info.right_cam.cx, rect_calib_info.right_cam.cy],
                'Resolution (height, width)': [rect_calib_info.right_cam.image_size.height, rect_calib_info.right_cam.image_size.width],
                'Distortion Coefficients': rect_calib_info.right_cam.disto.tolist(),   # [ k1, k2, p1, p2, k3 ]
            },
            "R": rect_calib_info.R.tolist(),   # Each value represents 'tilt', 'convergence' and 'roll'
            "T": rect_calib_info.T.tolist(),   # First value is baseline
        }
    }
    print("ZED Camera Calibration Parameters:")
    print(data)

    # Save the data to YAML
    with open(os.path.join(root, scene_name, 'ZED_calib.yaml'), 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

    return zed
