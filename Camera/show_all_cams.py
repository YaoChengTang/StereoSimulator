import os
import cv2
import yaml
import argparse
import numpy as np
import pyzed.sl as sl
import pyrealsense2 as rs

from datetime import datetime



def init_paths(args):
    # Create scene directory if it doesn't exist
    scene_name = args.scene_name  # Get the scene name from arguments
    root = args.root
    sv_dir = os.path.join(root, scene_name)
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)

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


def stack_images_centered(stacked_image, images):
    h1, w1 = stacked_image.shape[:2]
    h2, w2 = images.shape[:2]
    
    max_height = max(h1, h2)
    total_width = w1 + w2
    
    result = np.zeros((max_height, total_width, 3), dtype=np.uint8)
    
    result[(max_height-h1)//2:(max_height-h1)//2 + h1, :w1] = stacked_image
    result[(max_height-h2)//2:(max_height-h2)//2 + h2, w1:w1+w2] = images
    
    return result



def save_data(args, data, idx):
    scene_name = args.scene_name  # Get the scene name from arguments
    root = args.root
    sv_dir = os.path.join(root, scene_name, f"{idx:04d}")
    if not os.path.exists(sv_dir):
        os.makedirs(sv_dir)
    
    keys = []
    for key, value in data.items():
        filename = os.path.join(sv_dir, f"{key}.png")
        cv2.imwrite(filename, value)
        keys.append(key)
    print(f"Saved {keys} into {sv_dir}.")


def main(args):
    init_paths(args)
    pipeline, align, depth_scale = init_L515(args)
    zed = init_zed(args)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 6 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create image objects
    left_raw_image = sl.Mat()
    left_rectified_image = sl.Mat()
    right_raw_image = sl.Mat()
    right_rectified_image = sl.Mat()

    # Set the runtime parameters (to retrieve images)
    runtime_parameters = sl.RuntimeParameters()
    # runtime_parameters.sensing_mode = sl.SENSING_MODE_STANDARD  # Can also be set to `SENSING_MODE_DEPTH`

    try:
        idx = 0
        cnt = 0
        while True:
        # for i in range(50):
            # Grab an image from the camera
            if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                # Retrieve raw (uncalibrated) image
                zed.retrieve_image(left_raw_image, sl.VIEW.LEFT_UNRECTIFIED)  # Use sl.VIEW_RIGHT for right camera
                zed.retrieve_image(right_raw_image, sl.VIEW.RIGHT_UNRECTIFIED)  # Use sl.VIEW_RIGHT for right camera

                # Retrieve rectified image
                zed.retrieve_image(left_rectified_image, sl.VIEW.LEFT)  # For rectified image
                zed.retrieve_image(right_rectified_image, sl.VIEW.RIGHT)  # For rectified image

                # Convert images to OpenCV format for display
                left_raw_image_cv = left_raw_image.get_data()[..., :3]
                left_rectified_image_cv = left_rectified_image.get_data()[..., :3]
                right_raw_image_cv = right_raw_image.get_data()[..., :3]
                right_rectified_image_cv = right_rectified_image.get_data()[..., :3]

                # Resize both images to 1/4 of their original size
                left_raw_resized = cv2.resize(left_raw_image_cv, (0, 0), fx=0.25, fy=0.25)
                left_rectified_resized = cv2.resize(left_rectified_image_cv, (0, 0), fx=0.25, fy=0.25)
                right_raw_resized = cv2.resize(right_raw_image_cv, (0, 0), fx=0.25, fy=0.25)
                right_rectified_resized = cv2.resize(right_rectified_image_cv, (0, 0), fx=0.25, fy=0.25)

                # Stack the images vertically
                left_stacked_image = cv2.vconcat([left_raw_resized, left_rectified_resized])
                right_stacked_image = cv2.vconcat([right_raw_resized, right_rectified_resized])
                stacked_image = cv2.hconcat([left_stacked_image, right_stacked_image])



                # Get frameset of color and depth
                frames = pipeline.wait_for_frames()
                # frames.get_depth_frame() is a 640x360 depth image

                # Align the depth frame to color frame
                aligned_frames = align.process(frames)

                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
                color_frame = aligned_frames.get_color_frame()

                # Validate that both frames are valid
                if not aligned_depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())


                # Remove background - Set pixels further than clipping_distance to grey
                grey_color = 153
                depth_image_3d = np.dstack((depth_image, depth_image, depth_image))  # depth image is 1 channel, color is 3 channels
                # bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
                bg_removed = color_image

                # Render images:
                #   depth align to color on left
                #   depth on right
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                # Resize bg_removed and depth_colormap to half resolution
                bg_removed_resized = cv2.resize(bg_removed, (bg_removed.shape[1] // 2, bg_removed.shape[0] // 2))
                depth_colormap_resized = cv2.resize(depth_colormap, (depth_colormap.shape[1] // 2, depth_colormap.shape[0] // 2))
                # Stack the resized images side by side
                images = np.vstack((bg_removed_resized, depth_colormap_resized))
                # images = np.hstack((bg_removed, depth_colormap))

                vis_imgs = stack_images_centered(stacked_image, images)


                # Display the stacked image
                cv2.imshow("Raw and Rectified Images, RGB and Depth Images", vis_imgs)

                cnt += 1
                if args.auto_save and cnt % 30 == 0:
                    data = {
                        "L515_color_image": color_image,
                        "L515_depth_image": depth_image,
                        "zed_left_color_image": left_rectified_image_cv,
                        "zed_right_color_image": right_rectified_image_cv,
                    }
                    save_data(args, data, idx)
                    idx += 1
                    cnt = 0

                key = cv2.waitKey(1)
                if key == ord('s'):  # Save image on pressing 's'
                    data = {
                        "L515_color_image": color_image,
                        "L515_depth_image": depth_image,
                        "zed_left_color_image": left_rectified_image_cv,
                        "zed_right_color_image": right_rectified_image_cv,
                    }
                    save_data(args, data, idx)
                    idx += 1
                
                elif key == ord('q'):  # Exit on pressing 'q'
                    print("Pressed 'q' key!")
                    break

                elif key == 27:  # Exit on pressing 'ESC'
                    break
    finally:
        pipeline.stop()

        # Close the camera and release resources
        # zed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RealSense Stream")
    parser.add_argument('--root', type=str, default="./Dataset", help='Dataset root')
    parser.add_argument('--scene_name', type=str, default="", help='Scene name for saving images')
    parser.add_argument('--auto_save', action='store_true', help='Save data automatically')
    args = parser.parse_args()
    args.scene_name = f"{args.scene_name}" + "-"*(len(args.scene_name)>0) + datetime.now().strftime('%Y%m%d_%H%M%S')

    main(args)