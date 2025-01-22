import pyrealsense2 as rs
import numpy as np
import cv2
import os
import argparse
from datetime import datetime

# Parse command line arguments
parser = argparse.ArgumentParser(description="RealSense Stream")
parser.add_argument('--scene_name', type=str, default=datetime.now().strftime('%Y%m%d_%H%M%S'), help='Scene name for saving images')
args = parser.parse_args()

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

config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 6 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Create scene directory if it doesn't exist
scene_name = args.scene_name  # Get the scene name from arguments
if not os.path.exists(scene_name):
    os.makedirs(scene_name)

# Streaming loop
cnt = 0
try:
    while True:
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

        # Save images every 30 frames
        if cnt % 30 == 0:
            color_image_filename = os.path.join(scene_name, f"color_image_{cnt // 30}.png")
            depth_image_filename = os.path.join(scene_name, f"depth_image_{cnt // 30}.png")

            # Save the color and depth images
            cv2.imwrite(color_image_filename, color_image)
            cv2.imwrite(depth_image_filename, depth_image)

            print(f"Saved {color_image_filename} and {depth_image_filename}")
        cnt += 1


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

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Align Example', images.shape[1], images.shape[0])
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
