import pyzed.sl as sl
import cv2

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.camera_fps = 30  # Set fps at 30
    # init_params.sdk_verbose = False

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera.")
        exit(1)

    # Get camera information (ZED serial number)
    cam_info = zed.get_camera_information()
    print("Camera Serial Number:", cam_info.serial_number)

    # Create image objects
    raw_image = sl.Mat()
    rectified_image = sl.Mat()

    # Set the runtime parameters (to retrieve images)
    runtime_parameters = sl.RuntimeParameters()
    # runtime_parameters.sensing_mode = sl.SENSING_MODE_STANDARD  # Can also be set to `SENSING_MODE_DEPTH`

    while True:
    # for i in range(50):
        # Grab an image from the camera
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Retrieve raw (uncalibrated) image
            zed.retrieve_image(raw_image, sl.VIEW.LEFT_UNRECTIFIED)  # Use sl.VIEW_RIGHT for right camera

            # Retrieve rectified image
            zed.retrieve_image(rectified_image, sl.VIEW.LEFT)  # For rectified image

            # Convert images to OpenCV format for display
            raw_image_cv = raw_image.get_data()
            rectified_image_cv = rectified_image.get_data()

            # Resize both images to 1/4 of their original size
            raw_resized = cv2.resize(raw_image_cv, (0, 0), fx=0.25, fy=0.25)
            rectified_resized = cv2.resize(rectified_image_cv, (0, 0), fx=0.25, fy=0.25)

            # Stack the images vertically
            stacked_image = cv2.vconcat([raw_resized, rectified_resized])

            # Display the stacked image
            cv2.imshow("Raw and Rectified Images", stacked_image)

            key = cv2.waitKey(1)
            if key == 27:  # Exit on pressing 'ESC'
                break

    # Close the camera and release resources
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
