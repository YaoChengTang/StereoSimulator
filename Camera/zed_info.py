import pyzed.sl as sl


def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.sdk_verbose = False

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Get camera information (ZED serial number)
    cam_info = zed.get_camera_information()
    print("Camera Information:")
    print("Serial Number:", cam_info.serial_number)
    print("Model:", cam_info.camera_model)
    print("Camera Resolution:", cam_info.camera_resolution.height, cam_info.camera_resolution.width)
    print("Firmware Version:", cam_info.camera_firmware_version)

    # Get and print calibration parameters
    calibration_params = cam_info.calibration_parameters
    print("\nCalibration Parameters:")
    print("Camera Intrinsics:")
    print("  - Fx:", calibration_params.left_cam.fx)
    print("  - Fy:", calibration_params.left_cam.fy)
    print("  - Cx:", calibration_params.left_cam.cx)
    print("  - Cy:", calibration_params.left_cam.cy)

    print("\nDistortion Coefficients:")
    print("  - K1:", calibration_params.left_cam.disto[0])
    print("  - K2:", calibration_params.left_cam.disto[1])
    print("  - P1:", calibration_params.left_cam.disto[2])
    print("  - P2:", calibration_params.left_cam.disto[3])

    # Close the camera
    zed.close()


if __name__ == "__main__":
    main()
