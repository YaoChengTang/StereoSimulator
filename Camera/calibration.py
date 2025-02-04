import yaml
import numpy as np



class L515Calibration:
    def __init__(self, calib_file):
        self.raw_calib = {'raw': {}}
        self.depth_info = {}
        self._load_calib(calib_file)
    
    def _load_calib(self, calib_file):
        with open(calib_file, 'r') as f:
            calib_data = yaml.safe_load(f)
        
        # Extract RGB camera parameters from the file
        rgb_camera = calib_data['RGB Camera']
        self.raw_calib['raw']['fx'], self.raw_calib['raw']['fy'] = rgb_camera['Focal Length (fx, fy)']
        self.raw_calib['raw']['cx'], self.raw_calib['raw']['cy'] = rgb_camera['Principal Point (cx, cy)']
        self.raw_calib['raw']['dist'] = np.array(rgb_camera['Distortion Coefficients'])
        
        # Create intrinsic matrix for RAW camera
        self.raw_calib['raw']['intrinsic_matrix'] = np.array([
            [self.raw_calib['raw']['fx'], 0, self.raw_calib['raw']['cx']],
            [0, self.raw_calib['raw']['fy'], self.raw_calib['raw']['cy']],
            [0, 0, 1]
        ])

        self.depth_info["depth_scale"] = calib_data['Depth Sensor']['Depth Scale']
    
    def get_raw_intrinsic_matrix(self):
        return self.raw_calib['raw']['intrinsic_matrix']
    
    def get_raw_distortion_coefficients(self):
        return self.raw_calib['raw']['dist']
    
    def get_raw_parameters(self):
        return self.raw_calib['raw']

    def get_depth_scale(self):
        return self.depth_info["depth_scale"]



class ZedCalibration:
    def __init__(self, calib_file):
        self.raw_calib = {'left': {}, 'right': {}}
        self.rectified_calib = {'left': {}, 'right': {}, 'R': {}, 'T': {}}
        self._load_calib(calib_file)
    
    def _load_calib(self, calib_file):
        with open(calib_file, 'r') as f:
            calib_data = yaml.safe_load(f)
        
        # Extract RAW parameters
        self._load_raw_calib(calib_data['RAW'])
        
        # Extract RECTIFIED parameters
        self._load_rectified_calib(calib_data['RECTIFIED'])

    def _load_raw_calib(self, raw_data):
        # Left Camera RAW
        raw_left_camera = raw_data['Left RGB Camera']
        self.raw_calib['left']['fx'], self.raw_calib['left']['fy'] = raw_left_camera['Focal Length (fx, fy)']
        self.raw_calib['left']['cx'], self.raw_calib['left']['cy'] = raw_left_camera['Principal Point (cx, cy)']
        self.raw_calib['left']['dist'] = np.array(raw_left_camera['Distortion Coefficients'])
        
        # Right Camera RAW
        raw_right_camera = raw_data['Right RGB Camera']
        self.raw_calib['right']['fx'], self.raw_calib['right']['fy'] = raw_right_camera['Focal Length (fx, fy)']
        self.raw_calib['right']['cx'], self.raw_calib['right']['cy'] = raw_right_camera['Principal Point (cx, cy)']
        self.raw_calib['right']['dist'] = np.array(raw_right_camera['Distortion Coefficients'])
        
        # Create intrinsic matrices for RAW cameras
        self.raw_calib['left']['intrinsic'] = np.array([
            [self.raw_calib['left']['fx'], 0, self.raw_calib['left']['cx']],
            [0, self.raw_calib['left']['fy'], self.raw_calib['left']['cy']],
            [0, 0, 1]
        ])
        
        self.raw_calib['right']['intrinsic'] = np.array([
            [self.raw_calib['right']['fx'], 0, self.raw_calib['right']['cx']],
            [0, self.raw_calib['right']['fy'], self.raw_calib['right']['cy']],
            [0, 0, 1]
        ])
    
    def _load_rectified_calib(self, rectified_data):
        # Left Camera RECTIFIED
        rectified_left_camera = rectified_data['Left RGB Camera']
        self.rectified_calib['left']['fx'], self.rectified_calib['left']['fy'] = rectified_left_camera['Focal Length (fx, fy)']
        self.rectified_calib['left']['cx'], self.rectified_calib['left']['cy'] = rectified_left_camera['Principal Point (cx, cy)']
        self.rectified_calib['left']['dist'] = np.array(rectified_left_camera['Distortion Coefficients'])
        
        # Right Camera RECTIFIED
        rectified_right_camera = rectified_data['Right RGB Camera']
        self.rectified_calib['right']['fx'], self.rectified_calib['right']['fy'] = rectified_right_camera['Focal Length (fx, fy)']
        self.rectified_calib['right']['cx'], self.rectified_calib['right']['cy'] = rectified_right_camera['Principal Point (cx, cy)']
        self.rectified_calib['right']['dist'] = np.array(rectified_right_camera['Distortion Coefficients'])
        
        # Create intrinsic matrices for RECTIFIED cameras
        self.rectified_calib['left']['intrinsic'] = np.array([
            [self.rectified_calib['left']['fx'], 0, self.rectified_calib['left']['cx']],
            [0, self.rectified_calib['left']['fy'], self.rectified_calib['left']['cy']],
            [0, 0, 1]
        ])
        
        self.rectified_calib['right']['intrinsic'] = np.array([
            [self.rectified_calib['right']['fx'], 0, self.rectified_calib['right']['cx']],
            [0, self.rectified_calib['right']['fy'], self.rectified_calib['right']['cy']],
            [0, 0, 1]
        ])

        # Extract rotation (R) and translation (T) matrices for RECTIFIED cameras
        self._load_rotation_and_translation(rectified_data)

    def _load_rotation_and_translation(self, rectified_data):
        R_left = rectified_data['R']  # [tilt, convergence, roll]
        T_left = rectified_data['T']  # [Tx, Ty, Tz]
        self.rectified_calib['R'] = self._rotation_from_euler_angles(R_left)
        self.rectified_calib['T'] = np.array(T_left)

    def _rotation_from_euler_angles(self, angles):
        """
        Converts tilt, convergence, and roll (in degrees) to a 3x3 rotation matrix.
        The angles are assumed to be in the order: [tilt, convergence, roll].
        """
        tilt, convergence, roll = np.deg2rad(angles)  # Convert to radians
        
        # Rotation matrix for tilt, convergence, and roll
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(tilt), -np.sin(tilt)],
            [0, np.sin(tilt), np.cos(tilt)]
        ])
        
        Ry = np.array([
            [np.cos(convergence), 0, np.sin(convergence)],
            [0, 1, 0],
            [-np.sin(convergence), 0, np.cos(convergence)]
        ])
        
        Rz = np.array([
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1]
        ])
        
        # Total rotation matrix (R = Rx * Ry * Rz)
        R = Rz @ Ry @ Rx
        return R
    
    def get_raw_calib(self):
        return self.raw_calib
    
    def get_rectified_calib(self):
        return self.rectified_calib



class WarpCalibration:
    def __init__(self, calib_file):
        self.calib = {'R': {}, 'T': {}, 'E': {}, 'F': {}}
        self._load_calib(calib_file)
    
    def _load_calib(self, calib_file):
        with open(calib_file, 'r') as f:
            calib_data = yaml.safe_load(f)
        
        self.calib['R'] = np.array(calib_data['R'])
        self.calib['T'] = np.array(calib_data['T'])
        self.calib['E'] = np.array(calib_data['E'])
        self.calib['F'] = np.array(calib_data['F'])

    def get_rotation(self):
        return self.calib['R']

    def get_translation(self):
        return self.calib['T']

    def get_essential(self):
        return self.calib['E']

    def get_fundamental(self):
        return self.calib['F']



if __name__ == "__main__":
    # Example usage
    calib = ZedCalibration('./Dataset/calib/Zed_calib.yaml')

    print("\r\n-", "-"*25, "ZED", "-"*25, "-")
    # RAW calibration data
    print("RAW Left Camera Intrinsic Matrix:")
    print(calib.get_raw_calib()['left']['intrinsic'])
    print("RAW Right Camera Intrinsic Matrix:")
    print(calib.get_raw_calib()['right']['intrinsic'])

    # RECTIFIED calibration data
    print("RECTIFIED Left Camera Intrinsic Matrix:")
    print(calib.get_rectified_calib()['left']['intrinsic'])
    print("RECTIFIED Right Camera Intrinsic Matrix:")
    print(calib.get_rectified_calib()['right']['intrinsic'])


    # Example usage
    calib = L515Calibration('./Dataset/calib/L515_calib.yaml')

    print("\r\n-", "-"*25, "REALSENSE L515", "-"*25, "-")
    # Get raw intrinsic matrix and distortion coefficients
    print("Raw Intrinsic Matrix:")
    print(calib.get_raw_intrinsic_matrix())
    print("Raw Distortion Coefficients:")
    print(calib.get_raw_distortion_coefficients())

    # Get all raw parameters in dictionary form
    print("All Raw Calibration Parameters:")
    print(calib.get_raw_parameters())
