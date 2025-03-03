import cv2
import numpy as np

def load_images(warped_depth_path, zed_left_path, zed_right_path):
    warped_depth = cv2.imread(warped_depth_path, cv2.IMREAD_UNCHANGED)
    zed_left = cv2.imread(zed_left_path)
    zed_right = cv2.imread(zed_right_path)
    return warped_depth, zed_left, zed_right

def depth_to_disparity(depth, focal_length, baseline):
    disparity = (focal_length * baseline) / (depth + 1e-6)  # Avoid division by zero
    disparity[depth == 0] = 0  # Set disparity to 0 where depth is 0
    return disparity

def show_correspondence(left_image, right_image, disparity, num_points=5):
    h, w = left_image.shape[:2]
    left_image_raw = left_image.copy()
    right_image_raw = right_image.copy()

    while True:
        gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray_left, maxCorners=100, qualityLevel=0.01, minDistance=10)
        corners = np.int0(corners)

        valid_corners = [corner.ravel() for corner in corners if disparity[corner.ravel()[1], corner.ravel()[0]] != 0]
        if len(valid_corners) < num_points:
            print("Not enough valid points with non-zero disparity.")
            break

        chosen_points = np.random.choice(len(valid_corners), size=num_points, replace=False)
        chosen_points = [valid_corners[i] for i in chosen_points]

        for (x, y) in chosen_points:
            color = tuple(np.random.randint(0, 256, 3).tolist())
            cv2.circle(left_image, (x, y), 12, color, -1)  # Larger red points in left image
            corresponding_x = x - int(disparity[y, x])
            print(f"corresponding_x: {corresponding_x}, x: {x}, disparity: {disparity[y, x]}")
            if 0 <= corresponding_x < w:
                cv2.circle(right_image, (corresponding_x, y), 12, color, -1)  # Larger blue points in right image

        scale = 3
        combined_image = np.hstack((left_image, right_image))
        combined_image = cv2.resize(combined_image, (combined_image.shape[1] // scale, combined_image.shape[0] // scale))

        for (x, y) in chosen_points:
            corresponding_x = x - int(disparity[y, x])
            if 0 <= corresponding_x < w:
                cv2.line(combined_image, (x // scale, y // scale), ((corresponding_x + w) // scale, y // scale), (0, 255, 0), 2)  # Green line connecting points

        cv2.imshow('Left Image and Right Image with Correspondence', combined_image)

        key = cv2.waitKey(0)
        if key == 27:  # ESC key to exit
            break
        left_image = left_image_raw.copy()
        right_image = right_image_raw.copy()
    cv2.destroyAllWindows()

def main():
    warped_depth_path = './Dataset/demo/20250303_162929/0000/zed_depth_image.png'
    zed_left_path = './Dataset/demo/20250303_162929/0000/zed_left_color_image.png'
    zed_right_path = './Dataset/demo/20250303_162929/0000/zed_right_color_image.png'

    focal_length = 1517.052734375  # Example value, adjust accordingly
    baseline = 62.97140121459961 / 1000  # Example value in meters, adjust accordingly
    depth_scale = 0.00025

    warped_depth, zed_left, zed_right = load_images(warped_depth_path, zed_left_path, zed_right_path)
    warped_depth = warped_depth * depth_scale
    disparity = depth_to_disparity(warped_depth, focal_length, baseline)
    show_correspondence(zed_left, zed_right, disparity)

if __name__ == "__main__":
    main()
