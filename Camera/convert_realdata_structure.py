import os
import shutil

def convert_dataset_structure(root_dir, new_root_dir):
    # Traverse through the scenes
    for scene in os.listdir(root_dir):
        scene_path = os.path.join(root_dir, scene)
        if os.path.isdir(scene_path):
            # Create the new scene directory
            new_scene_path = os.path.join(new_root_dir, scene)
            os.makedirs(new_scene_path, exist_ok=True)

            # Traverse through the objects in each scene
            for obj in os.listdir(scene_path):
                if obj == "calib":
                    continue
                obj_path = os.path.join(scene_path, obj)
                if os.path.isdir(obj_path):
                    # Create the new object directory
                    new_obj_path = os.path.join(new_scene_path, obj)
                    os.makedirs(new_obj_path, exist_ok=True)

                    # Traverse through the shoot (timestamps) in each object
                    for shoot in os.listdir(obj_path):
                        shoot_path = os.path.join(obj_path, shoot)
                        if os.path.isdir(shoot_path):
                            # Traverse through the files in each shoot directory
                            for file_name in os.listdir(shoot_path):
                                file_path = os.path.join(shoot_path, file_name)
                                if os.path.isfile(file_path):
                                    # Get the file name without extension
                                    file_base_name, file_extension = os.path.splitext(file_name)

                                    # If the file is one of the images we are interested in (L515 or ZED left)
                                    if file_name == "L515_color_image.png" or file_name == "zed_left_color_image.png":
                                        # Create the new directory for the file based on the original filename
                                        new_file_dir = os.path.join(new_obj_path, file_base_name)
                                        os.makedirs(new_file_dir, exist_ok=True)

                                        # Copy the file to the new directory with the original filename
                                        new_file_path = os.path.join(new_file_dir, shoot + file_extension)
                                        shutil.copy(file_path, new_file_path)


def revert_dataset_structure(new_root_dir, original_root_dir):
    # Traverse through the new dataset structure (scene level)
    for scene in os.listdir(new_root_dir):
        new_scene_path = os.path.join(new_root_dir, scene)
        if os.path.isdir(new_scene_path):
            # Create the original scene directory if it doesn't exist
            original_scene_path = os.path.join(original_root_dir, scene)
            os.makedirs(original_scene_path, exist_ok=True)

            # Traverse through the objects in each scene
            for obj in os.listdir(new_scene_path):
                new_obj_path = os.path.join(new_scene_path, obj)
                if os.path.isdir(new_obj_path):
                    # Create the original object directory if it doesn't exist
                    original_obj_path = os.path.join(original_scene_path, obj)
                    os.makedirs(original_obj_path, exist_ok=True)

                    # Traverse through the L515 and ZED directories (L515_color_image and zed_left_color_image)
                    for category in os.listdir(new_obj_path):
                        new_category_path = os.path.join(new_obj_path, category)
                        if os.path.isdir(new_category_path):
                            # Traverse through the files in each category
                            for file_name in os.listdir(new_category_path):
                                file_path = os.path.join(new_category_path, file_name)
                                if os.path.isfile(file_path):
                                    # Extract the original shoot number from the filename
                                    shoot_number = file_name.split('_')[0]  # Assumes original format: shootX_L515_color_image.png
                                    original_shoot_path = os.path.join(original_obj_path, shoot_number)
                                    os.makedirs(original_shoot_path, exist_ok=True)

                                    # Rebuild the original file path
                                    original_file_path = os.path.join(original_shoot_path, file_name)
                                    
                                    # Copy the file back to the original structure
                                    shutil.copy(file_path, original_file_path)

def copy_sam2dataset(src_root, tar_root):
    for scene in os.listdir(src_root):
        src_scene_path = os.path.join(src_root, scene)
        if not os.path.isdir(src_scene_path):
            continue
        
        # Traverse through the objects in each scene
        for obj in os.listdir(src_scene_path):
            src_obj_path = os.path.join(src_scene_path, obj)
            if not os.path.isdir(src_obj_path):
                continue
            
            # Traverse through the L515 and ZED directories (L515_color_image and zed_left_color_image)
            for camera in os.listdir(src_obj_path):
                src_camera_path = os.path.join(src_obj_path, camera)
                if not os.path.isdir(src_camera_path):
                    continue
                
                # Traverse through the files in each category
                for mask_file in os.listdir(src_camera_path):
                    mask_name, suffix = os.splitext(mask_file)
                    frame_id = mask_name.split('-')[0]

                    # Create the target directory
                    tar_mask_dir = os.path.join(tar_root, scene, obj, frame_id)
                    tar_mask_name = mask_name.replace(frame_id, camera)
                    tar_mask_path = os.path.join(tar_mask_dir, tar_mask_name)

                    # Copy the file back to the original structure
                    os.makedirs(tar_mask_dir, exist_ok=True)
                    shutil.copy(os.path.join(src_camera_path, mask_file), tar_mask_path)


# Example usage
root_dir = '/data2/Fooling3D/real_data/Dataset_second/Dataset'
new_root_dir = '/data2/Fooling3D/real_data/Dataset_second/Dataset_new_structure'
convert_dataset_structure(root_dir, new_root_dir)

# src_root = '/data2/Fooling3D/real_data/SAM'
# tar_root = '/data2/Fooling3D/real_data/sam_mask'
# copy_sam2dataset(src_root, tar_root)