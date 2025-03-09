import numpy as np
import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation
def icp_registration(source, target, max_iterations=30, threshold=0.02):
    """
    使用Open3D进行ICP点云配准
    
    参数:
    source (open3d.geometry.PointCloud): 源点云
    target (open3d.geometry.PointCloud): 目标点云
    max_iterations (int): 最大迭代次数
    threshold (float): 匹配距离阈值
    
    返回:
    open3d.registration.RegistrationResult: 包含配准结果的对象
    """
    # 检查输入有效性
    if not source.has_points() or not target.has_points():
        raise ValueError("输入点云不能为空")
    
    # 预处理步骤（可选）
    # 1. 降采样
    source = source.voxel_down_sample(voxel_size=0.005)
    target = target.voxel_down_sample(voxel_size=0.005)
    
    # 2. 估计法线（ICP的某些变种需要法线信息）
    source.estimate_normals()
    target.estimate_normals()
    
    # 执行ICP配准
    reg_result = o3d.pipelines.registration.registration_icp(
        source, 
        target, 
        threshold,
        np.identity(4),  # 初始变换矩阵（单位矩阵）
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),  # 使用点对面ICP
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
    )
    
    return reg_result

def load(depth_path):
    if depth_path.endswith('.png'):
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32)*0.00025
        if len(depth.shape) == 3:
            depth = depth[:,:,0]
        mask = (depth>0)&(depth<np.inf)
    elif depth_path.endswith('.npy'):
        depth = np.load(depth_path)
        depth = depth.astype(np.float32)
        mask = depth > 0
    return depth,mask

def depth_to_pointcloud(depth, K):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    points = np.stack([x, y, z], axis=-1)
    return points


def average_transformations(transforms, weights=None):
    # 分解旋转矩阵和平移向量
    rotations = [t[:3, :3] for t in transforms]
    translations = [t[:3, 3] for t in transforms]
    
    # 平均平移
    t_avg = np.average(translations, axis=0, weights=weights)
    
    # 将旋转矩阵转为四元数
    quats = [Rotation.from_matrix(R).as_quat() for R in rotations]
    
    # 符号对齐
    base_quat = quats[0]
    for i in range(1, len(quats)):
        if np.dot(quats[i], base_quat) < 0:
            quats[i] *= -1
    
    # 加权平均四元数
    if weights is None:
        weights = np.ones(len(quats)) / len(quats)
    q_avg = np.average(quats, axis=0, weights=weights)
    q_avg /= np.linalg.norm(q_avg)
    
    # 转为旋转矩阵
    R_avg = Rotation.from_quat(q_avg).as_matrix()
    
    # 构建平均变换矩阵
    T_avg = np.eye(4)
    T_avg[:3, :3] = R_avg
    T_avg[:3, 3] = t_avg
    return T_avg

# 示例用法
if __name__ == "__main__":
    scene_list =['0000','0001','0002','0003','0004','0005','0006','0007','0008','0009']
    transform_list = []

    for scene in scene_list:
        source_mask_path = f'/data2/TMP/SAM_Depth_zed/{scene}-01-01-illusion.jpg'
        source_mask=cv2.imread(source_mask_path, cv2.IMREAD_UNCHANGED)>0

        target_mask_path = f'/data2/TMP/SAM/scene/scene1/zed_left_color_image/{scene}-01-01-illusion.jpg'
        target_mask=cv2.imread(target_mask_path, cv2.IMREAD_UNCHANGED)
        target_mask = cv2.resize(target_mask.astype(np.uint8),(target_mask.shape[1]//2,target_mask.shape[0]//2),interpolation=cv2.INTER_NEAREST)>0

        source_path = f'/data2/TMP/Dataset/scene/scene1/{scene}/' + 'zed_depth_image.png'

        target_path = f'/data4/lzd/baseline/disp/scene1/fuse/{scene}/' + 'depth_filtered.npy'
        scale = 0.5
        source,s_mask = load(source_path)
        target,t_mask = load(target_path)
        s_mask_raw = s_mask.copy()
        t_mask_raw = t_mask.copy()
        s_mask = s_mask & source_mask
        t_mask = t_mask & target_mask

        K_zed = np.array([[1516.77, 0, 939.13],[0, 1516.77, 550.55],[0, 0, 1]])
        K_zed_scale = K_zed.copy()
        K_zed_scale[:2] = K_zed_scale[:2]*scale
        K_L515 = np.array([[901.75146484375, 0, 902.5201416015625],[0, 650.59228515625, 367.704345703125],[0, 0, 1]])
        source_np = depth_to_pointcloud(source, K_zed)
        target_np = depth_to_pointcloud(target, K_zed_scale)
        # 生成示例数据（实际使用时替换为真实点云）
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(source_np[np.tile(s_mask[:,:,None],(1,1,3))].reshape(-1,3))
        source_raw = o3d.geometry.PointCloud()
        source_raw.points = o3d.utility.Vector3dVector(source_np[np.tile(s_mask_raw[:,:,None],(1,1,3))].reshape(-1,3))
        # save source point cloud
        o3d.io.write_point_cloud(f'/data2/TMP/Dataset/scene/scene1/{scene}/' + 'source.ply', source)
        
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(target_np[np.tile(t_mask[:,:,None],(1,1,3))].reshape(-1,3))    
        # save target point cloud
        o3d.io.write_point_cloud(f'/data4/lzd/baseline/disp/scene1/fuse/{scene}/' + 'target.ply', target)
        
        # 执行ICP配准
        result = icp_registration(source, target)
        
        # 输出结果
        print("变换矩阵:")
        print(result.transformation)
        print("配准评估:", result)
        print("欧式拟合误差:", result.inlier_rmse)
        transform_list.append(result.transformation.copy())
        # ret = R.from_matrix(rot)
        # quaternion = ret.as_quat()
        # print("四元数:", quaternion)
        # quaternion_list.append(quaternion)
        # translation = result.transformation[:3,3].copy()
        # print("平移:", translation)
        # translation_list.append(translation)
        
        # trans = np.array([[ 0.99988964,  0.01002983, -0.01095947,  0.00581501],
        #                     [-0.01056802 , 0.99868294, -0.05020653,  0.05395981],
        #                     [ 0.01044147,  0.05031681,  0.99867872,  0.02254279],
        #                     [ 0. ,         0.  ,        0. ,         1. ,       ]])
        # result.transformation = trans

        # source_copy = np.array(source.points).copy()
        source_align=source.transform(result.transformation)
        # save source aligned point cloud
        o3d.io.write_point_cloud(f'/data2/TMP/Dataset/scene/scene1/{scene}/' + 'source_aligned.ply', source_align)

        # source_align_left = result.transformation[:3,:3] @ np.array(source_copy).T + result.transformation[:3,3][:,None]
        # source_align_left = source_align_left.T
        # source_align_new = o3d.geometry.PointCloud()
        # source_align_new.points = o3d.utility.Vector3dVector(source_align_left)
        # o3d.io.write_point_cloud(f'/data2/TMP/Dataset/scene/scene1/{scene}/' + 'source_aligned_left.ply', source_align_new)

        
        # project sourse aligned point cloud to image
        source_align_np = np.asarray(source_raw.transform(result.transformation).points)
        source_align_np = K_zed @ source_align_np.T
        source_align_np[:2] /= source_align_np[2]
        source_uv = source_align_np[:2].T

        new_Depth_map = np.zeros((1080,1920))
        new_Depth_map.fill(np.inf)
        for i in range(source_uv.shape[0]):
            u,v = source_uv[i].astype(np.uint16)
            if u>=0 and u<1920 and v>=0 and v<1080:
                new_Depth_map[v,u] = source_align_np[2,i]
        # new_Depth_map = cv2.resize(new_Depth_map,(1920,1080),interpolation=cv2.INTER_NEAREST)

        # save aligned depth map
        cv2.imwrite(f'/data2/TMP/Dataset/scene/scene1/{scene}/' + 'aligned_depth_map.png', (new_Depth_map/0.00025).astype(np.uint16))
        
        # # 可视化结果（可选）
        # source.transform(result.transformation).paint_uniform_color([1, 0, 0])  # 红色为配准后的源点云
        # target.paint_uniform_color([0, 1, 0])  # 绿色为目标点云
        # o3d.visualization.draw_geometries([source, target])
    avg_trans = average_transformations(transform_list)
    print('average transformation', avg_trans)