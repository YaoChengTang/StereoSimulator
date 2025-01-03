import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import normalize
from matplotlib.colors import BoundaryNorm
def fit_plane_ransac(points, threshold=0.01, max_trials=1000, seed = 42):
    """
    使用 RANSAC 拟合三维点的平面。

    参数:
        points (np.ndarray): 三维点集，形状为 (N, 3)。
        threshold (float): 点到平面的距离阈值，用于定义内点。
        max_trials (int): RANSAC 的最大迭代次数。
    
    返回:
        plane_params (np.ndarray): 平面参数 (a, b, c, d)，满足 ax + by + cz + d = 0。
        inliers (np.ndarray): 内点索引数组。
    """
    # 提取三维点的坐标
    X = points[:, :2]  # 使用 x 和 y 作为特征
    y = points[:, 2]   # z 作为目标值

    # 使用 RANSAC 进行平面拟合
    ransac = RANSACRegressor(residual_threshold=threshold, max_trials=max_trials, random_state=seed)
    ransac.fit(X, y)

    # 提取 RANSAC 的平面参数
    inliers = ransac.inlier_mask_
    a, b = ransac.estimator_.coef_
    d = ransac.estimator_.intercept_

    # 平面参数 ax + by + cz + d = 0
    normal_vector = np.array([a, b, -1.0])
    # normal_vector = normalize(normal_vector.reshape(1, -1))[0]  # 法向量归一化
    # d = d / np.linalg.norm(normal_vector)
    plane_params = np.append(normal_vector, d)

    return plane_params, inliers
def guided_filter(input_image, guide_image, radius, eps):
    """
    Applies guided filtering to the input image.

    Parameters:
        input_image (np.ndarray): The input image to be filtered (e.g., mask region).
        guide_image (np.ndarray): The guide image (e.g., source image or mask itself).
        radius (int): Radius of the filter (window size).
        eps (float): Regularization parameter (controls the smoothness).

    Returns:
        np.ndarray: The filtered output image.
    """
    # Convert images to float32
    scale = input_image.max()
    input_image = input_image.astype(np.float32) / scale
    guide_image = guide_image.astype(np.float32) / 255.0
    
    # Apply guided filter
    filtered = cv2.ximgproc.guidedFilter(guide_image, input_image, radius, eps)
    
    # Scale back to 0-255 range and convert to uint8
    filtered = (filtered * scale).astype(np.uint16)
    return filtered


def load_image(fpath):
    mask = Image.open(fpath)
    return np.array(mask)
def fit_plane_svd(points_xyz: np.ndarray):
    """
    使用 SVD 对给定的 3D 点集进行平面拟合，返回 (a, b, c, d)，
    使得 a*x + b*y + c*z + d = 0.
    points_xyz: (N, 3) 的 numpy 数组，每行是 (x, y, z).
    """
    # 构造矩阵 M: shape = (N, 4)
    M = np.hstack([points_xyz, np.ones((points_xyz.shape[0], 1))])  # [x, y, z, 1]
    
    # 对 M 做 SVD 分解
    # M = U * S * V^T
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    
    # Vt 的最后一行（即转置前的最后一列）是最小奇异值对应的特征向量
    plane_params = Vt[-1, :]  # [a, b, c, d]
    
    # 为了后续使用方便，可以让 (a, b, c) 归一化
    a, b, c, d = plane_params
    norm_factor = np.sqrt(a*a + b*b + c*c)
    if norm_factor > 1e-12:
        plane_params /= norm_factor
    
    return plane_params  # (a, b, c, d)



def visualize_ransac(points, inliers, plane_params):
    """
    可视化 RANSAC 内点和外点以及拟合的平面。

    参数:
        points (np.ndarray): 原始三维点集，形状为 (N, 3)。
        inliers (np.ndarray): 内点的布尔索引数组，形状为 (N,)。
        plane_params (np.ndarray): 拟合平面的参数 (a, b, c, d)。
    """
    # 分离内点和外点
    inlier_points = points[inliers]
    outlier_points = points[~inliers]

    # 创建 3D 绘图对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制内点和外点
    ax.scatter(inlier_points[:, 0], inlier_points[:, 1], inlier_points[:, 2], 
               c='b', label='Inliers', s=5)
    ax.scatter(outlier_points[:, 0], outlier_points[:, 1], outlier_points[:, 2], 
               c='r', label='Outliers', s=5)

    # 绘制拟合的平面
    x = np.linspace(points[:, 0].min(), points[:, 0].max(), 10)
    y = np.linspace(points[:, 1].min(), points[:, 1].max(), 10)
    x, y = np.meshgrid(x, y)
    z = (-plane_params[0] * x - plane_params[1] * y - plane_params[3]) / plane_params[2]
    ax.plot_surface(x, y, z, color='g', alpha=0.5, label="Fitted Plane")

    # 设置图形属性

    plt.savefig('/data4/lzd/iccv25/vis/haha.png')

# test
if __name__ == '__main__':
    mask_small_path = "/data4/lzd/iccv25/vis/mask/Y2metaapp3D_Trick_Art_Hole_On_Line_Paper_Traffic_signs_No_horn_honking/326.jpg"
    mask_large_path = "/data4/lzd/iccv25/vis/mask/Y2metaapp3D_Trick_Art_Hole_On_Line_Paper_Traffic_signs_No_horn_honking/326_all.jpg"
    depth_path = "/data4/lzd/iccv25/vis/depth_anything/Y2metaapp3D_Trick_Art_Hole_On_Line_Paper_Traffic_signs_No_horn_honking/326.png"
    source_path = "/data4/lzd/iccv25/vis/imgL/Y2metaapp3D_Trick_Art_Hole_On_Line_Paper_Traffic_signs_No_horn_honking/326.png"
    mask_small = load_image(mask_small_path)
    mask_large = load_image(mask_large_path)
    source_image = load_image(source_path)
    mask_small = mask_small > 0
    mask_large = mask_large > 0
    Z = load_image(depth_path)
    plt.imsave('/data4/lzd/iccv25/vis/test_post/1_o.png',Z, cmap='jet')
    coords_2d = np.argwhere(mask_large & ~mask_small)
    points_xyz = []
    for (i, j) in coords_2d:
        x_val = i
        y_val = j
        z_val = Z[i, j]
        points_xyz.append([x_val, y_val, z_val])
    points_xyz = np.array(points_xyz)  # (N, 3)
    # return mask_

    # a, b, c, d = fit_plane_svd(points_xyz)
    # print("拟合得到的平面参数 (a, b, c, d) = ", (a, b, c, d))
    # print(f"平面方程: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

    plane_params, inliers = fit_plane_ransac(points_xyz, threshold=1, max_trials=1000)
    a,b,c,d = plane_params
    print("拟合的平面参数: ax + by + cz + d = 0")
    print(f"a, b, c, d = {plane_params}")
    print(f"内点数量: {np.sum(inliers)}")
    cnt = 0
    # rows, cols = np.where(mask_large & ~mask_small)
    # for (i,j) in zip(rows, cols):
    #     Z[i,j] -= (-d - (a * i + b * j)) / c
    #     Z[i,j] = abs(Z[i,j])
    #     if Z[i,j] < np.sqrt(a*a + b*b + 1):
    #         cnt += 1
    # print(cnt)
    #     # print(Z[i,j])
    # rows, cols = np.where(~(mask_large & ~mask_small))
    # for (i,j) in zip(rows, cols):
    #     Z[i,j] = 0

    # bounds = [0, 10, 20, 30, 50, 80, 100]  # 分段边界
    # norm = BoundaryNorm(bounds, ncolors=256)  # 定义边界归一化
    # cmap = plt.get_cmap('jet')  # 使用 jet 颜色映射
    # plt.imshow(Z, cmap=cmap, norm=norm)
    # plt.axis('off')  # 去掉坐标轴
    # plt.savefig('/data4/lzd/iccv25/vis/test_post/error_1000.png', bbox_inches='tight', pad_inches=0)
    # plt.close()
    # 保存图像
    # exit(0)
    # visualize_ransac(points_xyz, inliers, plane_params)

    rows, cols = np.where(mask_small)
    for (i,j) in zip(rows, cols):
        Z[i,j] = (-d - (a * i + b * j)) / c



    Z = Z.astype(np.uint16)
    # plt.imsave('/data4/lzd/iccv25/vis/test_post/1.png',Z, cmap='jet')
    cv2.imwrite('/data4/lzd/iccv25/vis/test_post/1_.png', Z.astype(np.uint16))
    # Load the source image and mask
    radius = 8  # Window radius
    eps = 1e-2  # Regularization parameter
    filtered_image = guided_filter(Z, source_image, radius, eps)
    Z[mask_large & ~mask_small] = filtered_image[mask_large & ~mask_small]
    # plt.imsave('/data4/lzd/iccv25/vis/test_post/1_f.png',Z, cmap='jet')
    cv2.imwrite('/data4/lzd/iccv25/vis/test_post/1_f_.png', Z.astype(np.uint16))
    # img = Image.fromarray(filtered_image)
    # img.save('/data4/lzd/iccv25/vis/test_post')