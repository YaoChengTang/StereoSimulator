import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2

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

# test
if __name__ == '__main__':
    mask_small_path = 
    mask_large_path = 
    depth_path = 
    mask_small = load_image(mask_small_path)
    mask_large = load_image(mask_large_path)
    Z = load_image(depth_path)

    coords_2d = np.argwhere(mask_large & ~mask_small)

    points_xyz = []
    for (i, j) in coords_2d:
        x_val = j
        y_val = i
        z_val = Z[i, j]
        points_xyz.append([x_val, y_val, z_val])
    points_xyz = np.array(points_xyz)  # (N, 3)
    # return mask_
    # 5) 拟合 3D 平面
    a, b, c, d = fit_plane_svd(points_xyz)
    rows, cols = np.where(mask_small)
    for (i,j) in zip(rows, cols):
        Z[i,j] = (-d - (a * i + b * j)) / c
    print("拟合得到的平面参数 (a, b, c, d) = ", (a, b, c, d))
    print(f"平面方程: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")