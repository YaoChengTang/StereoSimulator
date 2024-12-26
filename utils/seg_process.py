import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def load_mask(path):
    mask = np.load(path)
    return mask
import numpy as np
from scipy.ndimage import distance_transform_edt

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



import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation

def get_mask_edge(mask: np.ndarray):
    """
    获取 mask 的边缘像素：即在 mask 中但其 8 邻域(或结构元素)中有 False 的那部分。
    """
    # 结构元素，这里示范 3x3，也可根据需求更换
    se = np.ones((3,3), dtype=bool)
    eroded = binary_erosion(mask, structure=se)
    boundary = mask & ~eroded
    return boundary

def get_mask_within_5_pixels_of_edge(mask: np.ndarray, dist=5):
    """
    获取 mask 中距离边缘 <= dist 个像素的区域
    """
    boundary = get_mask_edge(mask)
    
    # 膨胀边缘 dist 次(或用一个更大的结构元素一次性膨胀)
    se = np.ones((3,3), dtype=bool)
    expanded = boundary.copy()
    for _ in range(dist):
        expanded = binary_dilation(expanded, structure=se)
    
    # 最后再与原 mask 做 AND，保证结果仍在 mask 内
    within_dist = expanded & ~mask
    return np.argwhere(within_dist)


def main_2d_example():
    # 1) 构造一个示例 2D mask
    mask = load_mask('/data4/lzd/iccv25/data/mask_test/1_1/0.npy')
    # 在某个矩形区域设置 True
    Z = np.load('/data4/lzd/iccv25/vis/depth/1_1/0.npy')
    # 3) 找到距离 mask 外部 <= 5 像素的区域
    return Z
    coords_2d = get_mask_within_5_pixels_of_edge(mask, 6)

    mask_ = np.zeros_like(mask)
    points_xyz = []
    for (i, j) in coords_2d:
        x_val = j
        y_val = i
        z_val = Z[i, j]
        mask_[i,j] = 255
        points_xyz.append([x_val, y_val, z_val])
    points_xyz = np.array(points_xyz)  # (N, 3)
    # return mask_
    # 5) 拟合 3D 平面
    a, b, c, d = fit_plane_svd(points_xyz)
    rows, cols = np.where(mask)
    for (i,j) in zip(rows, cols):
        Z[i,j] = (-d - (a * i + b * j)) / c
    print("拟合得到的平面参数 (a, b, c, d) = ", (a, b, c, d))
    print(f"平面方程: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
    return Z

def process(depth, mask):
    new_depth = depth



    return new_depth

if __name__ == "__main__":
    Z = main_2d_example()
    from matplotlib import pyplot as plt
    import cv2
    plt.imsave(f'haha.png', Z, cmap='jet')
    disp = np.round(Z * 64).astype(np.uint16)
    cv2.imwrite('origin.png', disp)