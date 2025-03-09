import os
import numpy as np
# import open3d as o3d
import re

import cv2
import matplotlib.pyplot as plt
def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

def disp_filter(disp_list,max_threshold=1):
    """
    使用Open3D进行视差图滤波

    参数:
    disp_list (list): 包含视差图路径的列表

    返回:
    list: 滤波后的视差图
    """
    # 检查输入有效性
    if not disp_list:
        raise ValueError("输入视差图列表不能为空")


    disp_stack=np.stack(disp_list,axis=0)
    
    disp_median = np.median(disp_stack, axis=0)

    disp_inconsistency=np.abs(disp_stack-disp_median)

    filter_mask = np.max(disp_inconsistency,axis=0)<max_threshold

    return disp_median,filter_mask

def depth_filter(depth_list, max_threshold=0.01):
    """
    使用Open3D进行深度图滤波

    参数:
    depth_list (list): 包含深度图路径的列表

    返回:
    list: 滤波后的深度图
    """
    # 检查输入有效性
    if not depth_list:
        raise ValueError("输入深度图列表不能为空")

    depth_stack=np.stack(depth_list,axis=0)
    
    depth_median = np.median(depth_stack, axis=0)

    depth_inconsistency=np.abs(depth_stack-depth_median)

    filter_mask = np.max(depth_inconsistency,axis=0)<max_threshold

    return depth_median,filter_mask

def writePFM(file, array):
    import os
    assert type(file) is str and type(array) is np.ndarray and \
           os.path.splitext(file)[1] == ".pfm"
    with open(file, 'wb') as f:
        H, W = array.shape
        headers = ["Pf\n", f"{W} {H}\n", "-1\n"]
        for header in headers:
            f.write(str.encode(header))
        array = np.flip(array, axis=0).astype(np.float32)
        f.write(array.tobytes())

if __name__=='__main__':
    # 读取示例数据（实际使用时替换为真实视差图路径）
    root = '/data4/lzd/baseline/disp/scene1'
    methods = os.listdir(root)
    scenes = os.listdir(os.path.join(root,methods[0]))
    methods.remove('fuse')
    fx = 1516.77
    b = 0.06297
    scale =0.5 # half resolution
    fx = fx*scale
    for scene in scenes:
        disp_list = []
        depth_list = []
        for method in methods:
            disp_list.append(readPFM(os.path.join(root,method,scene,'zed_left_color_image.pfm')))
        for disp in disp_list:
            depth = fx*b/disp
            depth[disp<0]=0
            depth_list.append(depth)
        disp_filtered,filter_mask_disp = disp_filter(disp_list)
        depth_filtered,filter_mask_depth = depth_filter(depth_list)
        filter_mask=filter_mask_disp&filter_mask_depth
        print('mask_Rate:',np.sum(filter_mask)/filter_mask.size)
        disp_filtered[~filter_mask]=0
        depth_filtered[~filter_mask]=0
        # 保存滤波后的视差图
        print(os.path.join(root,'fuse',scene))
        os.makedirs(os.path.join(root,'fuse',scene),exist_ok=True)
        writePFM(os.path.join(root,'fuse',scene,'disp_filtered.pfm'), disp_filtered)
        # 保存滤波后的深度图
        np.save(os.path.join(root,'fuse',scene,'depth_filtered.npy'), depth_filtered)
        # 可视化滤波后的视差图
        plt.imsave(os.path.join(root,'fuse',scene,'disp_filtered.png'), disp_filtered, cmap='jet')
        # 保存滤波mask
        cv2.imwrite(os.path.join(root,'fuse',scene,'filter_mask.png'), filter_mask.astype(np.uint8)*255)


    # 执行视差图滤波
    # disp_filtered = disp_filter(disp_list)


    