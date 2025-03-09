import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
import re
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

def sift_stereo_matching(img_left_path, img_right_path, disp,
                        ratio_thresh=0.7, 
                        ransac_thresh=5.0,
                        vis=True,
                        figsize=(16,8)):
    """
    双目图像SIFT特征匹配函数
    
    参数：
    img_left_path  : 左图像路径
    img_right_path : 右图像路径
    ratio_thresh   : Lowe's ratio阈值(0-1)
    ransac_thresh  : RANSAC重投影阈值(像素)
    vis            : 是否显示可视化结果
    figsize        : 显示图像尺寸
    
    返回：
    good_matches   : 筛选后的匹配对
    matched_vis     : 可视化图像(OpenCV格式)
    """
    # 读取图像
    img_left = cv2.imread(img_left_path)
    img_right = cv2.imread(img_right_path)
    # ill_mask_path = img_left_path.replace('left', 'mask')
    # ill_mask = cv2.imread(ill_mask_path.replace('.png','.jpg'), cv2.IMREAD_GRAYSCALE)
    assert img_left is not None, f'Invalid image path: {img_left_path}'
    assert img_right is not None, f'Invalid image path: {img_right_path}'
    # disp = readPFM(disp_path)
    mask_disp = (disp>0)&(disp<np.inf)
    
    # 转换为灰度图
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    # gray_left[ill_mask==255] = 255
    # gray_right[ill_mask==255] = 255

    # 初始化SIFT检测器
    sift = cv2.SIFT_create()

    # 检测关键点并计算描述子
    kp1, des1 = sift.detectAndCompute(gray_left, None)
    kp2, des2 = sift.detectAndCompute(gray_right, None)
    
    # get position of valid point
    # import pdb; pdb.set_trace()

    # right_points = []
    # for i in range(len(kp2)):
    #     x, y = kp2[i].pt
    #     x, y = int(x), int(y)
    #     if mask[y, x]:
    #         right_points.append((x, y))

    # error = [(left[0]-right[0],left[1]-right[1]) for left,right in zip(left_points, right_points)]
    # import pdb ; pdb.set_trace()
    # FLANN匹配器配置
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=10)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # KNN特征匹配
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test筛选
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # 可选：RANSAC几何验证
    if len(good_matches) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        good_matches = [good_matches[i] for i in range(len(mask)) if mask[i]]

    # get match points
    left_match_points = []
    right_match_points = []
    remove_list = []
    for i in range(len(good_matches)):
        x1, y1 = kp1[good_matches[i].queryIdx].pt
        x1, y1 = int(x1), int(y1)
        x2, y2 = kp2[good_matches[i].trainIdx].pt
        x2, y2 = int(x2), int(y2)
        # if x1>=disp.shape[1]//3:
        #     remove_list.append(i)
        #     continue
        if mask_disp[y1, x1]:
            left_match_points.append((x1, y1))
            right_match_points.append((x2, y2))
    
    # good_matches = [good_matches[i] for i in range(len(good_matches)) if i not in remove_list]

    disp_gt = [disp[y, x] for x, y in left_match_points]
    disp_pred = [(x1-x2,y1-y2) for (x1, y1), (x2, y2) in zip(left_match_points, right_match_points)]
    error = [gt-pred[0] for gt, pred in zip(disp_gt, disp_pred)]
    import pdb; pdb.set_trace()
    print(f'error mean: {np.mean(error)}')
    print(f'error std: {np.std(error)}')
    print(f'error max: {np.max(error)}')
    print(f'error min: {np.min(error)}')
    print(f'error median: {np.median(error)}')
    # draw a distribution of error
    plt.figure()
    plt.hist(error, bins=100)
    plt.title('Distribution of error')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.savefig(f'./tmp/{os.path.dirname(img_left_path).split("/")[-1]}_{os.path.basename(img_left_path)}_error.png')
    # import pdb; pdb.set_trace()



    # 生成可视化图像
    draw_params = dict(
        matchColor=(0, 255, 0),  # 绿色连线
        singlePointColor=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    matched_vis = cv2.drawMatches(img_left, kp1, img_right, kp2, 
                                good_matches, None, **draw_params)

    # Matplotlib可视化
    if vis:
        plt.figure(figsize=figsize)
        matched_vis_rgb = cv2.cvtColor(matched_vis, cv2.COLOR_BGR2RGB)
        plt.imshow(matched_vis_rgb)
        plt.title(f'SIFT Feature Matches (Count: {len(good_matches)})')
        plt.axis('off')
        # plt.show()
        os.makedirs(f'./tmp', exist_ok=True)
        plt.savefig(f'./tmp/{os.path.dirname(img_left_path).split("/")[-1]}_{os.path.basename(img_left_path)}_sift.png')

    return good_matches, matched_vis

# 使用示例
if __name__ == '__main__':
    
    left_dir='/data2/Fooling3D/real_data/Dataset_second/Dataset/monitor/monitor_pose2-Drivng3/0004'
    right_dir='/data2/Fooling3D/real_data/Dataset_second/Dataset/monitor/monitor_pose2-Drivng3/0004'
    disp_dir='/data2/Fooling3D/real_data/Dataset_second/Dataset/monitor/monitor_pose2-Drivng3/0004'
    left_sub_path = 'zed_left_color_image.png'
    right_sub_path = 'zed_right_color_image.png'
    depth_sub_path = 'zed_depth_image.png'
    # disp_sub_path = left_sub_path.replace('.png', '.pfm')
    left_path = os.path.join(left_dir, left_sub_path)
    right_path = os.path.join(right_dir, right_sub_path)
    depth_path = os.path.join(disp_dir, depth_sub_path)
    depth = cv2.imread(depth_path,cv2.IMREAD_UNCHANGED)*0.00025

    intrinsic = np.array([[1516.77, 0, 939.13],[0, 1516.77, 550.55],[0, 0, 1]])
    fx = intrinsic[0,0]
    b = 0.06297

    disp = b*fx/depth.copy()
    disp[depth<=0] = 0
    disp[depth==np.inf] = 0
    assert not np.any(np.isnan(disp)), 'Invalid depth value'
    assert not np.any(np.isinf(disp)), 'Invalid depth value'
    print(disp)

    # disp_path = os.path.join(disp_dir, disp_sub_path)
    matches, vis_image = sift_stereo_matching(
        img_left_path=left_path,
        img_right_path=right_path,
        disp=disp,
        ratio_thresh=0.7,
        ransac_thresh=5.0,
        vis=True
    )