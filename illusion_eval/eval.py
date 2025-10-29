import numpy as np
import argparse
import cv2
import glob
import os
import re
import torch
from tqdm import tqdm
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt

def vis_depth_map(depth,vmin=0,vmax=10,mask=None):
    # depth[depth<0] = 0
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    depth = (depth - vmin) / (vmax - vmin)
    depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    if mask is not None:
        depth[mask] = 0
    return depth

def vis_disp_map(disp,vmin=0,vmax=192,mask=None):
    disp[disp<0] = 0
    cmap = matplotlib.colormaps.get_cmap('jet')
    disp = (disp - vmin) / (vmax - vmin)
    disp = np.clip(disp,0,1)
    disp = (cmap(disp)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    if mask is not None:
        disp[mask] = 0
    return disp

def cal_bad(pred, gt, mask, threshold=2):
    # pred: predicted disp
    # gt: ground truth disp
    # mask: valid mask
    return np.sum((np.abs(pred-gt)>threshold).astype(np.float32)[mask])/np.sum(mask.astype(np.float32))

def cal_abs_rel(pred, gt, mask, valid_mask):
    # pred: predicted inverse depth
    # gt: ground truth inverse depth
    # mask: valid mask

    # affine transform
    pred = pred[valid_mask]
    gt = gt[valid_mask]
    mask = mask[valid_mask]

    pred = pred-np.median(pred)
    gt = gt-np.median(gt)
    pred = pred/np.mean(np.abs(pred))
    gt = gt/np.mean(np.abs(gt))
    
    # normalize to 0~1
    pred = (pred-np.min(pred))/(np.max(pred)-np.min(pred))
    gt = (gt-np.min(gt))/(np.max(gt)-np.min(gt))

    abs_rel = np.mean((np.abs(pred-gt)/gt)[(mask)&(gt!=0)])

    return abs_rel

def cal_abs_rel_metric(pred, gt, mask):
    mask = mask & (gt>0)
    pred = pred[mask]
    gt = gt[mask]
    # assert np.all(gt>0),[gt.min(),gt.max()]
    # assert np.all(gt>0),[gt.min(),gt.max()]
    abs_rel = np.mean(np.abs(pred-gt)/gt)
    return abs_rel

def cal_depth_rms(pred, gt, mask):
    pred = pred[mask]
    gt = gt[mask]
    rms = np.sqrt(np.mean((pred-gt)**2))
    return rms

def cal_depth_log10(pred, gt, mask):
    pred = pred[mask]
    gt = gt[mask]
    log10 = np.mean(np.abs(np.log10(pred+1e-8)-np.log10(gt+1e-8)))
    return log10

def cal_delta_n(pred, gt, mask, n, threshold=1.25):
    mask = mask & (gt>0)
    pred = pred[mask]
    gt = gt[mask]
    delta_n = np.mean(np.maximum(pred/gt, gt/pred)<threshold**n)
    return delta_n

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

def read_calib_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    fx=0
    baseline=0
    for line in lines:
        if line.startswith('cam0'):
            fx = float(line.split('[')[1].split(']')[0].split(' ')[0])
            fy = float(line.split('[')[1].split(']')[0].split('; ')[1].split(' ')[1])
            cx = float(line.split('[')[1].split(']')[0].split('; ')[0].split(' ')[2])
            cy = float(line.split('[')[1].split(']')[0].split('; ')[1].split(' ')[2])
        elif line.startswith('baseline'):
            baseline = float(line.split('=')[1].strip())
        elif line.startswith('doffs'):
            doffs = float(line.split('=')[1].strip())
    assert fx != 0 and baseline != 0
    return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]]), baseline, doffs

def save_as_pcd(rgb, depth, intrinsic, filename, mask=None):
    h, w = depth.shape

    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    points = []
    colors = []

    x_range = np.arange(0, w)
    y_range = np.arange(0, h)
    x, y = np.meshgrid(x_range, y_range)
    grid = np.stack([x, y, np.ones_like(x)], axis=-1)
    grid = grid.reshape(-1, 3)
    points = depth.reshape(-1, 1) * np.dot(np.linalg.inv(intrinsic), grid.T).T
    colors = rgb.reshape(-1, 3)
    if mask is not None:
        points = points[mask.reshape(-1)]
        colors = colors[mask.reshape(-1)]
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors/255.0)
    o3d.io.write_point_cloud(filename, pcd)

class affineInvDepth(nn.Module):
    def __init__(self,predInvDepth):
        super(affineInvDepth, self).__init__()
        self.predInvDepth = predInvDepth
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.shift = nn.Parameter(torch.tensor(0.0))
    
    def get_metric_depth(self):
        return self.scale*(self.predInvDepth) + self.shift

def get_depth_by_pred_type(pred,pred_type,b,fx,doffs=0):
    # assert np.all(pred>=0),[pred.min(),pred.max()]
    # assert np.all(pred<np.inf),[pred.min(),pred.max()]
    if pred_type=='depth':
        pred_depth = pred
    elif pred_type=='inv_depth':
        pred_depth = 1/pred
    else:
        pred_depth = fx*b/(pred+doffs)
    return pred_depth

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation for Middlebury')
    parser.add_argument('--pred-path', type=str)
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--gt-path', type=str)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--resize-scale', type=float, default=1.0)
    parser.add_argument('--pred_type', choices=['disp', 'depth', 'inv_depth'])
    parser.add_argument('--focal-align', action='store_true')
    parser.add_argument('--max-depth-align', action='store_true')
    parser.add_argument('--scale-shift-align', action='store_true')
    parser.add_argument('--scale-align', action='store_true')
    parser.add_argument('--optim-align', action='store_true')
    parser.add_argument('--vis', action='store_true')

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    pred_files = glob.glob(os.path.join(args.pred_path, '**/*.png'), recursive=True)+glob.glob(os.path.join(args.pred_path, '**/*.npy'), recursive=True)+\
        glob.glob(os.path.join(args.pred_path, '**/*.pfm'), recursive=True)

    pred_files.sort()
    # print(len(pred_files))
    # print(pred_files)
    # exit()
    # all
    epe = []
    bad2 = []
    bad3 = []
    bad4 = []
    bad5 = []
    bad6 = []
    bad7 = []
    abs_rel=[] 
    rms=[]
    log10=[]
    delta1=[]
    delta2=[]
    delta3=[]

    abs_rel_depth_affine=[]
    rms_depth_affine=[]
    log10_depth_affine=[]
    delta1_depth_affine=[]
    delta2_depth_affine=[]
    delta3_depth_affine=[]

    abs_rel_inv_depth_affine=[]
    rms_inv_depth_affine=[]
    log10_inv_depth_affine=[]
    delta1_inv_depth_affine=[]
    delta2_inv_depth_affine=[]
    delta3_inv_depth_affine=[]

    # ill
    ill_epe=[]
    ill_bad2=[]
    ill_bad3=[]
    ill_bad4=[]
    ill_bad5=[]
    ill_bad6=[]
    ill_bad7=[]
    ill_abs_rel=[]
    ill_rms=[]
    ill_log10=[]
    ill_delta1=[]
    ill_delta2=[]
    ill_delta3=[]

    ill_abs_rel_depth_affine=[]
    ill_rms_depth_affine=[]
    ill_log10_depth_affine=[]
    ill_delta1_depth_affine=[]
    ill_delta2_depth_affine=[]
    ill_delta3_depth_affine=[]

    ill_abs_rel_inv_depth_affine=[]
    ill_rms_inv_depth_affine=[]
    ill_log10_inv_depth_affine=[]
    ill_delta1_inv_depth_affine=[]
    ill_delta2_inv_depth_affine=[]
    ill_delta3_inv_depth_affine=[]

    # nill
    nill_epe=[]
    nill_bad2=[]
    nill_bad3=[]
    nill_bad4=[]
    nill_bad5=[]
    nill_bad6=[]
    nill_bad7=[]
    nill_abs_rel=[]
    nill_rms=[]
    nill_log10=[]
    nill_delta1=[]
    nill_delta2=[]
    nill_delta3=[]

    nill_abs_rel_depth_affine=[]
    nill_rms_depth_affine=[]
    nill_log10_depth_affine=[]
    nill_delta1_depth_affine=[]
    nill_delta2_depth_affine=[]
    nill_delta3_depth_affine=[]

    nill_abs_rel_inv_depth_affine=[]
    nill_rms_inv_depth_affine=[]
    nill_log10_inv_depth_affine=[]
    nill_delta1_inv_depth_affine=[]
    nill_delta2_inv_depth_affine=[]
    nill_delta3_inv_depth_affine=[]

    # metric list
    metric_list = ['epe','bad2','bad3','bad4','bad5','bad6','bad7','abs_rel','rms','log10','delta1','delta2','delta3']
    metric_list = metric_list + ['ill_'+metric for metric in metric_list] + ['nill_'+metric for metric in metric_list]
    
    depth_metric_list = ['abs_rel','rms','log10','delta1','delta2','delta3']

    metric_list = metric_list + [metric+'_depth_affine' for metric in depth_metric_list]+['ill_'+metric+'_depth_affine' for metric in depth_metric_list] + ['nill_'+metric+'_depth_affine' for metric in depth_metric_list] + \
        [metric+'_inv_depth_affine' for metric in depth_metric_list]+['ill_'+metric+'_inv_depth_affine' for metric in depth_metric_list] + ['nill_'+metric+'_inv_depth_affine' for metric in depth_metric_list]
    
    # stats
    stats = {}

    # parameters
    intrinsic = np.array([[1450.4127197265625, 0, 938.3252563476562],[0, 1450.4127197265625, 547.462646484375],[0, 0, 1]])
    intrinsic[:2, :] = intrinsic[:2, :] * args.resize_scale
    fx = intrinsic[0,0]
    b = 0.06281671905517578
    doffs = 0.0

    for k, pred_file in tqdm(enumerate(pred_files)):
        # print(f'Progress {k+1}/{len(pred_files)}: {pred_file}')
        
        # files
        rgb_file = os.path.join(args.data_path, os.path.dirname(pred_file).split('/')[-1], os.path.basename(pred_file).split('.')[0].split("_depth")[0]+'.png')
        gt_file = os.path.join(args.gt_path, os.path.dirname(pred_file).split('/')[-1], os.path.basename(pred_file).split('.')[0].split("_depth")[0]+'.pfm')
        mask_file = os.path.join(args.gt_path.replace('disp','mask'), os.path.dirname(pred_file).split('/')[-1].split("_depth")[0], os.path.basename(pred_file).split('.')[0].split("_depth")[0]+'.jpg')
        
        # read rgb
        if not os.path.exists(rgb_file):
            print(f'No rgb file: {rgb_file}')
            continue
        rgb = cv2.imread(rgb_file, cv2.IMREAD_COLOR)[...,::-1]

        # resize
        if args.resize_scale!=1.0:
            im_size = (int(rgb.shape[1]*args.resize_scale), int(rgb.shape[0]*args.resize_scale))
            rgb = cv2.resize(rgb, im_size, interpolation=cv2.INTER_LINEAR)
        
        # read ill mask
        mask = cv2.imread(mask_file)[:,:,0]
        if args.resize_scale!=1.0:
            mask = cv2.resize(mask, im_size, interpolation=cv2.INTER_NEAREST)
        nill_mask = (mask==0)
        ill_mask = (mask==255)

        # read pred
        if pred_file.endswith('.npy'):
            pred = np.load(pred_file)
        elif pred_file.endswith('.png') or pred_file.endswith('.tiff'):
            pred = cv2.imread(pred_file, cv2.IMREAD_UNCHANGED)
            if len(pred.shape)==3:
                pred = pred[:,:,0]
        elif pred_file.endswith('.pfm'):
            pred = readPFM(pred_file)
        else:
            raise NotImplementedError

        if args.resize_scale!=1.0 and rgb.shape[:2]!=pred.shape[:2]:
            pred = cv2.resize(pred, im_size, interpolation=cv2.INTER_LINEAR)
            if args.pred_type=='disp':
                pred = pred*(rgb.shape[:2]/pred.shape[:2])

        valid_mask = (pred<np.inf) & (pred>0)  # pred==0不好界定，算作无效

        # get depth, inv_depth, disp
        pred_depth = get_depth_by_pred_type(pred,args.pred_type,b,fx,doffs)
        pred_inv_depth = 1/(pred_depth)
        pred_disp = fx*b/(pred_depth+doffs)
        
        # pred raw
        pred_depth_raw = pred_depth.copy()
        pred_inv_depth_raw = pred_inv_depth.copy()
        pred_disp_raw = pred_disp.copy()

        # read gt
        if gt_file.endswith('.npy'):
            gt_disp = np.load(gt_file)
        elif gt_file.endswith('.png') or gt_file.endswith('.tiff'):
            gt_disp = cv2.imread(gt_file, cv2.IMREAD_UNCHANGED)
            if len(gt_disp.shape)==3:
                gt_disp = gt_disp[:,:,0]
        elif gt_file.endswith('.pfm'):
            gt_disp = readPFM(gt_file)
        
        if args.resize_scale!=1.0:
            gt_disp = cv2.resize(gt_disp, im_size, interpolation=cv2.INTER_NEAREST)
            gt_disp = gt_disp*args.resize_scale

        gt_depth = fx*b/(gt_disp+doffs)
        gt_inv_depth = 1/gt_depth

        # gt raw
        gt_inv_depth_raw = gt_inv_depth.copy()
        gt_depth_raw = gt_depth.copy()
        gt_disp_raw = gt_disp.copy()

        gt_mask = (gt_disp<np.inf) & (gt_disp>0)

        valid_mask = (gt_disp<np.inf) & (gt_disp>0) & (gt_depth<5) & (gt_depth>0.3) & valid_mask 
        ill_mask = ill_mask & valid_mask
        nill_mask = nill_mask & valid_mask


        # align
        align_region = nill_mask
        if args.scale_shift_align:
            if args.pred_type=='inv_depth':                
                gt_shift = np.median(gt_inv_depth[align_region])
                gt_scale = np.mean(np.abs(gt_inv_depth[align_region]-gt_shift))

                pred_shift = np.median(pred_inv_depth[align_region])
                pred_scale = np.mean(np.abs(pred_inv_depth[align_region]-pred_shift))

                pred_inv_depth = gt_scale/pred_scale * (pred_inv_depth-pred_shift) + gt_shift
                
                pred_depth = np.clip(1/pred_inv_depth, 0.3, 5)
                pred_disp = fx*b/pred_depth - doffs
                pred_inv_depth = 1/pred_depth

            elif args.pred_type=='depth':
                gt_shift = np.median(gt_depth[align_region])
                gt_scale = np.mean(np.abs(gt_depth[align_region]-gt_shift))

                pred_shift = np.median(pred_depth[align_region])
                pred_scale = np.mean(np.abs(pred_depth[align_region]-pred_shift))

                pred_depth = gt_scale/pred_scale * (pred_depth-pred_shift) + gt_shift
                
                pred_depth = np.clip(pred_depth, 0.3, 5)
                pred_disp = fx*b/pred_depth - doffs
                pred_inv_depth = 1/pred_depth
        
        elif args.scale_align:
            if args.pred_type=='inv_depth':             
                gt_scale = np.mean(gt_inv_depth[align_region])

                pred_scale = np.mean(pred_inv_depth[align_region])

                pred_inv_depth = gt_scale/pred_scale * (pred_inv_depth)
                
                pred_depth = np.clip(1/pred_inv_depth, 0.3, 5)
                pred_disp = fx*b/pred_depth - doffs
                pred_inv_depth = 1/pred_depth

            elif args.pred_type=='depth':
                gt_scale = np.mean(gt_depth[align_region])

                pred_scale = np.mean(pred_depth[align_region])

                pred_depth = gt_scale/pred_scale * (pred_depth)
                
                pred_depth = np.clip(pred_depth, 0.3, 5)
                pred_disp = fx*b/pred_depth - doffs
                pred_inv_depth = 1/pred_depth
                
        else:
            pred_depth = np.clip(pred_depth, 0.3, 5)
            pred_disp = fx*b/pred_depth - doffs
            pred_inv_depth = 1/pred_depth

        # if args.optim_align:
        #     gt_inv_depth = 1/gt_depth
        #     pred_inv_depth_pt = torch.tensor(pred_inv_depth).unsqueeze(0).unsqueeze(0).cuda() # 1,1,h,w
        #     gt_inv_depth_pt = torch.tensor(gt_inv_depth).unsqueeze(0).unsqueeze(0).cuda() # 1,1,h,w 
        #     nill_mask_pt = torch.tensor(nill_mask).unsqueeze(0).unsqueeze(0).cuda() # 1,1,h,w
        #     pred_affine_inv_depth = affineInvDepth(pred_inv_depth_pt).cuda()
        #     optimizer = torch.optim.Adam(pred_affine_inv_depth.parameters(), lr=0.01)
        #     # learning rate scheduler
        #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.7)
        #     criterion = torch.nn.SmoothL1Loss(reduction='mean')
        #     iters=1000
        #     for i in range(iters):
        #         # forward
        #         optimizer.zero_grad()
        #         pred_inv_depth = pred_affine_inv_depth.get_metric_depth()    
        #         thres_mask = torch.abs(1/pred_inv_depth[nill_mask_pt]-1/gt_inv_depth_pt[nill_mask_pt])<1*((0.5)**(3*(i/iters)))
        #         loss = criterion(pred_inv_depth[nill_mask_pt][thres_mask], gt_inv_depth_pt[nill_mask_pt][thres_mask])
        #         loss.backward()
        #         optimizer.step()
        #         scheduler.step()
        #         print(f'Iter {i}: Loss {loss.item()}')
        #     pred_depth = 1/pred_affine_inv_depth.get_metric_depth().squeeze().squeeze().cpu().detach().numpy()
        #     pred_disp = fx*b/pred_depth - doffs

        
        # # all
        epe.append(np.mean(np.abs(pred_disp[valid_mask] - gt_disp[valid_mask])))
        bad2.append(cal_bad(pred_disp, gt_disp, valid_mask, 2))
        bad3.append(cal_bad(pred_disp, gt_disp, valid_mask, 3))
        bad4.append(cal_bad(pred_disp, gt_disp, valid_mask, 4))
        bad5.append(cal_bad(pred_disp, gt_disp, valid_mask, 5))
        bad6.append(cal_bad(pred_disp, gt_disp, valid_mask, 6))
        bad7.append(cal_bad(pred_disp, gt_disp, valid_mask, 7))
        
        # # ill
        ill_epe.append(np.mean(np.abs(pred_disp[ill_mask] - gt_disp[ill_mask])))
        ill_bad2.append(cal_bad(pred_disp, gt_disp, ill_mask, 2))
        ill_bad3.append(cal_bad(pred_disp, gt_disp, ill_mask, 3))
        ill_bad4.append(cal_bad(pred_disp, gt_disp, ill_mask, 4))
        ill_bad5.append(cal_bad(pred_disp, gt_disp, ill_mask, 5))
        ill_bad6.append(cal_bad(pred_disp, gt_disp, ill_mask, 6))
        ill_bad7.append(cal_bad(pred_disp, gt_disp, ill_mask, 7))
        
        # # nill
        nill_epe.append(np.mean(np.abs(pred_disp[nill_mask] - gt_disp[nill_mask])))
        nill_bad2.append(cal_bad(pred_disp, gt_disp, nill_mask, 2))
        nill_bad3.append(cal_bad(pred_disp, gt_disp, nill_mask, 3))
        nill_bad4.append(cal_bad(pred_disp, gt_disp, nill_mask, 4))
        nill_bad5.append(cal_bad(pred_disp, gt_disp, nill_mask, 5))
        nill_bad6.append(cal_bad(pred_disp, gt_disp, nill_mask, 6))
        nill_bad7.append(cal_bad(pred_disp, gt_disp, nill_mask, 7))

        # all
        abs_rel.append(cal_abs_rel_metric(pred_depth, gt_depth, valid_mask))
        rms.append(cal_depth_rms(pred_depth, gt_depth, valid_mask))
        log10.append(cal_depth_log10(pred_depth, gt_depth, valid_mask))
        delta1.append(cal_delta_n(pred_depth, gt_depth, valid_mask, 1))
        delta2.append(cal_delta_n(pred_depth, gt_depth, valid_mask, 2))
        delta3.append(cal_delta_n(pred_depth, gt_depth, valid_mask, 3))

        # ill
        ill_abs_rel.append(cal_abs_rel_metric(pred_depth, gt_depth, ill_mask))
        ill_rms.append(cal_depth_rms(pred_depth, gt_depth, ill_mask))
        ill_log10.append(cal_depth_log10(pred_depth, gt_depth, ill_mask))
        ill_delta1.append(cal_delta_n(pred_depth, gt_depth, ill_mask, 1))
        ill_delta2.append(cal_delta_n(pred_depth, gt_depth, ill_mask, 2))
        ill_delta3.append(cal_delta_n(pred_depth, gt_depth, ill_mask, 3))

        # nill
        nill_abs_rel.append(cal_abs_rel_metric(pred_depth, gt_depth, nill_mask))
        nill_rms.append(cal_depth_rms(pred_depth, gt_depth, nill_mask))
        nill_log10.append(cal_depth_log10(pred_depth, gt_depth, nill_mask))
        nill_delta1.append(cal_delta_n(pred_depth, gt_depth, nill_mask, 1))
        nill_delta2.append(cal_delta_n(pred_depth, gt_depth, nill_mask, 2))
        nill_delta3.append(cal_delta_n(pred_depth, gt_depth, nill_mask, 3))

        # # # affine inverse depth
        pred_inv_depth_affine = pred_inv_depth_raw
        if not args.scale_align:
            pred_inv_depth_affine = pred_inv_depth_affine-np.median(pred_inv_depth_affine[valid_mask])
        pred_inv_depth_affine = pred_inv_depth_affine/np.mean(np.abs(pred_inv_depth_affine[valid_mask]))
        pred_inv_depth_affine = (pred_inv_depth_affine-np.min(pred_inv_depth_affine[valid_mask]))/(np.max(pred_inv_depth_affine[valid_mask])-np.min(pred_inv_depth_affine[valid_mask])) # 0~1

        gt_inv_depth_affine = gt_inv_depth_raw
        if not args.scale_align:
            gt_inv_depth_affine = gt_inv_depth_affine-np.median(gt_inv_depth_affine[valid_mask])
        gt_inv_depth_affine = gt_inv_depth_affine/np.mean(np.abs(gt_inv_depth_affine[valid_mask]))
        gt_inv_depth_affine = (gt_inv_depth_affine-np.min(gt_inv_depth_affine[valid_mask]))/(np.max(gt_inv_depth_affine[valid_mask])-np.min(gt_inv_depth_affine[valid_mask])) # 0~1

        # # # affine depth
        pred_depth_affine = pred_depth_raw       
        if not args.scale_align:
            pred_depth_affine = pred_depth_affine-np.median(pred_depth_affine[valid_mask])
        pred_depth_affine = pred_depth_affine/np.mean(np.abs(pred_depth_affine[valid_mask]))
        pred_depth_affine = (pred_depth_affine-np.min(pred_depth_affine[valid_mask]))/(np.max(pred_depth_affine[valid_mask])-np.min(pred_depth_affine[valid_mask])) # 0~1
        
        gt_depth_affine = gt_depth_raw
        if not args.scale_align:
            gt_depth_affine = gt_depth_affine-np.median(gt_depth_affine[valid_mask])
        gt_depth_affine = gt_depth_affine/np.mean(np.abs(gt_depth_affine[valid_mask]))
        gt_depth_affine = (gt_depth_affine-np.min(gt_depth_affine[valid_mask]))/(np.max(gt_depth_affine[valid_mask])-np.min(gt_depth_affine[valid_mask])) # 0~1        
        
        # depth affine
        # depth_affine_valid = valid_mask & (gt_depth_affine>0)
        abs_rel_depth_affine.append(cal_abs_rel_metric(pred_depth_affine, gt_depth_affine, valid_mask))
        rms_depth_affine.append(cal_depth_rms(pred_depth_affine, gt_depth_affine, valid_mask))
        log10_depth_affine.append(cal_depth_log10(pred_depth_affine, gt_depth_affine, valid_mask))
        delta1_depth_affine.append(cal_delta_n(pred_depth_affine, gt_depth_affine, valid_mask, 1))
        delta2_depth_affine.append(cal_delta_n(pred_depth_affine, gt_depth_affine, valid_mask, 2))
        delta3_depth_affine.append(cal_delta_n(pred_depth_affine, gt_depth_affine, valid_mask, 3))

        ill_abs_rel_depth_affine.append(cal_abs_rel_metric(pred_depth_affine, gt_depth_affine, ill_mask))
        ill_rms_depth_affine.append(cal_depth_rms(pred_depth_affine, gt_depth_affine, ill_mask))
        ill_log10_depth_affine.append(cal_depth_log10(pred_depth_affine, gt_depth_affine, ill_mask))
        ill_delta1_depth_affine.append(cal_delta_n(pred_depth_affine, gt_depth_affine, ill_mask, 1))
        ill_delta2_depth_affine.append(cal_delta_n(pred_depth_affine, gt_depth_affine, ill_mask, 2))
        ill_delta3_depth_affine.append(cal_delta_n(pred_depth_affine, gt_depth_affine, ill_mask, 3))
        
        nill_abs_rel_depth_affine.append(cal_abs_rel_metric(pred_depth_affine, gt_depth_affine, nill_mask))
        nill_rms_depth_affine.append(cal_depth_rms(pred_depth_affine, gt_depth_affine, nill_mask))
        nill_log10_depth_affine.append(cal_depth_log10(pred_depth_affine, gt_depth_affine, nill_mask))
        nill_delta1_depth_affine.append(cal_delta_n(pred_depth_affine, gt_depth_affine, nill_mask, 1))
        nill_delta2_depth_affine.append(cal_delta_n(pred_depth_affine, gt_depth_affine, nill_mask, 2))
        nill_delta3_depth_affine.append(cal_delta_n(pred_depth_affine, gt_depth_affine, nill_mask, 3))

        # inverse depth affine
        abs_rel_inv_depth_affine.append(cal_abs_rel_metric(pred_inv_depth_affine, gt_inv_depth_affine, valid_mask))
        rms_inv_depth_affine.append(cal_depth_rms(pred_inv_depth_affine, gt_inv_depth_affine, valid_mask))
        log10_inv_depth_affine.append(cal_depth_log10(pred_inv_depth_affine, gt_inv_depth_affine, valid_mask))
        delta1_inv_depth_affine.append(cal_delta_n(pred_inv_depth_affine, gt_inv_depth_affine, valid_mask, 1))
        delta2_inv_depth_affine.append(cal_delta_n(pred_inv_depth_affine, gt_inv_depth_affine, valid_mask, 2))
        delta3_inv_depth_affine.append(cal_delta_n(pred_inv_depth_affine, gt_inv_depth_affine, valid_mask, 3))

        ill_abs_rel_inv_depth_affine.append(cal_abs_rel_metric(pred_inv_depth_affine, gt_inv_depth_affine, ill_mask))
        ill_rms_inv_depth_affine.append(cal_depth_rms(pred_inv_depth_affine, gt_inv_depth_affine, ill_mask))
        ill_log10_inv_depth_affine.append(cal_depth_log10(pred_inv_depth_affine, gt_inv_depth_affine, ill_mask))
        ill_delta1_inv_depth_affine.append(cal_delta_n(pred_inv_depth_affine, gt_inv_depth_affine, ill_mask, 1))
        ill_delta2_inv_depth_affine.append(cal_delta_n(pred_inv_depth_affine, gt_inv_depth_affine, ill_mask, 2))
        ill_delta3_inv_depth_affine.append(cal_delta_n(pred_inv_depth_affine, gt_inv_depth_affine, ill_mask, 3))

        nill_abs_rel_inv_depth_affine.append(cal_abs_rel_metric(pred_inv_depth_affine, gt_inv_depth_affine, nill_mask))
        nill_rms_inv_depth_affine.append(cal_depth_rms(pred_inv_depth_affine, gt_inv_depth_affine, nill_mask))
        nill_log10_inv_depth_affine.append(cal_depth_log10(pred_inv_depth_affine, gt_inv_depth_affine, nill_mask))
        nill_delta1_inv_depth_affine.append(cal_delta_n(pred_inv_depth_affine, gt_inv_depth_affine, nill_mask, 1))
        nill_delta2_inv_depth_affine.append(cal_delta_n(pred_inv_depth_affine, gt_inv_depth_affine, nill_mask, 2))
        nill_delta3_inv_depth_affine.append(cal_delta_n(pred_inv_depth_affine, gt_inv_depth_affine, nill_mask, 3))


        file_name = os.path.dirname(pred_file).split('/')[-1]+'/'+os.path.basename(pred_file).split('.')[0]
        stats[file_name] = {metric:eval(metric+'[-1].item()') for metric in metric_list}
        assert not np.any(np.isnan(np.array(list(stats[file_name].values())))),stats[file_name]
        assert not np.any(np.isinf(np.array(list(stats[file_name].values())))),stats[file_name]
        
        if args.vis:
            # save disparity map
            disp_save_path = os.path.join(args.output_path.replace('pcd','disp_pred'), os.path.dirname(pred_file).split('/')[-1], os.path.basename(pred_file).split('.')[0]+'.png')
            os.makedirs(os.path.dirname(disp_save_path), exist_ok=True)

            max_disp = np.percentile(pred_disp[(pred_disp>0)&(pred_disp<np.inf)], 95)
            min_disp = np.percentile(pred_disp[(pred_disp>0)&(pred_disp<np.inf)], 5)  
            pred_disp_colored=vis_disp_map(pred_disp,vmin=min_disp,vmax=max_disp)
            cv2.imwrite(disp_save_path, pred_disp_colored)

            # save inv depth map
            depth_save_path = os.path.join(args.output_path.replace('pcd','depth_pred'), os.path.dirname(pred_file).split('/')[-1], os.path.basename(pred_file).split('.')[0]+'.png')
            os.makedirs(os.path.dirname(depth_save_path), exist_ok=True)

            max_gt_depth = np.percentile(pred_inv_depth_raw[(pred_inv_depth_raw>0)&(pred_inv_depth_raw<np.inf)], 95)
            min_gt_depth = np.percentile(pred_inv_depth_raw[(pred_inv_depth_raw>0)&(pred_inv_depth_raw<np.inf)], 5)
            pred_inv_depth_color=vis_depth_map(pred_inv_depth_raw,vmin=min_gt_depth,vmax=max_gt_depth)
            cv2.imwrite(depth_save_path, pred_inv_depth_color)
            
            # # save error map
            # error_save_path = os.path.join(args.output_path.replace('pcd','disp_error'), os.path.dirname(pred_file).split('/')[-1], os.path.basename(pred_file).split('.')[0]+'.png')
            # os.makedirs(os.path.dirname(error_save_path), exist_ok=True)
            
            # save as pcd
            save_path = os.path.join(args.output_path, os.path.dirname(pred_file).split('/')[-1], os.path.basename(pred_file).split('.')[0]+'.ply')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_as_pcd(rgb, pred_depth, intrinsic, save_path, mask=valid_mask)
            
            # save gt
            if '/DAV2/' in save_path:
                # pcd
                gt_save_path=save_path.replace('DAV2','gt')
                os.makedirs(os.path.dirname(gt_save_path), exist_ok=True)
                save_as_pcd(rgb, gt_depth, intrinsic, gt_save_path, mask=valid_mask)
                
                # disp
                gt_disp_save_path = disp_save_path.replace('DAV2','gt').replace('disp_pred','disp_gt')
                os.makedirs(os.path.dirname(gt_disp_save_path), exist_ok=True)
                max_gt_disp = np.percentile(gt_disp[(gt_disp>0)&(gt_disp<np.inf)], 95)
                min_gt_disp = np.percentile(gt_disp[(gt_disp>0)&(gt_disp<np.inf)], 5)  
                gt_mask = ~np.tile(gt_mask[:,:,None],(1,1,3))
                gt_disp_colored=vis_disp_map(gt_disp,vmin=min_gt_disp,vmax=max_gt_disp,mask=gt_mask)
                cv2.imwrite(gt_disp_save_path, gt_disp_colored)

                # use cv2 color map to visualize depth
                gt_depth_save_path = depth_save_path.replace('DAV2','gt').replace('depth_pred','depth_gt')
                os.makedirs(os.path.dirname(gt_depth_save_path), exist_ok=True)
                max_gt_depth = np.percentile(gt_inv_depth_raw[(gt_inv_depth_raw>0)&(gt_inv_depth_raw<np.inf)], 95)
                min_gt_depth = np.percentile(gt_inv_depth_raw[(gt_inv_depth_raw>0)&(gt_inv_depth_raw<np.inf)], 5)
                gt_depth_colored=vis_depth_map(gt_inv_depth_raw,vmin=min_gt_depth,vmax=max_gt_depth,mask=gt_mask)
                cv2.imwrite(gt_depth_save_path, gt_depth_colored)
                

    
    print('#######################')
    print('EPE:', np.mean(epe))
    # print('AbsRel:', np.mean(abs_rel))
    print('Bad2:', np.mean(bad2))
    print('Bad3:', np.mean(bad3))
    print('Bad4:', np.mean(bad4))
    print('Bad5:', np.mean(bad5))
    print('Bad6:', np.mean(bad6))
    print('Bad7:', np.mean(bad7))
    
    print('ill EPE:', np.mean(ill_epe))
    # print('ill AbsRel:', np.mean(ill_abs_rel))
    print('ill Bad2:', np.mean(ill_bad2))
    print('ill Bad3:', np.mean(ill_bad3))
    print('ill Bad4:', np.mean(ill_bad4))
    print('ill Bad5:', np.mean(ill_bad5))
    print('ill Bad6:', np.mean(ill_bad6))
    print('ill Bad7:', np.mean(ill_bad7))

    print('nill EPE:', np.mean(nill_epe))
    # print('nill AbsRel:', np.mean(nill_abs_rel))
    print('nill Bad2:', np.mean(nill_bad2))
    print('nill Bad3:', np.mean(nill_bad3))
    print('nill Bad4:', np.mean(nill_bad4))
    print('nill Bad5:', np.mean(nill_bad5))
    print('nill Bad6:', np.mean(nill_bad6))
    print('nill Bad7:', np.mean(nill_bad7))

    print('#######################')
    print('AbsRel:', np.mean(abs_rel))
    print('RMS:', np.mean(rms))
    print('Log10:', np.mean(log10))
    print('Delta1:', np.mean(delta1))
    print('Delta2:', np.mean(delta2))
    print('Delta3:', np.mean(delta3))

    print('#######################')
    print('ill AbsRel:', np.mean(ill_abs_rel))
    print('ill RMS:', np.mean(ill_rms))
    print('ill Log10:', np.mean(ill_log10))
    print('ill Delta1:', np.mean(ill_delta1))
    print('ill Delta2:', np.mean(ill_delta2))
    print('ill Delta3:', np.mean(ill_delta3))

    print('#######################')
    print('nill AbsRel:', np.mean(nill_abs_rel))
    print('nill RMS:', np.mean(nill_rms))
    print('nill Log10:', np.mean(nill_log10))
    print('nill Delta1:', np.mean(nill_delta1))
    print('nill Delta2:', np.mean(nill_delta2))
    print('nill Delta3:', np.mean(nill_delta3))

    print('#######################')
    print('AbsRel depth affine:', np.mean(abs_rel_depth_affine))
    print('RMS depth affine:', np.mean(rms_depth_affine))
    print('Log10 depth affine:', np.mean(log10_depth_affine))
    print('Delta1 depth affine:', np.mean(delta1_depth_affine))
    print('Delta2 depth affine:', np.mean(delta2_depth_affine))
    print('Delta3 depth affine:', np.mean(delta3_depth_affine))
    print('#######################')
    print('ill AbsRel depth affine:', np.mean(ill_abs_rel_depth_affine))
    print('ill RMS depth affine:', np.mean(ill_rms_depth_affine))
    print('ill Log10 depth affine:', np.mean(ill_log10_depth_affine))
    print('ill Delta1 depth affine:', np.mean(ill_delta1_depth_affine))
    print('ill Delta2 depth affine:', np.mean(ill_delta2_depth_affine))
    print('ill Delta3 depth affine:', np.mean(ill_delta3_depth_affine))
    print('#######################')
    print('nill AbsRel depth affine:', np.mean(nill_abs_rel_depth_affine))
    print('nill RMS depth affine:', np.mean(nill_rms_depth_affine))
    print('nill Log10 depth affine:', np.mean(nill_log10_depth_affine))
    print('nill Delta1 depth affine:', np.mean(nill_delta1_depth_affine))
    print('nill Delta2 depth affine:', np.mean(nill_delta2_depth_affine))
    print('nill Delta3 depth affine:', np.mean(nill_delta3_depth_affine))
    print('#######################')
    print('AbsRel inv depth affine:', np.mean(abs_rel_inv_depth_affine))
    print('RMS inv depth affine:', np.mean(rms_inv_depth_affine))
    print('Log10 inv depth affine:', np.mean(log10_inv_depth_affine))
    print('Delta1 inv depth affine:', np.mean(delta1_inv_depth_affine))
    print('Delta2 inv depth affine:', np.mean(delta2_inv_depth_affine))
    print('Delta3 inv depth affine:', np.mean(delta3_inv_depth_affine))
    print('#######################')
    print('ill AbsRel inv depth affine:', np.mean(ill_abs_rel_inv_depth_affine))
    print('ill RMS inv depth affine:', np.mean(ill_rms_inv_depth_affine))
    print('ill Log10 inv depth affine:', np.mean(ill_log10_inv_depth_affine))
    print('ill Delta1 inv depth affine:', np.mean(ill_delta1_inv_depth_affine))
    print('ill Delta2 inv depth affine:', np.mean(ill_delta2_inv_depth_affine))
    print('ill Delta3 inv depth affine:', np.mean(ill_delta3_inv_depth_affine))
    print('#######################')
    print('nill AbsRel inv depth affine:', np.mean(nill_abs_rel_inv_depth_affine))
    print('nill RMS inv depth affine:', np.mean(nill_rms_inv_depth_affine))
    print('nill Log10 inv depth affine:', np.mean(nill_log10_inv_depth_affine))
    print('nill Delta1 inv depth affine:', np.mean(nill_delta1_inv_depth_affine))
    print('nill Delta2 inv depth affine:', np.mean(nill_delta2_inv_depth_affine))
    print('nill Delta3 inv depth affine:', np.mean(nill_delta3_inv_depth_affine))
    print('#######################')


    stats['mean_metric_summary'] = {metric:eval('np.mean('+metric+').item()') for metric in metric_list}

    # save stats
    import json
    with open(os.path.join(args.output_path.replace('/pcd',''), 'stats_metric&affine.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    

