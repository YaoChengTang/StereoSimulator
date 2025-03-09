import json
import os

import numpy as np




def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


method = ['mine1','gt','DAV2', 'metric3d', 'Mocha-Stereo', 'raftstereo', 'DAV2_metric', 'depth-pro', 'marigold', 'selective-igev', 'selective-raft',  'dust3r']

root = '/home/smbu/jiaxi/workspace/StereoSimulator/eval_data'
type_list = ['disp_pred','depth_pred']

frame_list =['PaperOnWall_Vase/frame_0001.png',
             'PaperOnWall_StreetView/frame_0001.png',
             'Video_Objects/frame_0014.png',
             'Objects_FloralPainting/frame_0001.png',
             'PaperOnWall_Window1/frame_0001.png',
             'TransReflect_Showcase3/frame_0005.png',
             'TransReflect_Showcase3/frame_0006.png',
             'TransReflect_Window1/frame_0005.png',
             'PaperOnWall_Apple/frame_0001.png'
             ]
new_root = '/home/smbu/jiaxi/workspace/StereoSimulator/eval_data/final_vis'
for f in frame_list:
    new_dir = os.path.join(new_root, f.split('/')[0])
    os.makedirs(new_dir, exist_ok=True)
    for m in method:
        for t in type_list:
            if m =='marigold' and t == 'depth_pred':
                t = 'depth_colored'
                src = os.path.join(root, m, t, f.replace('.png', '_colored.png'))
            else:
                src=os.path.join(root, m, t, f)
            if m =='gt':
                src = os.path.join(root, 'gt', t.replace('_pred', '_gt'), f)

            if not os.path.exists(src):
                print(f'{src} not exist')
                continue
            t= 'depth' if 'depth' in t else 'disp'
           
            tar = os.path.join(new_dir, f.split('/')[1].replace('.png', f'_{m}_{t}.png'))
            os.system(f"cp {src} {tar}")
        pcd_f=f.replace('.png', '.ply')
        src = os.path.join(root, m, 'pcd', pcd_f)
        tar = os.path.join(new_dir, pcd_f.split('/')[1].replace('.ply', f'_{m}_{t}.ply'))
        os.system(f"cp {src} {tar}")
        