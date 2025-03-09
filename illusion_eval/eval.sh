export PATH=/usr/local/cuda-12.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH
# export CUDA_VISIBLE_DEVICES=5

# # dust3r
# CUDA_VISIBLE_DEVICES=5 python eval.py \
#     --data-path /data2/Fooling3D/real_data/testing/left \
#     --pred-path /home/smbu/jiaxi/workspace/StereoSimulator/eval_data/dust3r/depth  \
#     --gt-path /data2/Fooling3D/real_data/testing/disp \
#     --pred_type depth \
#     --resize-scale 0.5 \
#     --scale-shift-align \
#     --output-path /home/smbu/jiaxi/workspace/StereoSimulator/eval_data/dust3r/pcd \
#     --vis

# # mocha stereo
# CUDA_VISIBLE_DEVICES=5 python eval.py \
#     --data-path /data2/Fooling3D/real_data/testing/left \
#     --pred-path /data4/lzd/baseline/disp/Fooling3D/Mocha-Stereo \
#     --gt-path /data2/Fooling3D/real_data/testing/disp \
#     --pred_type disp \
#     --resize-scale 0.5 \
#     --output-path /home/smbu/jiaxi/workspace/StereoSimulator/eval_data/Mocha-Stereo/pcd \
#     --vis

# # selective raft
# CUDA_VISIBLE_DEVICES=5 python eval.py \
#     --data-path /data2/Fooling3D/real_data/testing/left \
#     --pred-path /data4/lzd/baseline/disp/Fooling3D/Selective-RAFT \
#     --gt-path /data2/Fooling3D/real_data/testing/disp \
#     --pred_type disp \
#     --resize-scale 0.5 \
#     --output-path /home/smbu/jiaxi/workspace/StereoSimulator/eval_data/selective-raft/pcd \
#     --vis


# # # selective igev
# CUDA_VISIBLE_DEVICES=5 python eval.py \
#     --data-path /data2/Fooling3D/real_data/testing/left \
#     --pred-path /data4/lzd/baseline/disp/Fooling3D/Selective-IGEV \
#     --gt-path /data2/Fooling3D/real_data/testing/disp \
#     --pred_type disp \
#     --resize-scale 0.5 \
#     --output-path /home/smbu/jiaxi/workspace/StereoSimulator/eval_data/selective-igev/pcd \
#     --vis


# # Raftstereo
# CUDA_VISIBLE_DEVICES=5 python eval.py \
#     --data-path /data2/Fooling3D/real_data/testing/left \
#     --pred-path /data4/lzd/baseline/disp/Fooling3D/RAFT-Stereo \
#     --gt-path /data2/Fooling3D/real_data/testing/disp \
#     --pred_type disp \
#     --resize-scale 0.5 \
#     --output-path /home/smbu/jiaxi/workspace/StereoSimulator/eval_data/raftstereo/pcd \
#     --vis


# #depth-pro
# CUDA_VISIBLE_DEVICES=5 python eval.py \
#     --data-path /data2/Fooling3D/real_data/testing/left \
#     --pred-path /home/smbu/jiaxi/workspace/StereoSimulator/eval_data/depth-pro/depth \
#     --gt-path /data2/Fooling3D/real_data/testing/disp \
#     --pred_type depth \
#     --resize-scale 0.5 \
#     --output-path /home/smbu/jiaxi/workspace/StereoSimulator/eval_data/depth-pro/pcd \
#     --vis

# # marigold
# CUDA_VISIBLE_DEVICES=5 python eval.py \
#     --data-path /data2/Fooling3D/real_data/testing/left \
#     --pred-path /home/smbu/jiaxi/workspace/StereoSimulator/eval_data/marigold/depth_npy \
#     --gt-path /data2/Fooling3D/real_data/testing/disp \
#     --pred_type depth \
#     --resize-scale 0.5 \
#     --output-path /home/smbu/jiaxi/workspace/StereoSimulator/eval_data/marigold/pcd \
#     --scale-shift-align \
#     --vis

# metric3d
# CUDA_VISIBLE_DEVICES=5 python eval.py \
#     --data-path /data2/Fooling3D/real_data/testing/left \
#     --pred-path /home/smbu/jiaxi/workspace/StereoSimulator/eval_data/metric3d/depth \
#     --gt-path /data2/Fooling3D/real_data/testing/disp \
#     --pred_type depth \
#     --resize-scale 0.5 \
#     --output-path /home/smbu/jiaxi/workspace/StereoSimulator/eval_data/metric3d/pcd \
#     --vis 

# # ours
CUDA_VISIBLE_DEVICES=5 python eval.py \
    --data-path /data2/Fooling3D/real_data/testing/left \
    --pred-path /home/smbu/jiaxi/workspace/StereoSimulator/eval_data/mine3/depth \
    --gt-path /data2/Fooling3D/real_data/testing/disp \
    --pred_type disp \
    --resize-scale 0.5 \
    --output-path /home/smbu/jiaxi/workspace/StereoSimulator/eval_data/mine3/pcd \
    --vis 

# # depthanythingv2 metric
# CUDA_VISIBLE_DEVICES=5 python eval.py \
#     --data-path /data2/Fooling3D/real_data/testing/left \
#     --pred-path /home/smbu/jiaxi/workspace/StereoSimulator/eval_data/DAV2_metric/depth \
#     --gt-path /data2/Fooling3D/real_data/testing/disp \
#     --pred_type depth \
#     --resize-scale 0.5 \
#     --output-path /home/smbu/jiaxi/workspace/StereoSimulator/eval_data/DAV2_metric/pcd \
#     --vis

# # depthanythingv2 gt aligin
# CUDA_VISIBLE_DEVICES=5 python eval.py \
#     --data-path /data2/Fooling3D/real_data/testing/left \
#     --pred-path /home/smbu/jiaxi/workspace/StereoSimulator/eval_data/DAV2/depth \
#     --gt-path /data2/Fooling3D/real_data/testing/disp \
#     --pred_type inv_depth \
#     --resize-scale 0.5 \
#     --output-path /home/smbu/jiaxi/workspace/StereoSimulator/eval_data/DAV2/pcd \
#     --scale-shift-align \
#     --vis