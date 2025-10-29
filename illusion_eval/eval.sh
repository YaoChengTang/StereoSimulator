export PATH=/usr/local/cuda-12.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH
export data_path=/data2/Fooling3D/real/test/left
export gt_path=/data2/Fooling3D/real/test/disp
export pred_path=/data3/jiaxi/workspace_bak/workspace/StereoSimulator/eval_data_fooling3d
export output_path=/data3/jiaxi/workspace_bak/workspace/StereoSimulator/eval_data_fooling3d1
# # # # # # # # # STEREO EVAL # # # # # # # # # # #
# # StereoAnywhere
CUDA_VISIBLE_DEVICES=5 python eval.py \
    --data-path $data_path \
    --pred-path $pred_path/stereoanywhere/depth \
    --gt-path $gt_path \
    --pred_type disp \
    --resize-scale 0.5 \
    --output-path $output_path/stereoanywhere/pcd \
    # --vis

# # StereoAnything
CUDA_VISIBLE_DEVICES=5 python eval.py \
    --data-path $data_path \
    --pred-path $pred_path/StereoAnything/depth \
    --gt-path $gt_path \
    --pred_type disp \
    --resize-scale 0.5 \
    --output-path $output_path/StereoAnything/pcd \
    # --vis

# Monster
CUDA_VISIBLE_DEVICES=5 python eval.py \
    --data-path $data_path \
    --pred-path $pred_path/Monster/depth \
    --gt-path $gt_path \
    --pred_type disp \
    --resize-scale 0.5 \
    --output-path $output_path/Monster/pcd \
    # --vis

# mocha stereo
CUDA_VISIBLE_DEVICES=5 python eval.py \
    --data-path $data_path \
    --pred-path $pred_path/Mocha-Stereo/depth \
    --gt-path $gt_path \
    --pred_type disp \
    --resize-scale 0.5 \
    --output-path $output_path/Mocha-Stereo/pcd \
    # --vis

# selective raft
CUDA_VISIBLE_DEVICES=5 python eval.py \
    --data-path $data_path \
    --pred-path $pred_path/Selective-RAFT/depth \
    --gt-path $gt_path \
    --pred_type disp \
    --resize-scale 0.5 \
    --output-path $output_path/Selective-RAFT/pcd \
    # --vis


# selective igev
CUDA_VISIBLE_DEVICES=5 python eval.py \
    --data-path $data_path \
    --pred-path $pred_path/Selective-IGEV/depth \
    --gt-path $gt_path \
    --pred_type disp \
    --resize-scale 0.5 \
    --output-path $output_path/Selective-IGEV/pcd \
    # --vis


# Raftstereo
CUDA_VISIBLE_DEVICES=5 python eval.py \
    --data-path $data_path \
    --pred-path $pred_path/RAFT-Stereo/depth \
    --gt-path $gt_path \
    --pred_type disp \
    --resize-scale 0.5 \
    --output-path $output_path/RAFT-Stereo/pcd \
    # --vis

#ours
CUDA_VISIBLE_DEVICES=5 python eval.py \
    --data-path $data_path \
    --pred-path $pred_path/ours_new/depth \
    --gt-path $gt_path \
    --pred_type disp \
    --resize-scale 0.5 \
    --output-path $output_path/ours_new/pcd \
    # --vis

# # # # # # # # 3R WITH 2-VIEW EVAL # # # # # # # # # # # # # 
# dust3r
CUDA_VISIBLE_DEVICES=5 python eval.py \
    --data-path $data_path \
    --pred-path $pred_path/dust3r/depth  \
    --gt-path $gt_path \
    --pred_type depth \
    --resize-scale 0.5 \
    --scale-align \
    --output-path $output_path/dust3r/pcd \
    # --vis

# vggt
CUDA_VISIBLE_DEVICES=5 python eval.py \
    --data-path $data_path \
    --pred-path $pred_path/VGGT/depth  \
    --gt-path $gt_path \
    --pred_type depth \
    --resize-scale 0.5 \
    --scale-align \
    --output-path $output_path/VGGT/pcd \
    # --vis

# # # # # # # # # METRIC MONO (SCALE INVARIANT) EVAL # # # # # # # # # # #
# depth-pro
CUDA_VISIBLE_DEVICES=5 python eval.py \
    --data-path $data_path \
    --pred-path $pred_path/depth-pro/depth \
    --gt-path $gt_path \
    --pred_type depth \
    --resize-scale 0.5 \
    --output-path $output_path/depth-pro/pcd \
    # --vis

# metric3d v2
CUDA_VISIBLE_DEVICES=5 python eval.py \
    --data-path $data_path \
    --pred-path $pred_path/metric3d/depth \
    --gt-path $gt_path \
    --pred_type depth \
    --resize-scale 0.5 \
    --output-path $output_path/metric3d/pcd \
    # --vis 

# depthanythingv2 metric
CUDA_VISIBLE_DEVICES=5 python eval.py \
    --data-path $data_path \
    --pred-path $pred_path/DAV2_metric/depth \
    --gt-path $gt_path \
    --pred_type depth \
    --resize-scale 0.5 \
    --output-path $output_path/DAV2_metric/pcd \
    # --vis

# # # # # # # # # METRIC MONO (SCALE INVARIANT) + SCALE ALIGN EVAL # # # # # # # # # # #

# depth-pro-scale-align
CUDA_VISIBLE_DEVICES=5 python eval.py \
    --data-path $data_path \
    --pred-path $pred_path/depth-pro/depth \
    --gt-path $gt_path \
    --pred_type depth \
    --resize-scale 0.5 \
    --scale-align \
    --output-path $output_path/depth-pro_scale_align/pcd \
    # --vis

# metric3d-v2-scale-align
CUDA_VISIBLE_DEVICES=5 python eval.py \
    --data-path $data_path \
    --pred-path $pred_path/metric3d/depth \
    --gt-path $gt_path \
    --pred_type depth \
    --resize-scale 0.5 \
    --scale-align \
    --output-path $output_path/metric3d_scale_align/pcd \
    # --vis 

# depthanythingv2-metric-scale-align
CUDA_VISIBLE_DEVICES=5 python eval.py \
    --data-path $data_path \
    --pred-path $pred_path/DAV2_metric/depth \
    --gt-path $gt_path \
    --pred_type depth \
    --scale-align \
    --resize-scale 0.5 \
    --output-path $output_path/DAV2_metric_scale_align/pcd \
    # --vis


# # # # # # # # # # AFFINE INVARIANT MONO EVAL # # # # # # # # # # #
# # marigold
CUDA_VISIBLE_DEVICES=3 python eval.py \
    --data-path $data_path \
    --pred-path $pred_path/marigold_new_10/depth_npy \
    --gt-path $gt_path \
    --pred_type depth \
    --resize-scale 0.5 \
    --output-path $output_path/marigold_new_10/pcd \
    --scale-shift-align \
    # --vis

# # depthanythingv2 gt aligin
CUDA_VISIBLE_DEVICES=5 python eval.py \
    --data-path $data_path \
    --pred-path $pred_path/DAV2/depth \
    --gt-path $gt_path \
    --pred_type inv_depth \
    --resize-scale 0.5 \
    --output-path $output_path/DAV2/pcd \
    --scale-shift-align \
    # --vis


############################OURS ABLATION########################################

# CUDA_VISIBLE_DEVICES=5 python eval.py \
#     --data-path $data_path \
#     --pred-path $pred_path/ABLATION/Fooling3D_BetaConf2/depth \
#     --gt-path $gt_path \
#     --pred_type disp \
#     --resize-scale 0.5 \
#     --output-path $output_path/ABLATION/Fooling3D_BetaConf2/pcd \
#     # --vis

# CUDA_VISIBLE_DEVICES=5 python eval.py \
#     --data-path $data_path \
#     --pred-path $pred_path/ABLATION/Fooling3D_ME_AdaptivePostFusion/depth \
#     --gt-path $gt_path \
#     --pred_type disp \
#     --resize-scale 0.5 \
#     --output-path $output_path/ABLATION/Fooling3D_ME_AdaptivePostFusion/pcd \
#     # --vis

# CUDA_VISIBLE_DEVICES=5 python eval.py \
#     --data-path $data_path \
#     --pred-path $pred_path/ABLATION/Fooling3D_ME_AdaptivePostFusion2/depth \
#     --gt-path $gt_path \
#     --pred_type disp \
#     --resize-scale 0.5 \
#     --output-path $output_path/ABLATION/Fooling3D_ME_AdaptivePostFusion2/pcd \
#     # --vis

# CUDA_VISIBLE_DEVICES=5 python eval.py \
#     --data-path $data_path \
#     --pred-path $pred_path/ABLATION/Fooling3D_ME_AdaptiveSingle/depth \
#     --gt-path $gt_path \
#     --pred_type disp \
#     --resize-scale 0.5 \
#     --output-path $output_path/ABLATION/Fooling3D_ME_AdaptiveSingle/pcd \
#     # --vis

# CUDA_VISIBLE_DEVICES=5 python eval.py \
#     --data-path $data_path \
#     --pred-path $pred_path/ABLATION/Fooling3D_ME_AdaptiveSingle2/depth \
#     --gt-path $gt_path \
#     --pred_type disp \
#     --resize-scale 0.5 \
#     --output-path $output_path/ABLATION/Fooling3D_ME_AdaptiveSingle2/pcd \
#     # --vis

# CUDA_VISIBLE_DEVICES=5 python eval.py \
#     --data-path $data_path \
#     --pred-path $pred_path/ABLATION/Fooling3D_ME_PostFusion/depth \
#     --gt-path $gt_path \
#     --pred_type disp \
#     --resize-scale 0.5 \
#     --output-path $output_path/ABLATION/Fooling3D_ME_PostFusion/pcd \
#     # --vis

# CUDA_VISIBLE_DEVICES=5 python eval.py \
#     --data-path $data_path \
#     --pred-path $pred_path/ABLATION/Fooling3D_NoME_PostFusion/depth \
#     --gt-path $gt_path \
#     --pred_type disp \
#     --resize-scale 0.5 \
#     --output-path $output_path/ABLATION/Fooling3D_NoME_PostFusion/pcd \
#     # --vis

# CUDA_VISIBLE_DEVICES=5 python eval.py \
#     --data-path $data_path \
#     --pred-path $pred_path/ABLATION/Fooling3D_RaftStereo/depth \
#     --gt-path $gt_path \
#     --pred_type disp \
#     --resize-scale 0.5 \
#     --output-path $output_path/ABLATION/Fooling3D_RaftStereo/pcd \
#     # --vis