#!/bin/bash

# frames_root="/data2/Fooling3D/video_frame_sequence"
# frames_root="./cache/video_frame_sequence"
frames_root="/data2/Fooling3D/video_frame_sequence_beta"
videos_root="/data2/Fooling3D/videos"
# CSV_FILE="./cache/tmp_file.csv"    # Path to the CSV file containing the video paths
CSV_FILE="/data2/Fooling3D/meta_data/frames_metadata_beta.csv"    # Path to the CSV file containing the video paths

# meta_root="/data2/Fooling3D/meta_data"
meta_root="./cache"
# model_path="/mount_points/nas/Qwen2-VL-7B-Instruct"
# model_path="/mount_points/nas/Qwen2-VL-2B-Instruct"
model_path="/data5/yao/pretrained/Qwen2-VL-72B-Instruct-GPTQ-Int4"

max_step=100
num_clip=10
batch_size=1
num_workers=4                    # Number of workers for data loading
num_thread=10                     # Number of threads for saving frames

DEBUG_ENVS=False
# DEBUG_ENVS=True

GPU_ID="4"

# Check if exp_name is provided as an argument
if [ -z "$1" ]; then
    exp_name="EXP"  # Default value for exp_name
else
    exp_name="$1"
fi

export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/data5/yao/cache"

DEBUG_ENVS=$DEBUG_ENVS CUDA_VISIBLE_DEVICES=$GPU_ID python filter_bad_frames.py --video_path_csv "$CSV_FILE" --gpu_id "$GPU_ID" --batch_size "$batch_size" --frames_root "$frames_root" --videos_root "$videos_root" --num_workers "$num_workers" --num_thread "$num_thread" --meta_root "$meta_root" --max_step "$max_step" --model_path "$model_path" --exp_name "$exp_name" --num_clip "$num_clip"
