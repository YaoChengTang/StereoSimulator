#!/bin/bash

# frames_root="/data2/Fooling3D/video_frame_sequence"
frames_root="./cache/video_frame_sequence"
videos_root="/data2/Fooling3D/videos"
# meta_root="/data2/Fooling3D/meta_data"
meta_root="./cache"
max_workers=6
num_producers=1
step=1

# Check if the GPU IDs are provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 <gpu_ids>"
    exit 1
fi

# List of GPU IDs available
GPU_IDS=(${1//,/ })

# Number of GPUs available
NUM_GPUS=${#GPU_IDS[@]}

# Path to the CSV file
CSV_FILE="./cache/tmp_file.csv"

# Number of lines in the CSV file (excluding the header)
NUM_LINES=$(($(wc -l < "$CSV_FILE") - 1))

# Number of lines to process per GPU
LINES_PER_GPU=$((NUM_LINES / NUM_GPUS))

# Extract the header
HEADER=$(head -n 1 "$CSV_FILE")

# Split the CSV file into smaller files for each GPU, keeping the header
tail -n +2 "$CSV_FILE" | split -l $LINES_PER_GPU - ./cache/part_

# Add the header to each part file
for PART_FILE in ./cache/part_*; do
    (echo "$HEADER" && cat "$PART_FILE") > temp_file && mv temp_file "$PART_FILE"
done

# Function to process a part of the CSV file on a specific GPU
process_part() {
    PART_FILE=$1
    GPU_ID=$2
    frames_root=$3
    videos_root=$4
    max_workers=$5
    num_producers=$6
    meta_root=$7
    step=$8
    echo "Processing $PART_FILE on GPU $GPU_ID"
    CUDA_VISIBLE_DEVICES=$GPU_ID python filter_bad_frames.py --video_path_csv "$PART_FILE" --gpu_id "$GPU_ID" --max_queue_size 10 --frames_root "$frames_root" --videos_root "$videos_root" --max_workers "$max_workers" --num_producers "$num_producers" --meta_root "$meta_root" --step "$step"
}

# Export the function so it can be used by parallel
export -f process_part

# Get the list of part files
PART_FILES=$(ls ./cache/part_*)

echo "Frames root: $frames_root, Videos root: $videos_root, Max workers: $max_workers, Num producers: $num_producers, Meta root: $meta_root, Step: $step"

# Process each part file on a different GPU
parallel --jobs $NUM_GPUS process_part ::: $PART_FILES ::: "${GPU_IDS[@]}" ::: $frames_root ::: $videos_root ::: $max_workers ::: $num_producers ::: $meta_root ::: $step

# Clean up the part files
rm ./cache/part_*