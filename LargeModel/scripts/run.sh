#!/bin/bash

# frames_root="/data2/Fooling3D/video_frame_sequence"
frames_root="./cache/video_frame_sequence"
videos_root="/data2/Fooling3D/videos"
# meta_root="/data2/Fooling3D/meta_data"
meta_root="./cache"
max_workers=6
num_producers=1
step=1
DEBUG_ENVS=False

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
LINES_PER_GPU=$(( (NUM_LINES + NUM_GPUS - 1) / NUM_GPUS ))

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
    PAIR=$1
    frames_root=$2
    videos_root=$3
    max_workers=$4
    num_producers=$5
    meta_root=$6
    step=$7
    DEBUG_ENVS=$8

    # Decouple PART_FILE and GPU_ID from PAIR
    IFS=',' read -r PART_FILE GPU_ID <<< "$PAIR"

    echo "Processing $PART_FILE on GPU $GPU_ID"
    DEBUG_ENVS=$DEBUG_ENVS CUDA_VISIBLE_DEVICES=$GPU_ID python filter_bad_frames.py --video_path_csv "$PART_FILE" --gpu_id "$GPU_ID" --max_queue_size 10 --frames_root "$frames_root" --videos_root "$videos_root" --max_workers "$max_workers" --num_producers "$num_producers" --meta_root "$meta_root" --step "$step"
}

# Export the function so it can be used by parallel
export -f process_part

# Get the list of part files
PART_FILES=$(ls ./cache/part_*)

echo "NUM_GPUS: ${NUM_GPUS}, GPU_IDS: '${GPU_IDS[@]}', PART_FILES: '${PART_FILES}', NUM_LINES: ${NUM_LINES}, LINES_PER_GPU: ${LINES_PER_GPU}"
echo "Frames root: $frames_root, Videos root: $videos_root, Max workers: $max_workers, Num producers: $num_producers, Meta root: $meta_root, Step: $step"

# Create pairs of part files and GPU IDs (one-to-one mapping)
PAIRS=()
i=0
for PART_FILE in $PART_FILES; do
    GPU_ID=${GPU_IDS[$i % NUM_GPUS]}
    PAIRS+=("$PART_FILE,$GPU_ID")
    i=$((i + 1))
done

# Process each pair in parallel
parallel --jobs $NUM_GPUS --line-buffer --no-notice process_part {1} {2} {3} {4} {5} {6} {7} {8} ::: "${PAIRS[@]}" ::: $frames_root ::: $videos_root ::: $max_workers ::: $num_producers ::: $meta_root ::: $step ::: $DEBUG_ENVS




# Clean up the part files
rm ./cache/part_*