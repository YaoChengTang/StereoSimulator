#!/bin/bash

cd /data4/lzd/iccv25/code/

# 创建日志目录
LOG_DIR="./logs"
mkdir -p $LOG_DIR

# 启动并行任务
for i in {5..6}; do
    # 根据任务索引分配 GPU
    if [[ $i -lt 2 ]]; then
        GPU_ID=5
    else
        GPU_ID=5
    fi

    # 启动任务并记录日志
    CUDA_VISIBLE_DEVICES=$GPU_ID python prepare_depth_depthAnything.py --idx $i > $LOG_DIR/task_$i.log 2>&1 &
    echo "Task $i started with CUDA_VISIBLE_DEVICES=$GPU_ID, logging to $LOG_DIR/task_$i.log"
done

# 等待所有任务完成
wait
echo "All tasks completed."