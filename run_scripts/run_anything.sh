#!/bin/bash

cd /data4/lzd/iccv25/code/

# 创建日志目录
LOG_DIR="./logs"
mkdir -p $LOG_DIR

# 启动并行任务
for i in {4..7}; do
    CUDA_VISIBLE_DEVICES=$(($i)) python prepare_depth_depthAnything.py --idx $i > $LOG_DIR/task_$i.log 2>&1 &
    echo "Task $i started with CUDA_VISIBLE_DEVICES=$(($i)), logging to $LOG_DIR/task_$i.log"
done

# 等待所有任务完成
wait
echo "All tasks completed."