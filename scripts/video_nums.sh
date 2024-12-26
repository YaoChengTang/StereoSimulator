#!/bin/bash

# 遍历文件夹 xx/xxx/xx/xx{1..10}
sum=0
for i in {0..10}; do
  # 拼接路径
  folder="/data2/videos/youtube/video$i"
  
  # 检查文件夹是否存在
  if [ -d "$folder" ]; then
    # 进入文件夹并统计文件数
    file_count=$(find "$folder" -maxdepth 1 -type f | wc -l)
    sum=$((sum + $file_count))
    echo "Folder: $folder, File Count: $file_count"
  else
    echo "Folder: $folder does not exist."
  fi
done
echo "sum: $sum"