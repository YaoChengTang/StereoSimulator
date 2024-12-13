##Build Video database from xxx dictionary

import os
import csv
dict_path = '/data4/lzd/iccv25/data/video_sets1'
csv_file = '/data4/lzd/iccv25/data/video_sets1.csv'


List = os.listdir(dict_path)
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['video_name', 'segmentation_processed'])
    for video_file in List:
        writer.writerow([video_file, 0])

print(f"CSV file saved as {csv_file}")