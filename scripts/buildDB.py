##Build Video database from xxx dictionary

import os
import csv

for i in range(11):
    dict_path = f'/data2/videos/youtube/video{i}'
    csv_file = f'/data2/videos/youtube/video{i}.csv'


    List = os.listdir(dict_path)
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['video_name', 'segmentation_processed','depth'])
        for video_file in List:
            writer.writerow([video_file, 0, 0])

    print(f"CSV file saved as {csv_file}")