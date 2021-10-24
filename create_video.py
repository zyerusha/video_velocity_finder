import cv2
import numpy as np
from numpy.lib.arraysetops import unique
import pandas as pd
import glob
import os
import json
import datetime

make_video = False
dataset_dir_path = '../DataSets/Video/AU-AIR-2019/'

annotations_path = dataset_dir_path + 'auair2019annotations/'
image_path = dataset_dir_path + 'auair2019data/images/'
annotations_file = annotations_path + 'annotations.json'
image_files = image_path + '*.jpg'


def CreateVideo(images, fps, video_name, video_frame_limit=3000):
    total_cnt = len(images)

    print(f"Found {total_cnt} images for this video recording.")
    image_count = total_cnt
    remaining_count = total_cnt
    video_idx = 0
    split_video = False
    while(remaining_count > 0):
        # limit size of videos
        if(remaining_count > video_frame_limit):
            image_count = video_frame_limit
            split_video = True
        else:
            image_count = remaining_count

        video_idx += 1

        remaining_count -= max(image_count, 0)  # keep it positive

        print(f"Collecting {image_count} / {total_cnt} images...")
        img_array = []
        for i in range(image_count):
            img = cv2.imread(images.iloc[i])
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
            if((i % 200) == 0):
                print(str(round((i/image_count) * 100)) + ' %')

        if(split_video):
            full_video_name = video_name + '_' + str(video_idx) + '.avi'
        else:
            full_video_name = video_name + '.avi'

        if os.path.exists(full_video_name):
            os.remove(full_video_name)

        print("Combining images to: " + full_video_name)
        out = cv2.VideoWriter(
            full_video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

        image_count = len(img_array)
        for i in range(image_count):
            out.write(img_array[i])
            if((i % 200) == 0):
                print(str(round((i/image_count) * 100)) + ' %')

        out.release()
        print("Created video: " + full_video_name)


# def AddDateTime(names):
#     s = pd.Series(names.str.split('_'))
#     df_tmp = pd.DataFrame(s.tolist(), columns=['head', 'time', 'x', 'id.jpg'])
#     s = pd.Series(df_tmp['id.jpg'].str.split('.'))
#     df_tmp2 = pd.DataFrame(s.tolist(), columns=['id', 'jpg'])
#     df['time'] = df_tmp['time']
#     df['id'] = df_tmp2['id']
#     df.set_index('image_name')

# using annotations:
print("Loading annotations...")
data = json.load((open(annotations_file)))
df_ann = pd.json_normalize(data, 'annotations').sort_values(by=['image_name'])
# df_ann = pd.json_normalize(
#     data, ['annotations', 'bbox']).sort_values(by=['image_name'])
print(df_ann)

# df_ann_date = df_ann[df_ann['image_name'].str.contains(search_date)]

# using images:
print("Finding image names...")
files = np.array(glob.glob(image_files))
df_imgs = pd.concat([pd.DataFrame([os.path.basename(file)], columns=['image_name']) for file in files],
                    ignore_index=True)
df_imgs['full_path'] = files
s = pd.Series(df_imgs['image_name'].str.split('_'))
df_tmp = pd.DataFrame(s.tolist(), columns=['head', 'time', 'x', 'id.jpg'])
s = pd.Series(df_tmp['id.jpg'].str.split('.'))
df_tmp2 = pd.DataFrame(s.tolist(), columns=['id', 'jpg'])
df_imgs['time'] = df_tmp['time']
df_imgs['id'] = df_tmp2['id']
df_imgs.set_index('image_name')

# df_imgs = df_imgs.sort_values(by=['time', 'id']).drop_duplicates(
#     subset=['time', 'id'])

df_merge = pd.merge(df_ann, df_imgs, how='inner', on='image_name')


df_grp = df_merge.groupby(
    by=['time.year', 'time.month', 'time.day', 'time.hour', 'time.min', 'time.sec'])
print("Date grouping: /n")
print(df_grp.size())

unique_dates = unique(df_merge['time'])
video_counts = len(unique_dates)
print(f"There are {video_counts} video recordings:")
print(unique_dates)


# do for loop here on date
for i in range(0, video_counts):
    search_date = unique_dates[i]
    print('####### START #########')
    print(f'Video # {i + 1} / {video_counts}')
    print(f'Selecting date: {search_date}')
    df_merge_date = df_merge[df_merge['image_name'].str.contains(search_date)]
    images = pd.Series(df_merge_date['full_path'].values,
                       index=df_merge_date['full_path'])

    MSEC_2_SEC = 0.001
    # ((df_merge_date['time.ms'].max() - df_merge_date['time.ms'].min())/len(df_merge_date)) * MSEC_2_SEC
    fps = 15
    print(f'fps: {fps}')
    if(make_video):
        video_name = 'video_' + search_date
        if os.path.exists(video_name):
            os.remove(video_name)

        CreateVideo(images, fps, video_name, 3000)

    print('######## END ##########')
