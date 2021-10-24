import cv2
import numpy as np
from numpy.core.numeric import True_
from numpy.lib.arraysetops import unique
import pandas as pd
import glob
import os
import json
import datetime

make_video = True
dataset_dir_path = '../DataSets/Video/AU-AIR-2019/'

annotations_path = dataset_dir_path + 'auair2019annotations/'
image_path = dataset_dir_path + 'auair2019data/images/'
annotations_file = annotations_path + 'annotations.json'
image_files = image_path + '*.jpg'
video_ext = '.avi'


def CollectingImages(count, total_cnt):
    print(f"Collecting {count} / {total_cnt} images...")
    img_array = []
    for i in range(count):
        img = cv2.imread(images.iloc[i])
        height, width, layers = img.shape
        image_size = (width, height)
        img_array.append(img)
        if((i % 200) == 0):
            print(str(round((i/count) * 100)) + ' %')

    return img_array, image_size


def CombineImages(img_array, video_name, fps, size):
    print("Combining images to: " + video_name)
    out = cv2.VideoWriter(
        video_name, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    image_count = len(img_array)
    for i in range(image_count):
        out.write(img_array[i])
        if((i % 200) == 0):
            print(str(round((i/image_count) * 100)) + ' %')

    out.release()
    print("Created video: " + video_name)


def CreateVideo(images, fps, video_name, video_ext, video_frame_limit=3000):
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

        img_array, img_size = CollectingImages(image_count, total_cnt)

        if(split_video):
            full_video_name = video_name + '_' + str(video_idx) + video_ext
        else:
            full_video_name = video_name + video_ext

        if os.path.exists(full_video_name):
            os.remove(full_video_name)

        CombineImages(img_array, full_video_name, fps, img_size)

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
print(
    f'Number for annotation duplicates is {df_ann.image_name.duplicated().sum()}')

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
print(
    f'Number for image duplicates is {df_imgs.image_name.duplicated().sum()}')

# df_imgs = df_imgs.sort_values(by=['time', 'id']).drop_duplicates(
#     subset=['time', 'id'])
# year = df_ann['time.year']
# month = df_ann['time.month']
# day = df_ann['time.day']
# min = df_ann['time.min']
# sec = df_ann['time.sec']
# msec = df_ann['time.ms']
# print(year.values)
# df_ann['datetime'] = pd.Timestamp(
#     str(year.values) + '-' + str(month.values) + '-' + str(day.values))
# # 'time.year', 'time.month', 'time.day', 'time.hour', 'time.min', 'time.sec')

# print(df_ann['datetime'])


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
for i in range(0, 1):
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
        if os.path.exists(video_name + video_ext):
            os.remove(video_name + video_ext)

        CreateVideo(images, fps, video_name, video_ext, 3000)

    print('######## END ##########')
