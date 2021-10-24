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

video_ext = '.avi'
image_ext = '.jpg'
image_files = image_path + '*' + image_ext


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
#     df_full['time'] = df_tmp['time']
#     df_full['id'] = df_tmp2['id']
#     df_full.set_index('image_name')


# using annotations:
print("Loading annotations...")
data = json.load((open(annotations_file)))
df_json = pd.json_normalize(data, 'annotations').sort_values(by=['image_name'])
df_bbox = pd.json_normalize(
    data, ['annotations', 'bbox'])
print(len(df_json), len(df_bbox))
print(df_bbox)
print(
    f'Number for annotation duplicates is {df_json.image_name.duplicated().sum()}')

# df_json_date = df_json[df_json['image_name'].str.contains(search_date)]

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
# year = df_json['time.year']
# month = df_json['time.month']
# day = df_json['time.day']
# min = df_json['time.min']
# sec = df_json['time.sec']
# msec = df_json['time.ms']
# print(year.values)
# df_json['datetime'] = pd.Timestamp(
#     str(year.values) + '-' + str(month.values) + '-' + str(day.values))
# # 'time.year', 'time.month', 'time.day', 'time.hour', 'time.min', 'time.sec')

# print(df_json['datetime'])


df_full = pd.merge(df_json, df_imgs, how='inner', on='image_name')
df_grp = df_full.groupby(
    by=['time.year', 'time.month', 'time.day', 'time.hour', 'time.min', 'time.sec'])
print("Date grouping: /n")
print(df_grp.size())

unique_dates = unique(df_full['time'])
video_counts = len(unique_dates)
print(f"There are {video_counts} video recordings:")
print(unique_dates)


# do for loop here on date
# for i in range(0, video_counts):
for i in range(0, video_counts):
    search_date = unique_dates[i]
    print('####### START #########')
    print(f'Video # {i + 1} / {video_counts}')
    print(f'Selecting date: {search_date}')
    df_date = df_full[df_full['image_name'].str.contains(search_date)]
    images = pd.Series(df_date['full_path'].values,
                       index=df_date['full_path'])

# Calculate Frames Per Second
    timespan_ms = df_date['time.ms'].iloc[-1] - df_date['time.ms'].iloc[0]
    timespan_sec = timespan_ms/1000
    print(
        f'Recording duration is {timespan_sec} sec')
    fps = len(df_date)/timespan_sec
    print(f'fps: {fps}')

    if(make_video):
        video_name = 'video_' + search_date
        if os.path.exists(video_name + video_ext):
            os.remove(video_name + video_ext)

        CreateVideo(images, fps, video_name, video_ext, 2000)

    print('######## END ##########')
