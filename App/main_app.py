import ctypes
from tf_yolov4.preprocess.save_model import save_tf
from app_utils.bbox_utils import Evaluate
from app_utils.video_utils import VideoUtils
from app_utils.velocity_utils import VelocityUtils
from app_utils.image_utils import ImageUtils
from deepsort_yolo import DeepsortYolo
import cv2
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from absl import app, flags, logging
from absl.flags import FLAGS
import logging
import pathlib
import threading
import time
import os
# from app_utils.folder_utils import FolderUtils
from shutil import copy2
from sys import path
from yaml import SafeLoader, load
from PIL import ImageFont
import matplotlib.pyplot as plt
import sys
import traceback

# class TracePrints(object):
#   def __init__(self):
#     self.stdout = sys.stdout
#   def write(self, s):
#     self.stdout.write("Writing %r\n" % s)
#     traceback.print_stack(file=self.stdout)
# sys.stdout = TracePrints()


flags.DEFINE_integer('starttime', 0, 'Start time of video [sec]')
flags.DEFINE_integer('duration', -1, 'Duration time of video [sec], -1 = full video')
flags.DEFINE_string('output_dir', './output', 'Path to output directory')
flags.DEFINE_integer('select_id', -1, 'Object ID to follow')
flags.DEFINE_bool('process_video', False, 'does video need reprocessing or use csv if existing')
flags.DEFINE_float('cam_tilt', 30, 'Camera tilt angle in deg from horizon')
flags.DEFINE_float('cam_height', 15, 'Camera hight above object [m]')
flags.DEFINE_float('cam_fov', 41.1, 'Camera field of view deg')
flags.DEFINE_float('cam_focal', -1, 'Camera focal length [m]')
flags.DEFINE_float('vert_image_dim', -1, 'Vertical dimension of 35 mm image format')
flags.DEFINE_float('vel_unit_scale', 2.23694, 'scale to convert velocity units')

flags.DEFINE_bool('create_vel_video', True, 'Disables the creation of a velocity video')

vUtils = VideoUtils()
deepsortYolo = DeepsortYolo()
velUtils = VelocityUtils()


def GetVideoDuration(src_video, start_time, duration):
    video_in = cv2.VideoCapture(src_video)
    if video_in.isOpened():
        fps, total_frames, frame_size = VideoUtils.GetVideoData(video_in)
        start_count, end_count = VideoUtils.GetStartEndCount(fps, total_frames, start_time, duration)
        video_duration = int((end_count-start_count) / fps)
        video_in.release()
    return video_duration


def SaveWeightsYoloV4(output_dir, weights):
    if not os.path.exists(output_dir):
        save_tf(output_dir, weights)


def ProcessVideo(src_video, video_dest_path, model_dir, starttime, duration):
    processed_postfix = 'yolo_'
    logging.info(f"Processing video {src_video}")
    video_duration = GetVideoDuration(src_video, starttime, duration)
    print(video_duration)
    basename = pathlib.Path(src_video).stem + '_'
    filename = os.path.join(video_dest_path, basename + VideoUtils.AddTimestampToName(processed_postfix, starttime, video_duration))
    print(filename)
    tracker_file_csv = filename + '.csv'
    output_video_file = filename + pathlib.Path(src_video).suffix
    print(tracker_file_csv)
    if (not os.path.exists(tracker_file_csv)) or (FLAGS.process_video):
        model_filename = 'yolo_params/mars-small128.pb'
        if os.path.exists(output_video_file):
            os.remove(output_video_file)

        logging.info(f"Start processing video...")
        output_video_file, trk_bbox = deepsortYolo.ProcessVideo(model_dir, model_filename, src_video,
                                                                video_dest_path, output_video_file, starttime, video_duration, save_images=False)

        print(trk_bbox.head(10))
        trk_bbox.to_csv(tracker_file_csv, index=False)
        logging.info(f"Done processing video")

    return output_video_file, tracker_file_csv


def GetFps(src_video):
    success = False
    video_in = cv2.VideoCapture(src_video)
    fps = 0
    if video_in.isOpened():
        fps, total_frames, frame_size = VideoUtils.GetVideoData(video_in)
        video_in.release()
        success = True
    else:
        logging.warn(f"Can't open video {src_video}")

    return success, fps, total_frames, frame_size


def AddVelocitiesToVideo(src_video, video_dest_path, bb_file, select_id, starttime, duration):
    if (not src_video):
        raise Exception(f"File not found: {src_video}")

    logging.info(f"Start adding velocity")
    processed_postfix = 'vel_'
    video_duration = GetVideoDuration(src_video, starttime, duration)

    # extract the file name and extension
    basename = pathlib.Path(src_video).stem + '_'
    filename = os.path.join(video_dest_path, basename + VideoUtils.AddTimestampToName(processed_postfix, starttime, video_duration))
    output_video_file = filename + pathlib.Path(src_video).suffix
    df = pd.read_csv(bb_file)
    df = df.reset_index()
    df = df.drop(columns=['index'])

    # Open original video
    video_in = cv2.VideoCapture(src_video)
    if video_in.isOpened():
        fps, total_frames, frame_size = VideoUtils.GetVideoData(video_in)
        scale = velUtils.CalculateScale(df)
        print(f"scale from average:  {scale}")

        scale = velUtils.CalculateScaleCameraProperites(
            camera_tilt_angle_deg=FLAGS.cam_tilt, cam_height=FLAGS.cam_height, image_height=frame_size[1],
            cam_fov_deg=FLAGS.cam_fov, cam_focal_length=FLAGS.cam_focal, vert_image_dim=FLAGS.vert_image_dim)
        print(f"scale from camera properties:  {scale}")

        for id in df['object_id'].unique():
            df = velUtils.AddVelocity(df, id, fps, scale, frame_size)

        df.to_csv(bb_file, index=False)
        print(f"updated file: {bb_file}")
        if(select_id >= 0):
            df_sub = pd.DataFrame(df[(df['object_id'] == select_id)])
            object_csv_filename = os.path.splitext(bb_file)[0] + "ID-" + str(select_id) + '.csv'
            df_sub.to_csv(object_csv_filename, index=False)
            print(f"Created file: {object_csv_filename}")

        start_count, end_count = VideoUtils.GetStartEndCount(fps, total_frames, starttime, video_duration)
        count = start_count
        fourcc = cv2.VideoWriter_fourcc(*"XVID")

        video_in.set(cv2.CAP_PROP_POS_FRAMES, count)
        i = 0

        if(FLAGS.create_vel_video):
            video_out = cv2.VideoWriter(output_video_file, fourcc, int(fps), frame_size)
            while (True):
                success, img = video_in.read()

                if(success):
                    if((count % 25) == 0):
                        percent_complete = ((count-start_count)/(end_count-start_count))*100
                        logging.info(f"Created frame id {count:2d}, {count/fps:0.2f} sec in video; completed:  {percent_complete:0.1f} %")

                    # All bounding box info for this frame
                    # Add all annotations to the frame
                    df_img = df[df['Frame'] == count]

                    for i in range(len(df_img)):
                        background_color = ImageUtils.ColorGenerator(i)
                        text_color = (255, 255, 255)
                        text_color = (0, 255, 255)
                        x = int(df_img.iloc[i]["center_x"])
                        y = int(df_img.iloc[i]['center_y'])
                        category = df_img.iloc[i]['category']
                        object_id = df_img.iloc[i]['object_id']
                        txt1 = str(category).upper() + "-" + str(int(object_id))

                        txt2 = ""

                        velocity = round(df_img.iloc[i]["adj_vel"] * FLAGS.vel_unit_scale,1)
                        filt_velocity = round(df_img.iloc[i]["filt_vel"] * FLAGS.vel_unit_scale,1)
                        vel_display_limit = 1
                        if((filt_velocity > vel_display_limit) and (velocity > vel_display_limit)):
                            txt2 = 'Vel: ' + str(velocity) + " / " + str(filt_velocity)

                        fontFace = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 0.5
                        thickness = 2
                        size_txt1 = cv2.getTextSize(txt1, fontFace, fontScale, thickness)
                        size_txt2 = cv2.getTextSize(txt2, fontFace, fontScale, thickness)

                        _x11 = x + 10
                        _y11 = y
                        _x12 = _x11 + max(size_txt1[0][0], size_txt2[0][0])
                        _y12 = _y11 + (size_txt1[0][1] + size_txt2[0][1]) + 15
                        _x21 = _x11
                        _y21 = _y12 - 5
                        # cv2.rectangle(img, (_x11, _y11), (_x12, _y12), background_color, cv2.FILLED)
                        cv2.putText(img, txt1, (_x11, (_y11+15)), fontFace, fontScale, text_color, thickness)
                        cv2.putText(img, txt2, (_x21, _y21), fontFace, fontScale, text_color, thickness)
                        cv2.circle(img, (x, y), 4, (0, 0, 255), -1)
                        if (select_id == object_id):
                            ImageUtils.DrawBbox(img, df_img.iloc[i]["bb_left"],df_img.iloc[i]["bb_top"],df_img.iloc[i]["bb_right"],df_img.iloc[i]["bb_bottom"], text_color)


                    video_out.write(img)

                    count += 1
                    if(count > end_count):
                        break
                else:
                    break

            video_out.release()            
            
        else:
            output_video_file = ""


    video_in.release()  # done with original video
    logging.info(f"Done adding velocity")
    cv2.destroyAllWindows()
    return output_video_file, df


def main(_argv):
    start_time = time.time()
    model_dir = './checkpoints/yolov4'  # 'path to output'
    weights = './yolo_params/yolov4.weights'  # 'path to weights file'
    # top directory where the dataset is located
    video_name = 'VIRAT_S_050000_07_001014_001126'
    src_video = '../sample_datasets/VIRAT/' + video_name + '/' + video_name + '.mp4'

    src_video = './src_videos/VIRAT_S_050000_07_001014_001126.mp4'
    src_video = './src_videos/video_20190905091750_1.mp4'
    
    video_dest_path = FLAGS.output_dir  # location where to place processed videos/data
    video_start = FLAGS.starttime
    video_duration = FLAGS.duration
    print("App started")
    print(f"selected_id: {FLAGS.select_id}")
    success, fps, total_frames, frame_size = GetFps(src_video)
    print(f"fps: {fps}, total_frames: {total_frames}, frame_size: {frame_size}, video_start: {video_start}, video_duration: {video_duration}")
    if(not success):
        print(f"Can't open video {src_video}")

    SaveWeightsYoloV4(model_dir, weights)
    video_out, bb_file = ProcessVideo(src_video, video_dest_path, model_dir, video_start, video_duration)
    print(f"Storing bounding boxes in: {bb_file}")
    print(f"Created video: {video_out}")

    video_out, df = AddVelocitiesToVideo(src_video, video_dest_path, bb_file, FLAGS.select_id, video_start, video_duration)
    print(f"Created video: {video_out}")

    df_sub = pd.DataFrame(df[(df['object_id'] == FLAGS.select_id)])
    print(df_sub.tail(50))

    # for id in df['object_id'].unique():
    #     mask = (df['object_id'] == id)
    #     sub_df = df[mask]
    #     # plt.plot(sub_df["Frame"], sub_df["vel"])
        
    #     if (sub_df[(sub_df['category'] == "person")]):
    #         plt.plot(sub_df["Frame"], sub_df["filt_vel"])
        
    # plt.show()

    duration = time.time() - start_time
    print(f"Completed {duration} seconds")
    # cnt = 0
    # sleep_sec = 2
    # while True:
    #     time.sleep(sleep_sec)
    #     print(cnt)
    #     cnt += sleep_sec


if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")

    logging.getLogger().setLevel(logging.DEBUG)

    app.run(main)
