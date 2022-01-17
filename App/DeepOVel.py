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
# from absl import app, flags, logging
# from absl.flags import FLAGS
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


class DeepOVel:
    vUtils = VideoUtils()
    deepsortYolo = DeepsortYolo()
    velUtils = VelocityUtils()
    # starttime = 0  # 'Start time of video [sec]'
    # duration = -1  # 'Duration time of video [sec], -1 = full video'
    select_id = -1  # 'Object ID to follow'
    process_video = False  # 'does video need reprocessing or use csv if existing'
    cam_tilt = 30  # 'Camera tilt angle in deg from horizon'
    cam_height = 15  # 'Camera hight above object [m]'
    cam_fov = 41.1  # 'Camera field of view deg'
    cam_focal = -1  # 'Camera focal length [m]'
    vert_image_dim = -1  # 'Vertical dimension of 35 mm image format'
    vel_unit_scale = 2.23694  # 'scale to convert velocity units'
    create_vel_video = True  # 'Disables the creation of a velocity video'
    input_file = ""
    model_dir = './checkpoints/yolov4'  # 'path to output'
    weights = './yolo_params/yolov4.weights'  # 'path to weights file'
    myProgress = 0
    IsRunning = False

    def __init__(self):
        self.IsRunning = False
        return None

    def GetProgress(self):
        return self.deepsortYolo.GetProgress(), self.myProgress

    def SetCameraParams(self, cam_tilt, cam_height, cam_fov=41.1, cam_focal=-1):
        self.cam_tilt = cam_tilt  # 'Camera tilt angle in deg from horizon')
        self.cam_height = cam_height  # 'Camera hight above object [m]')
        self.cam_fov = cam_fov  # 'Camera field of view deg')
        self.cam_focal = cam_focal  # 'Camera focal length [m]')

    def SetVelCalibarion(self, scaling_units):
        self.vel_unit_scale = scaling_units

    def GetVideoDuration(self, src_video, start_time, duration):
        video_in = cv2.VideoCapture(src_video)
        if video_in.isOpened():
            fps, total_frames, frame_size = VideoUtils.GetVideoData(video_in)
            start_count, end_count = VideoUtils.GetStartEndCount(fps, total_frames, start_time, duration)
            video_duration = int((end_count-start_count) / fps)
            video_in.release()
        return video_duration

    def SaveWeightsYoloV4(self, output_dir, weights):
        if not os.path.exists(output_dir):
            save_tf(output_dir, weights)

    def ProcessVideo(self, src_video, video_dest_path, model_dir, starttime, duration):
        processed_postfix = 'yolo_'
        logging.info(f"Processing video {src_video}")
        video_duration = self.GetVideoDuration(src_video, starttime, duration)
        print(video_duration)
        basename = pathlib.Path(src_video).stem + '_'
        filename = os.path.join(video_dest_path, basename + VideoUtils.AddTimestampToName(processed_postfix, starttime, video_duration))
        print(filename)
        tracker_file_csv = filename + '.csv'
        output_video_file = filename + pathlib.Path(src_video).suffix
        print(tracker_file_csv)
        if (not os.path.exists(tracker_file_csv)) or (self.process_video):
            model_filename = 'yolo_params/mars-small128.pb'
            if os.path.exists(output_video_file):
                os.remove(output_video_file)

            logging.info(f"Start processing video...")
            output_video_file, trk_bbox = self.deepsortYolo.ProcessVideo(model_dir, model_filename, src_video,
                                                                         video_dest_path, output_video_file, starttime, video_duration, save_images=False)

            print(trk_bbox.head(10))
            trk_bbox.to_csv(tracker_file_csv, index=False)
            logging.info(f"Done processing video")

        self.deepsortYolo.SetProgress(100)
        return output_video_file, tracker_file_csv

    def GetFps(self, src_video):
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

    def AddVelocitiesToVideo(self, src_video, video_dest_path, bb_file, starttime, duration):
        if (not src_video):
            raise Exception(f"File not found: {src_video}")

        logging.info(f"Start adding velocity")
        processed_postfix = 'vel_'
        video_duration = self.GetVideoDuration(src_video, starttime, duration)

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
            scale = self.velUtils.CalculateScale(df)
            print(f"scale from average:  {scale}")

            scale = self.velUtils.CalculateScaleCameraProperites(
                camera_tilt_angle_deg=self.cam_tilt, cam_height=self.cam_height, image_height=frame_size[1],
                cam_fov_deg=self.cam_fov, cam_focal_length=self.cam_focal, vert_image_dim=self.vert_image_dim)
            print(f"scale from camera properties:  {scale}")

            for id in df['object_id'].unique():
                df = self.velUtils.AddVelocity(df, id, fps, scale, frame_size)

            df.to_csv(bb_file, index=False)
            print(f"updated file: {bb_file}")
            if(self.select_id >= 0):
                df_sub = pd.DataFrame(df[(df['object_id'] == self.select_id)])
                object_csv_filename = os.path.splitext(bb_file)[0] + "ID-" + str(self.select_id) + '.csv'
                df_sub.to_csv(object_csv_filename, index=False)
                print(f"Created file: {object_csv_filename}")

            start_count, end_count = VideoUtils.GetStartEndCount(fps, total_frames, starttime, video_duration)
            count = start_count
            fourcc = cv2.VideoWriter_fourcc(*"XVID")

            video_in.set(cv2.CAP_PROP_POS_FRAMES, count)
            i = 0

            # if(create_vel_video):
            if(True):
                video_out = cv2.VideoWriter(output_video_file, fourcc, int(fps), frame_size)
                while (True):
                    success, img = video_in.read()

                    if(success):
                        percent_complete = ((count-start_count)/(end_count-start_count))*100

                        if((count % 25) == 0):
                            print(f"Created frame id {count:2d}, {count/fps:0.2f} sec in video; completed:  {percent_complete:0.1f} %")

                        self.myProgress = percent_complete
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

                            velocity = round(df_img.iloc[i]["adj_vel"] * self.vel_unit_scale, 1)
                            filt_velocity = round(df_img.iloc[i]["filt_vel"] * self.vel_unit_scale, 1)
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
                            if (self.select_id == object_id):
                                ImageUtils.DrawBbox(
                                    img, df_img.iloc[i]["bb_left"],
                                    df_img.iloc[i]["bb_top"],
                                    df_img.iloc[i]["bb_right"],
                                    df_img.iloc[i]["bb_bottom"],
                                    text_color)

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

    # def main(_argv):

    def Run(self, src_video, dest_path, video_duration=-1, video_start=0):
        self.IsRunning = True
        time_now = time.time()
        self.myProgress = 0
        print("App started")
        print(f"selected_id: {self.select_id}")
        success, fps, total_frames, frame_size = self.GetFps(src_video)
        print(f"fps: {fps}, total_frames: {total_frames}, frame_size: {frame_size}, video_start: {video_start}, video_duration: {video_duration}")
        if(not success):
            print(f"Can't open video {src_video}")

        self.SaveWeightsYoloV4(self.model_dir, self.weights)
        video_out, bb_file = self.ProcessVideo(src_video, dest_path, self.model_dir, video_start, video_duration)
        print(f"Storing bounding boxes in: {bb_file}")
        print(f"Created video: {video_out}")

        video_out, df = self.AddVelocitiesToVideo(src_video, dest_path, bb_file, video_start, video_duration)
        print(f"Created video: {video_out}")

        df_sub = pd.DataFrame(df[(df['object_id'] == self.select_id)])
        print(df_sub.tail(50))

        # for id in df['object_id'].unique():
        #     mask = (df['object_id'] == id)
        #     sub_df = df[mask]
        #     # plt.plot(sub_df["Frame"], sub_df["vel"])

        #     if (sub_df[(sub_df['category'] == "person")]):
        #         plt.plot(sub_df["Frame"], sub_df["filt_vel"])

        # plt.show()

        duration = time.time() - time_now
        print(f"Completed {duration} seconds")
        # cnt = 0
        # sleep_sec = 2
        # while True:
        #     time.sleep(sleep_sec)
        #     print(cnt)
        #     cnt += sleep_sec

        self.IsRunning = False
        return video_out
