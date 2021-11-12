import cv2
import os
import shutil
import pandas as pd
import numpy as np
import json


class VideoUtils:
    '''Utilities supporting the video and image manipulation'''

    annotationCategoryDict = {}

    def __init__(self, _categories=[]):
        self.annotationCategoryDict = _categories
        print(self.annotationCategoryDict)
        return None

    def CombineImages(self, img_array, full_filename, fps, frame_size):
        print("Combining images to: " + full_filename)
        out = cv2.VideoWriter(
            full_filename, cv2.VideoWriter_fourcc(*'DIVX'), fps, frame_size)

        _cnt = len(img_array)
        for i in range(_cnt):
            out.write(img_array[i])
            if((i % 200) == 0):
                print(str(round((i/_cnt) * 100)) + ' %')

        out.release()
        print("Created video: " + full_filename)

    def CreateImagesArray(self, frame_file_names, cnt):

        img_array = []
        for i in range(cnt):
            img = cv2.imread(frame_file_names[i])
            height, width, layers = img.shape
            image_size = (width, height)
            img_array.append(img)
            if((i % 200) == 0):
                print(str(round((i/cnt) * 100)) + ' %')

        return img_array, image_size

    def CreateVideo(self, frame_file_names, fps, full_filename, framelimit=3000):
        total_cnt = len(frame_file_names)
        filename, file_extension = os.path.splitext(full_filename)
        print(f"Found {total_cnt} images for this video recording.")
        image_count = total_cnt
        remaining_cnt = total_cnt
        video_idx = 0
        end_index = 0
        split_video = False
        while(remaining_cnt > 0):
            # limit size of videos
            if(remaining_cnt > framelimit):
                image_count = framelimit
                split_video = True
            else:
                image_count = remaining_cnt

            video_idx += 1
            start_index = end_index
            end_index = end_index+image_count
            print(f"Collecting {image_count} / {remaining_cnt} images...")
            img_array, img_size = self.CreateImagesArray(
                frame_file_names[start_index:end_index], image_count)
            remaining_cnt -= max(image_count, 0)  # keep it positive

            if(split_video):
                full_video_name = filename + '_' + \
                    str(video_idx) + file_extension
            else:
                full_video_name = filename + file_extension

            if os.path.exists(full_video_name):
                os.remove(full_video_name)

            self.CombineImages(img_array, full_video_name, fps, img_size)

    def AddAllFrameAnnotations(self, img, bbox_data, bbox_thickness):
        for j in range(len(bbox_data)):
            img = self.AddSingleAnnotation(
                img, bbox_data[j], bbox_thickness)
        return img

    def AddSingleAnnotation(self, img, bb_top, bb_left, bb_bottom, bb_right, category, object_id, vel, thickness):
        top = int(bb_top)
        left = int(bb_left)
        bottom = int(bb_bottom)
        right = int(bb_right)
        center_x = int((left + right)/2)
        center_y = int((top + bottom)/2)

        color = (0, 255, 255)
        text = 'Id: ' + str(object_id)
        if(bool(self.annotationCategoryDict)):
            text = text + '\n' + category
            category_code = int(self.annotationCategoryDict[category])
            if (category_code == 0):
                color = (0, 255, 255)
            elif (category_code == 1):
                color = (255, 0, 0)
            elif (category_code == 2):
                color = (255, 255, 0)
            elif (category_code == 3):
                color = (0, 255, 0)
            elif (category_code == 4):
                color = (0, 0, 255)
            elif (category_code == 5):
                color = (255, 255, 0)
            elif (category_code == 6):
                color = (255, 0, 255)
            elif (category_code == 7):
                color = (255, 0, 255)
            else:
                color = (255, 255, 255)

        cv2.rectangle(img, (left, top), (right, bottom), color, thickness)
        cv2.circle(img, (center_x, center_y), 3, (0, 0, 255), -1)
        # print(categories)

        if(vel != None):
            text = text + '\n' + str(vel)

        fontscale = 0.5
        label_width, label_height = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_COMPLEX, fontscale, 1)[0]

        for i, line in enumerate(text.split('\n')):
            y = top + i*(label_height + 5)
            cv2.putText(img, line, (right+10, y),
                        cv2.FONT_HERSHEY_COMPLEX, fontscale, color, 1)

        return img

    def AddAllFrameAnnotations(self, img, df_ann, box_thickness):
        for j in range(len(df_ann)):
            bb_top = df_ann.iloc[j]['bb_top']
            bb_left = df_ann.iloc[j]['bb_left']
            bb_bottom = df_ann.iloc[j]['bb_bottom']
            bb_right = df_ann.iloc[j]['bb_right']
            category = df_ann.iloc[j]['category']
            object_id = int(df_ann.iloc[j]['object_id'])
            vel = df_ann.iloc[j]['vel']
            img = self.AddSingleAnnotation(
                img, bb_top, bb_left, bb_bottom, bb_right, category, object_id, vel, box_thickness)
        return img

    def AnnotateVideo(self, output_dir, orig_v_full_path, new_v_filename, df_ann, start_time=0, duration=[]):
        # Open original video
        vidcap = cv2.VideoCapture(orig_v_full_path)
        if vidcap.isOpened():

            fps = vidcap.get(cv2.CAP_PROP_FPS)  # CV_CAP_PROP_FPS = Frame rate.
            # CV_CAP_PROP_FRAME_COUNT = Number of frames in the video file.
            total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
            print("Total frames in video: %2d @ %5.2f frames/sec" %
                  (total_frames, fps))
            width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_size = (width, height)

            start_count = int(start_time * fps)

            if(total_frames < start_count):
                start_count = 0
                start_time = 0

            if(bool(duration)):
                end_count = int((duration + start_time) * fps)
            else:
                end_count = total_frames - start_count
                print('running full video')

            count = start_count
            print(total_frames, start_count, end_count)
            # setting CV_CAP_PROP_POS_FRAMES at count
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, count)
            timestamp = str(int(start_time)) + '-' + \
                str(int(end_count / fps)) + '_'
            full_filename = output_dir + timestamp + new_v_filename
            if os.path.exists(full_filename):
                os.remove(full_filename)

            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            vidout = cv2.VideoWriter(
                full_filename, fourcc, int(fps), frame_size)

            while (True):
                success, img = vidcap.read()
                if(success):
                    height, width, layers = img.shape
                    image_size = (width, height)

                    if((count % 25) == 0):
                        percent_complete = (
                            (count-start_count)/(end_count-start_count))*100
                        print(
                            f"Created frame id {count:2d}, {count/fps:0.2f} sec in video; completed:  {percent_complete:0.1f} %")

                    # All bounding box info for this frame
                    # Add all annotations to the frame
                    df_img = df_ann[df_ann['frame_id'] == count]
                    img = self.AddAllFrameAnnotations(img, df_img, 2)
                    vidout.write(img)

                    count += 1

                    if(count > end_count):
                        break

                else:
                    break

            vidcap.release()  # done with original video
            vidout.release()

            print("Done: Created video: " + full_filename)

            cv2.destroyAllWindows()


class DirectoryUtils:
    '''Utilities supporting files and folders'''

    def __init__(self):
        return None

    def ClearFileType(self, dir, ext):
        files_in_directory = os.listdir(dir)
        filtered_files = [
            file for file in files_in_directory if file.endswith(ext)]
        for file in filtered_files:
            path_to_file = os.path.join(dir, file)
            os.remove(path_to_file)
