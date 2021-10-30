import cv2
import os
import shutil
import pandas as pd
import numpy as np


class VideoUtils:
    '''Utilities supporting the video and image manipulation'''

    annotationCategoryDict = {}

    def __init__(self, _categories=[]):
        self.annotationCategoryDict = _categories
        return None

    @staticmethod
    def SplitVideoToFrames(output_dir, video_filename):
        vidcap = cv2.VideoCapture(video_filename)
        success, image = vidcap.read()
        count = 0
        images = []
        img_array = np.array(images)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        while (success):
            save_img = output_dir + "frame%d.jpg" % count
            cv2.imwrite(save_img, image)     # save frame as JPEG file
            success, image = vidcap.read()
            if(success):
                img_array = np.append(img_array, save_img)
            count += 1
            print('Created a new frame: ', save_img)

        img_array = np.append(img_array, save_img)
        return img_array, count

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

    def AddAllFrameAnnotations(self, _frame, _bbox_array, _thickness):
        _img = cv2.imread(_frame)
        _item_cnt = len(_bbox_array)  # find how many items in image
        print(f'Adding {_item_cnt} annotations to {_frame}')
        for j in range(_item_cnt):
            updated_img = self.AddSingleAnnotation(
                _img, _bbox_array[j], _thickness)
            cv2.imwrite(_frame, updated_img)

    def AddSingleAnnotation(self, img, bbox, thickness):
        top = bbox['top']
        left = bbox['left']
        height = bbox['height']
        width = bbox['width']
        categories = bbox['class']
        bottom = top + height
        right = left + width
        center_x = int(left + width/2)
        center_y = int(top + height/2)

        color = (0, 255, 255)
        text = ''
        if(bool(self.annotationCategoryDict)):
            text = self.annotationCategoryDict[categories]
            if (categories == 0):
                color = (255, 0, 255)
            elif (categories == 1):
                color = (255, 255, 0)
            elif (categories == 2):
                color = (255, 0, 0)
            elif (categories == 3):
                color = (0, 255, 0)
            elif (categories == 4):
                color = (0, 0, 255)
            elif (categories == 5):
                color = (255, 255, 0)
            elif (categories == 6):
                color = (255, 0, 255)
            elif (categories == 7):
                color = (0, 255, 255)
            else:
                color = (255, 255, 255)

        cv2.rectangle(img, (left, top), (right, bottom), color, thickness)
        cv2.circle(img, (center_x, center_y), 10, (0, 0, 255), -1)
        # print(categories)
        cv2.putText(img, text, (right+10, top),
                    cv2.FONT_HERSHEY_COMPLEX, 0.75, color, 1)
        return img

    def AnnotateVideo(self, output_dir, orig_v_filename, new_v_filename, annotations, fps, v_max_frames=3000):
        _frame_names_array, cnt = self.SplitVideoToFrames(
            output_dir, orig_v_filename)

        for img_idx in range(len(_frame_names_array)):
            frame_name = _frame_names_array[img_idx]
            bbox_data_for_img = annotations[img_idx]
            self.AddAllFrameAnnotations(frame_name, bbox_data_for_img, 2)

        full_filename = output_dir + new_v_filename
        if os.path.exists(full_filename):
            os.remove(full_filename)

        self.CreateVideo(_frame_names_array, fps, full_filename, v_max_frames)


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
