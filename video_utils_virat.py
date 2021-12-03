import cv2
import os
import shutil
import pandas as pd
import numpy as np
import json
from pathlib import Path
from time import time
#from PIL import ImageGrab #from pip install pillow
import pyscreenshot as ImageGrab
# from video_utils_virat import YoloUtils
# from video_utils_virat import VideoUtils

# @article{yolov3,
#   title={YOLOv3: An Incremental Improvement},
#   author={Redmon, Joseph and Farhadi, Ali},
#   journal = {arXiv},
#   year={2018}
# }


class YoloUtils:
    yolo = None
    yolo_classes = []

    def __init__(self):
        return None

    def YoloOnVideo(self, yolo_weight_file, yolo_cfg_file, yolo_names_file,  output_dir, orig_v_full_path, new_v_filename, start_time_sec=0, duration_sec=None,  save_images=False):
        vUtils = VideoUtils()


        if (not orig_v_full_path):
            raise Exception(f"File not found: {orig_v_full_path}")

        self.PrepYolo(yolo_weight_file, yolo_cfg_file, yolo_names_file)
        print(orig_v_full_path)
        # Open original video
            
        bbox_data = {}
        video_in = cv2.VideoCapture(orig_v_full_path)
        if video_in.isOpened():
            print("opened video")
            fps, total_frames, frame_size = vUtils.GetVideoData(video_in)
            start_count, end_count = vUtils.GetStartEndCount(
                fps, total_frames, start_time_sec, duration_sec)
            count = start_count

            # setting CV_CAP_PROP_POS_FRAMES at count
            video_in.set(cv2.CAP_PROP_POS_FRAMES, count)
            timestamp = str(int(start_time_sec)) + '-' + \
                str(int(end_count / fps)) + '_'
            full_filename = output_dir + timestamp + new_v_filename
            if os.path.exists(full_filename):
                os.remove(full_filename)

            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            video_out = cv2.VideoWriter(
                full_filename, fourcc, int(fps), frame_size)

            i = 0
            while (True):
                success, img = video_in.read()
                if(success):
                    height, width, layers = img.shape
                    image_size = (width, height)

                    if((count % 25) == 0):
                        percent_complete = (
                            (count-start_count)/(end_count-start_count))*100
                        print(
                            f"Created frame id {count:2d}, {count/fps:0.2f} sec in video; completed:  {percent_complete:0.1f} %")

                    img, bbox = self.DoYolo(img)
                    bbox_data[i] = bbox
                    i += 1
                    video_out.write(img)
                    if(save_images):
                        cv2.imwrite(output_dir + str(count) + ".jpg", img)
                    count += 1

                    if(count > end_count):
                        break

                else:
                    break

            video_in.release()  # done with original video
            video_out.release()

            print("Done: Created video: " + full_filename)

        cv2.destroyAllWindows()

        return full_filename, bbox_data

    def PrepYolo(self, yolo_weight_file, yolo_cfg_file, yolo_names_file):
        self.yolo = cv2.dnn.readNet(yolo_weight_file, yolo_cfg_file)
        with open(yolo_names_file, "r") as f:
            self.yolo_classes = f.read().splitlines()

    def DoYolo(self, img):
        vUtils = VideoUtils()

        if self.yolo == None:
            Exception(f"Must call PrepYolo first!")

        layer_names = self.yolo.getLayerNames()
        scale_factor = 1/255
        image_reshape = (416, 416)
        blob = cv2.dnn.blobFromImage(
            img, scale_factor, image_reshape, (0, 0, 0), swapRB=True, crop=False)
        self.yolo.setInput(blob)
        output_layer_name = self.yolo.getUnconnectedOutLayersNames()
        layer_output = self.yolo.forward(output_layer_name)
        height, width, channel_cnt = img.shape
        boxes = []
        confidences = []
        class_ids = []

        for output in layer_output:
            for detection in output:
                # print(f'detection: {detection[:4]}')
                x = detection[0]  # center x
                y = detection[1]  # center y
                w = detection[2]  # width
                h = detection[3]  # height
                score_center = detection[4]  # p0
                # get only scores for the bbox (p1, .. Pc);
                score_layers = detection[5:]
                class_id = np.argmax(score_layers)
                confidence = score_layers[class_id]
                if confidence > 0.75:
                    # print(f'x: {x:0.3f}, y: {y:0.3f}, w: {w:0.3f}, h: {h:0.3f}, p0:{score_center:0.3f}')
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        box_cnt = 0
        bbox_data = []
        if(len(boxes) > 0):
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            font = cv2.FONT_HERSHEY_PLAIN
            # color = np.random.uniform(0, 255, size=(len(boxes), 3))
            color = (0, 255, 255)
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                left = int((x - w/2)*width)
                top = int((y - h/2)*height)
                right = int((x + w/2)*width)
                bottom = int((y + h/2)*height)

                label = str(self.yolo_classes[class_ids[i]])
                confi = str(round(confidences[i], 2))
                # color = colors[i]
                cv2.rectangle(img, (left, top), (right, bottom), color, 2)
                center_x = int((left + right)/2)
                center_y = int((top + bottom)/2)
                cv2.circle(img, (center_x, center_y), 3, (0, 0, 255), -1)

                text = label + ' ' + confi
                # cv2.putText(img, label + ' ' + confi,
                #             (right + 10, top), font, 2, color, 2)
                img = vUtils.PrintText(img, text, right, top, 10, 1,
                                       cv2.FONT_HERSHEY_COMPLEX, 0.5, color)
                box_cnt += 1
                bbox_data.append([left, top, right, bottom])

        return img, bbox_data

    def YoloOnScreen(self, yolo_weight_file, yolo_cfg_file, yolo_names_file, x1, y1, x2, y2):
        vUtils = VideoUtils()

        self.PrepYolo(yolo_weight_file, yolo_cfg_file, yolo_names_file)

        loop_time = time()
        while True:
            frame = vUtils.WindowCapture(x1, y1, x2, y2)
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5,
                               interpolation=cv2.INTER_AREA)
            frame = self.DoYolo(frame)
            cv2.imshow("Yolo", frame)
            fps = 1/(time() - loop_time)
            print(f'fps {fps:0.2f}')
            loop_time = time()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def YoloTheWebcam(self, yolo_weight_file, yolo_cfg_file, yolo_names_file, full_filename=None):
        vUtils = VideoUtils()
        vidcap = cv2.VideoCapture(0)
        if not vidcap.isOpened():
            raise IOError("Cannot open webcam")

        self.PrepYolo(yolo_weight_file, yolo_cfg_file, yolo_names_file)

        # width = 640
        # height = 480
        # vidcap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        # vidcap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        vidcap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
        fps, total_frames, frame_size = vUtils.GetVideoData(vidcap)

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        if(full_filename != None):
            vidout = cv2.VideoWriter(
                full_filename, fourcc, int(fps), frame_size)

        loop_time = time()
        while True:
            success, frame = vidcap.read()
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5,
                               interpolation=cv2.INTER_AREA)
            frame = self.DoYolo(frame)
            cv2.imshow("Yolo", frame)
            if(full_filename != None):
                vidout.write(frame)

            fps = 1/(time() - loop_time)
            print(f'fps {fps:0.2f}')
            loop_time = time()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if(full_filename != None):
            vidout.release()

        vidcap.release()
        cv2.destroyAllWindows()


class VideoUtils:
    '''Utilities supporting the video and image manipulation'''

    annotationCategoryDict = {}

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

        # print(f'{left},{top},{right},{bottom}')
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
        cv2.circle(img, (center_x, center_y), 3, (255, 0, 0), -1)
        # print(categories)

        if(vel != None):
            text = text + '\n' + str(vel)

        img = self.PrintText(img, text, right, top, 10, 1,
                             cv2.FONT_HERSHEY_COMPLEX, 0.5, color)

        return img

    def PrintText(self, img, text, x, y, offset, thickness, font, fontscale, color):
        label_width, label_height = cv2.getTextSize(
            text, font, fontscale, thickness)[0]

        for i, line in enumerate(text.split('\n')):
            pos = int(y + i*(label_height + offset/2))
            cv2.putText(img, line, (x+offset, pos), font, fontscale, color, 1)

        return img

    def AddAllFrameAnnotations(self, img, df_ann, box_thickness):
        bbox_data = []
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

            bbox_data.append([bb_left, bb_top, bb_right, bb_bottom])

        return img, bbox_data

    def GetVideoData(self, vidcap):
        fps = vidcap.get(cv2.CAP_PROP_FPS)  # CV_CAP_PROP_FPS = Frame rate.
        total_frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
        width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (width, height)
        print(
            f"Total frames in video: {total_frames:0.2f} @ {fps:0.2f} frames/sec")
        return fps, total_frames, frame_size

    def GetStartEndCount(self, fps, total_frames, start_time_sec, duration_sec=None):
        start_count = int(start_time_sec * fps)
        if(total_frames < start_count):
            start_count = 0
            start_time_sec = 0

        if(bool(duration_sec)):
            end_count = int((duration_sec + start_time_sec) * fps)
        else:
            end_count = total_frames - start_count
            print('running full video')
        return start_count, end_count

    def AnnotateVideo(self, output_dir, orig_v_full_path, new_v_filename, df_ann, start_time_sec=0, duration_sec=None, save_images=False):
        if (not orig_v_full_path):
            raise Exception(f"File not found: {orig_v_full_path}")

        bbox_data = {}
        # Open original video
        video_in = cv2.VideoCapture(orig_v_full_path)
        if video_in.isOpened():
            fps, total_frames, frame_size = self.GetVideoData(video_in)
            start_count, end_count = self.GetStartEndCount(
                fps, total_frames, start_time_sec, duration_sec)

            count = start_count
            print(total_frames, start_count, end_count)
            # setting CV_CAP_PROP_POS_FRAMES at count
            video_in.set(cv2.CAP_PROP_POS_FRAMES, count)
            timestamp = str(int(start_time_sec)) + '-' + \
                str(int(end_count / fps)) + '_'
            full_filename = output_dir + timestamp + new_v_filename
            if os.path.exists(full_filename):
                os.remove(full_filename)

            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            video_out = cv2.VideoWriter(
                full_filename, fourcc, int(fps), frame_size)

            i = 0
            while (True):
                success, img = video_in.read()
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
                    img, bbox = self.AddAllFrameAnnotations(img, df_img, 2)
                    bbox_data[i] = bbox
                    i += 1
                    video_out.write(img)
                    if(save_images):
                        cv2.imwrite(output_dir + str(count) + ".jpg", img)
                    count += 1

                    if(count > end_count):
                        break

                else:
                    break

            video_in.release()  # done with original video
            video_out.release()

            print("Done: Created video: " + full_filename)

        cv2.destroyAllWindows()
        return full_filename, bbox_data

    def ChangeVideoFrameRate(self, full_orig_name, full_new_name, desired_fps, start_time_sec=0, duration_sec=None):
        if (not full_orig_name):
            raise Exception(f"File not found: {full_orig_name}")
        if (not full_new_name):
            raise Exception(f"File not found: {full_new_name}")

        video_in = cv2.VideoCapture(full_orig_name)
        fps, total_frames, frame_size = self.GetVideoData(video_in)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video_out = cv2.VideoWriter(
            full_new_name, fourcc, desired_fps, frame_size)
        total_frames = video_in.get(cv2.CAP_PROP_FRAME_COUNT)

        start_count = int(start_time_sec * desired_fps)
        if(bool(duration_sec)):
            end_count = int((duration_sec - start_time_sec) * desired_fps)
        else:
            end_count = total_frames - start_count
            print('running full video')

        count = start_count
        while (True):
            success, img = video_in.read()
            if(success):
                count += 1

                if((count % 25) == 0):
                    percent_complete = (
                        (count-start_count)/(end_count-start_count))*100
                    print(
                        f"Created frame id {count:2d}, {count/desired_fps:0.2f} sec in video; completed:  {percent_complete:0.1f} %")

                video_out.write(img)
                if(count > end_count):
                    break
            else:
                break

        video_in.release()  # done with original video
        video_out.release()
        print(f'created video: {full_new_name} @ {desired_fps} fps')
        cv2.destroyAllWindows()

    def WindowCapture(self, x1, y1, x2, y2):
        screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))  # X1,Y1,X2,Y2
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
        return screenshot


class ImageUtils:
    def __init__(self):
        return None

    def StackImages(self, scale, imgArray):
        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range(0, rows):
                for y in range(0, cols):
                    if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                        imgArray[x][y] = cv2.resize(
                            imgArray[x][y], (0, 0), None, scale, scale)
                    else:
                        imgArray[x][y] = cv2.resize(
                            imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                    if len(imgArray[x][y].shape) == 2:
                        imgArray[x][y] = cv2.cvtColor(
                            imgArray[x][y], cv2.COLOR_GRAY2BGR)
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank]*rows
            hor_con = [imageBlank]*rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
            ver = np.vstack(hor)
        else:
            for x in range(0, rows):
                if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                    imgArray[x] = cv2.resize(
                        imgArray[x], (0, 0), None, scale, scale)
                else:
                    imgArray[x] = cv2.resize(
                        imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
                if len(imgArray[x].shape) == 2:
                    imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            hor = np.hstack(imgArray)
            ver = hor
        return ver

class Bbox:
    def __init__(self, x1, y1, x2, y2):
        self.left = min(x1, x2)
        self.right = max(x1, x2)
        self.top = min(y1, y2)
        self.bottom = max(y1, y2)
        self.width = abs(self.left - self.right)
        self.height = abs(self.bottom - self.top)
        self.center_x = int((self.left + self.right)/2)
        self.center_x = int((self.top + self.bottom)/2)

    @property
    def area(self):
        return (self.width + 1) * (self.height + 1)

    def UnionArea(self, bbox):
        intersection = self.IntersectionArea(bbox)
        return (self.area + bbox.area - intersection)

    def IntersectionArea(self, bbox):
        dx = min(self.right, bbox.right) - max(self.left, bbox.left) + 1
        dy = min(self.bottom, bbox.bottom) - max(self.top, bbox.top) + 1
        area_of_intersection = max(0, dx) * max(0, dy)
        return area_of_intersection

    def IOU(self, bbox):
        '''Returns intersection over union'''
        iou = self.IntersectionArea(bbox) / self.UnionArea(bbox)
        return iou

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
