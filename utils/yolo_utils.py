import cv2
import numpy as np
from time import time
from utils.video_utils import VideoUtils
import matplotlib.pyplot as plt

# deep_sort imports
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker

# tensorflow_yolov4_tflite imports
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import core.utils as utils
from core.yolov4 import filter_boxes
from core.config import cfg
from tools import preprocessing
from tools import generate_detections


class YoloUtils:
    yolo = None
    yolo_classes = []
    def __init__(self):
        return None
    def YoloOnVideo(self, yolo_weight_file, yolo_cfg_file, yolo_names_file, orig_video, output_dir, output_video, start_time_sec=0, duration_sec=None,  save_images=False):
        vUtils = VideoUtils()

        if (not orig_video):
            raise Exception(f"File not found: {orig_video}")

        self.PrepYolo(yolo_weight_file, yolo_cfg_file, yolo_names_file)
        print(orig_video)
        # Open original video
            
        bbox_data = {}
        video_in = cv2.VideoCapture(orig_video)
        if video_in.isOpened():
            print("opened video")
            fps, total_frames, frame_size = vUtils.GetVideoData(video_in)
            start_count, end_count = vUtils.GetStartEndCount(
                fps, total_frames, start_time_sec, duration_sec)
            count = start_count

            # setting CV_CAP_PROP_POS_FRAMES at count
            video_in.set(cv2.CAP_PROP_POS_FRAMES, count)

            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            video_out = cv2.VideoWriter(output_video, fourcc, int(fps), frame_size)

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

                    img, bbox = self.DoYolo(img, 0.5, 0.3)
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

            print("Done: Created video: " + output_video)

        cv2.destroyAllWindows()

        return output_video, bbox_data
    def PrepYolo(self, yolo_weight_file, yolo_cfg_file, yolo_names_file):
        self.yolo = cv2.dnn.readNet(yolo_weight_file, yolo_cfg_file)
        with open(yolo_names_file, "r") as f:
            self.yolo_classes = f.read().splitlines()
    def DoYolo(self, img, score_threshold, nms_threshold):
        # score_threshold	a threshold used to filter boxes by score.
        # nms_threshold	a threshold used in non maximum suppression.

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
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)
            font = cv2.FONT_HERSHEY_PLAIN
            # color = np.random.uniform(0, 255, size=(len(boxes), 3))
            color = (255, 0, 255)
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
                img = vUtils.PrintText(img, text, right, bottom, 10, 1,
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
            frame = self.DoYolo(frame, 0.5, 0.3)
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
            frame = self.DoYolo(frame, 0.5, 0.3)
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


class Bbox:
    def __init__(self, x1, y1, x2, y2):
        self.left = min(x1, x2)
        self.right = max(x1, x2)
        self.top = min(y1, y2)
        self.bottom = max(y1, y2)
        self.width = abs(self.left - self.right)
        self.height = abs(self.bottom - self.top)
        self.center_x = int((self.left + self.right)/2)
        self.center_y = int((self.top + self.bottom)/2)

    def __init__(self, x1, y1, x2, y2, image_height, image_width):
        self.left = int(min(x1, x2) * image_width)
        self.right = int(max(x1, x2) * image_width)
        self.top = int(min(y1, y2) * image_height)
        self.bottom = int(max(y1, y2) * image_height)
        self.width = abs(self.left - self.right)
        self.height = abs(self.bottom - self.top)
        self.center_x = int((self.left + self.right)/2)
        self.center_y = int((self.top + self.bottom)/2)

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
