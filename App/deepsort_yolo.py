import cv2
import numpy as np
from time import time
from app_utils.video_utils import VideoUtils
from app_utils.image_utils import ImageUtils
import pandas as pd

# deep_sort imports
from deepsort.core import nn_matching
from deepsort.core.detection import Detection
from deepsort.core.tracker import Tracker

# tensorflow_yolov4_tflite imports
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import tf_yolov4.core.utils as utils
from tf_yolov4.core.yolov4 import filter_boxes
from tf_yolov4.core.config import cfg
from deepsort.tools import preprocessing
from deepsort.tools import generate_detections

# Inspired by: https://github.com/theAIGuysCode/yolov4-deepsort


class DeepsortYolo:

    deepsortEncoder = None
    deepsortTracker = None

    def __init__(self):
        return None

    def ProcessVideo(self, yolo_weight_file, model_filename, orig_video, output_dir, output_video, starttime, duration,  save_images=False):
        if (not orig_video):
            raise Exception(f"File not found: {orig_video}")

        saved_model_loaded = tf.saved_model.load(
            yolo_weight_file, tags=[tag_constants.SERVING])
        yolo_model = saved_model_loaded.signatures['serving_default']

        yolo_all = True
        all_classes, find_classes = self.DoYoloV4Init(yolo_all)
        self.DoDeepsortInit(model_filename)
        # print(orig_video)

        bbox_data = pd.DataFrame()
        video_in = cv2.VideoCapture(orig_video)
        if video_in.isOpened():
            print("opened video")
            fps, total_frames, frame_size = VideoUtils.GetVideoData(video_in)
            start_count, end_count = VideoUtils.GetStartEndCount(fps, total_frames, starttime, duration)
            frame_count = start_count

            # setting CV_CAP_PROP_POS_FRAMES at count
            video_in.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            video_out = cv2.VideoWriter(
                output_video, fourcc, int(fps), frame_size)

            i = 0

            while (True):
                success, frame = video_in.read()
                if(success):

                    bboxes, scores, names = self.DoYoloV4(frame, yolo_model, all_classes, find_classes)
                    result, df_frame_data = self.DoDeepsort(frame, bboxes, scores, names)
                    cv2.putText(frame, "Object tracked: {}".format(len(names)), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
                    df_frame_data['Frame'] = frame_count

                    # print(df_frame_data)
                    bbox_data = pd.concat([bbox_data, df_frame_data])

                    i += 1
                    video_out.write(result)
                    if(save_images):
                        cv2.imwrite(output_dir + str(frame_count) + ".jpg", result)

                    frame_count += 1

                    if((frame_count % 25) == 0):
                        percent_complete = (
                            (frame_count-start_count)/(end_count-start_count))*100
                        print(
                            f"Created frame id {frame_count}, {frame_count/fps:0.2f} sec in video; Objects Cnt: {len(names)} completed:  {percent_complete:0.1f} %")

                    if(frame_count > end_count):
                        break

                else:
                    break

            video_in.release()  # done with original video
            video_out.release()

            # print("Done: Created video: " + output_video)

        cv2.destroyAllWindows()

        return output_video, bbox_data

    def DoYoloV4Init(self, yolo_all):
        all_classes = utils.read_class_names(cfg.YOLO.CLASSES)
        if(yolo_all):
            find_classes = list(all_classes.values())
        else:
            find_classes = ['car']

        # model = None
        # if(tf.saved_model.contains_saved_model(yolo_weight_file)):
        #     saved_model_loaded = tf.saved_model.load(yolo_weight_file, tags=[tag_constants.SERVING])
        #     model =  saved_model_loaded.signatures['serving_default']
        #     #.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        # return model
        return all_classes, find_classes

    def DoYoloV4(self, frame, yolo_model, all_classes, find_classes):
        # height, width, layers = frame.shape
        # image_size = (width, height)

        # frame, bbox = self.DoYolo(frame, 0.5, 0.3)
        input_size = 416
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        batch_data = tf.constant(image_data)
        pred_bbox = yolo_model(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.50
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        boxes = nmsed_boxes.numpy()[0]
        boxes = boxes[0:int(num_objects)]
        scores = nmsed_scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = nmsed_classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        boxes = utils.format_boxes(boxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [boxes, scores, classes, num_objects]

        boxes, scores, names = self.SelectObjects(
            boxes, scores, classes, num_objects, all_classes, find_classes)
        return boxes, scores, names

    def DoDeepsortInit(self, model_filename):
        # Definition of the parameters
        max_cosine_distance = 0.4
        nn_budget = None

        # initialize deep sort
        # model_filename = 'model_data/mars-small128.pb'
        self.deepsortEncoder = generate_detections.create_box_encoder(
            model_filename, batch_size=1)
        # calculate cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        # initialize tracker
        self.deepsortTracker = Tracker(metric)

    def DoDeepsort(self, frame, bboxes, scores, names):
        # encode yolo detections and feed to tracker
        features = self.deepsortEncoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox,
                      score, class_name, feature in zip(bboxes, scores, names, features)]

        # run non-maxima supression
        box = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        nms_max_overlap = 1.0
        indices = preprocessing.non_max_suppression(
            box, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        self.deepsortTracker.predict()
        self.deepsortTracker.update(detections)

        # update tracks
        data = {}
        output = pd.DataFrame([])
        for track in self.deepsortTracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            color = ImageUtils.ColorGenerator(track.track_id)
            # Create msg
            msg = str(class_name).upper() + "-" + str(track.track_id)
            ImageUtils.DrawBbox(frame, bbox[0], bbox[1], bbox[2], bbox[3], color, msg)
            data['bb_left'] = int(bbox[0])
            data['bb_top'] = int(bbox[1])
            data['bb_right'] = int(bbox[2])
            data['bb_bottom'] = int(bbox[3])
            data['category'] = class_name
            data['object_id'] = int(track.track_id)
            output = output.append(data, ignore_index=True)

        result = np.asarray(frame)

        # print(df_frame_data)
        return result, output

    def SelectObjects(self, bboxes, scores, classes, availabe_objects, class_names, allowed_classes):
        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(availabe_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)

        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)
        return bboxes, scores, names
