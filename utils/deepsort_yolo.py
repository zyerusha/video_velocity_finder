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


class DeepsortYolo:
    def __init__(self):
        return None

    def ProcessVideo(self, yolo_weight_file, yolo_names_file, orig_video, output_dir, output_video, start_time_sec=0, duration_sec=None,  save_images=False):
        """Inspired by: https://github.com/theAIGuysCode/yolov4-deepsort"""
        vUtils = VideoUtils()

        if (not orig_video):
            raise Exception(f"File not found: {orig_video}")

        # self.PrepYolo(yolo_weight_file, yolo_cfg_file, yolo_names_file)
        saved_model_loaded = tf.saved_model.load(yolo_weight_file, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

        # Definition of the parameters
        max_cosine_distance = 0.4
        nn_budget = None
        nms_max_overlap = 1.0
        
        # initialize deep sort
        model_filename = 'model_data/mars-small128.pb'
        encoder = generate_detections.create_box_encoder(model_filename, batch_size=1)
        # calculate cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        # initialize tracker
        tracker = Tracker(metric)
        
        print(orig_video)
        # Open original video
            
        bbox_data = {}
        video_in = cv2.VideoCapture(orig_video)
        if video_in.isOpened():
            print("opened video")
            fps, total_frames, frame_size = vUtils.GetVideoData(video_in)
            start_count, end_count = vUtils.GetStartEndCount(
                fps, total_frames, start_time_sec, duration_sec)
            frame_count = start_count

            # setting CV_CAP_PROP_POS_FRAMES at count
            video_in.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            video_out = cv2.VideoWriter(output_video, fourcc, int(fps), frame_size)

            i = 0
            while (True):
                success, frame = video_in.read()
                if(success):
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width, layers = frame.shape
                    image_size = (width, height)

                    # frame, bbox = self.DoYolo(frame, 0.5, 0.3)
                    input_size = 416
                    image_data = cv2.resize(frame, (input_size, input_size))
                    image_data = image_data / 255.
                    image_data = image_data[np.newaxis, ...].astype(np.float32)
                    batch_data = tf.constant(image_data)
                    pred_bbox = infer(batch_data)
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
                    bboxes = nmsed_boxes.numpy()[0]
                    bboxes = bboxes[0:int(num_objects)]
                    scores = nmsed_scores.numpy()[0]
                    scores = scores[0:int(num_objects)]
                    classes = nmsed_classes.numpy()[0]
                    classes = classes[0:int(num_objects)]

                    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
                    original_h, original_w, _ = frame.shape
                    bboxes = utils.format_boxes(bboxes, original_h, original_w)

                    # store all predictions in one parameter for simplicity when calling functions
                    pred_bbox = [bboxes, scores, classes, num_objects]

                    # read in all class names from config
                    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

                    # allowed_classes = list(class_names.values())
                    allowed_classes = ['car']
                    

                    bboxes, scores, names = self.select_objects(bboxes, scores, classes, num_objects, class_names, allowed_classes)
                    cv2.putText(frame, "Object tracked: {}".format(len(names)), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)

                    # # loop through objects and use class index to get class name, allow only classes in allowed_classes list
                    # names = []
                    # deleted_indx = []
                    # for i in range(num_objects):
                    #     class_indx = int(classes[i])
                    #     class_name = class_names[class_indx]
                    #     if class_name not in allowed_classes:
                    #         deleted_indx.append(i)
                    #     else:
                    #         names.append(class_name)
                    # names = np.array(names)
                    # name_count = len(names)
                    # cv2.putText(frame, "Object tracked: {}".format(name_count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
 
                    # # delete detections that are not in allowed_classes
                    # bboxes = np.delete(bboxes, deleted_indx, axis=0)
                    # scores = np.delete(scores, deleted_indx, axis=0)

                    # encode yolo detections and feed to tracker
                    features = encoder(frame, bboxes)
                    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

                    #initialize color map
                    cmap = plt.get_cmap('tab20b')
                    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

                    # run non-maxima supression
                    boxs = np.array([d.tlwh for d in detections])
                    scores = np.array([d.confidence for d in detections])
                    classes = np.array([d.class_name for d in detections])
                    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
                    detections = [detections[i] for i in indices]       

                    # Call the tracker
                    tracker.predict()
                    tracker.update(detections)

                    # update tracks
                    frame_bbox_list = []
                    for track in tracker.tracks:
                        if not track.is_confirmed() or track.time_since_update > 1:
                            continue 
                        bbox = track.to_tlbr()
                        class_name = track.get_class()
                        
                        # draw bbox on screen
                        color = colors[int(track.track_id) % len(colors)]
                        color = [i * 255 for i in color]
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                        cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                        frame_bbox_list.append(bbox)

                    result = np.asarray(frame)
                    result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    bbox_data[i] = frame_bbox_list
                    i += 1
                    video_out.write(result)
                    if(save_images):
                        cv2.imwrite(output_dir + str(frame_count) + ".jpg", result)
                    frame_count += 1

                    if((frame_count % 25) == 0):
                        percent_complete = (
                            (frame_count-start_count)/(end_count-start_count))*100
                        print(f"Created frame id {frame_count}, {frame_count/fps:0.2f} sec in video; Objects Cnt: {len(names)} completed:  {percent_complete:0.1f} %")

                    if(frame_count > end_count):
                        break

                else:
                    break

            video_in.release()  # done with original video
            video_out.release()

            print("Done: Created video: " + output_video)

        cv2.destroyAllWindows()
        
        return output_video, bbox_data

    def DeepSORT(self, yolo_weight_file, yolo_names_file, orig_video, output_dir, output_video, start_time_sec=0, duration_sec=None,  save_images=False):
        """Inspired by: https://github.com/theAIGuysCode/yolov4-deepsort"""
        vUtils = VideoUtils()

        if (not orig_video):
            raise Exception(f"File not found: {orig_video}")

        # self.PrepYolo(yolo_weight_file, yolo_cfg_file, yolo_names_file)
        saved_model_loaded = tf.saved_model.load(yolo_weight_file, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

        # Definition of the parameters
        max_cosine_distance = 0.4
        nn_budget = None
        nms_max_overlap = 1.0
        
        # initialize deep sort
        model_filename = 'model_data/mars-small128.pb'
        encoder = generate_detections.create_box_encoder(model_filename, batch_size=1)
        # calculate cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        # initialize tracker
        tracker = Tracker(metric)
        
        print(orig_video)
        # Open original video
            
        bbox_data = {}
        video_in = cv2.VideoCapture(orig_video)
        if video_in.isOpened():
            print("opened video")
            fps, total_frames, frame_size = vUtils.GetVideoData(video_in)
            start_count, end_count = vUtils.GetStartEndCount(
                fps, total_frames, start_time_sec, duration_sec)
            frame_count = start_count

            # setting CV_CAP_PROP_POS_FRAMES at count
            video_in.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            video_out = cv2.VideoWriter(output_video, fourcc, int(fps), frame_size)

            i = 0
            while (True):
                success, frame = video_in.read()
                if(success):
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width, layers = frame.shape
                    image_size = (width, height)

                    # frame, bbox = self.DoYolo(frame, 0.5, 0.3)
                    input_size = 416
                    image_data = cv2.resize(frame, (input_size, input_size))
                    image_data = image_data / 255.
                    image_data = image_data[np.newaxis, ...].astype(np.float32)
                    batch_data = tf.constant(image_data)
                    pred_bbox = infer(batch_data)
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
                    bboxes = nmsed_boxes.numpy()[0]
                    bboxes = bboxes[0:int(num_objects)]
                    scores = nmsed_scores.numpy()[0]
                    scores = scores[0:int(num_objects)]
                    classes = nmsed_classes.numpy()[0]
                    classes = classes[0:int(num_objects)]

                    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
                    original_h, original_w, _ = frame.shape
                    bboxes = utils.format_boxes(bboxes, original_h, original_w)

                    # store all predictions in one parameter for simplicity when calling functions
                    pred_bbox = [bboxes, scores, classes, num_objects]

                    # read in all class names from config
                    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

                    # allowed_classes = list(class_names.values())
                    allowed_classes = ['car']
                    

                    bboxes, scores, names = self.select_objects(bboxes, scores, classes, num_objects, class_names, allowed_classes)
                    cv2.putText(frame, "Object tracked: {}".format(len(names)), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)

                    # # loop through objects and use class index to get class name, allow only classes in allowed_classes list
                    # names = []
                    # deleted_indx = []
                    # for i in range(num_objects):
                    #     class_indx = int(classes[i])
                    #     class_name = class_names[class_indx]
                    #     if class_name not in allowed_classes:
                    #         deleted_indx.append(i)
                    #     else:
                    #         names.append(class_name)
                    # names = np.array(names)
                    # name_count = len(names)
                    # cv2.putText(frame, "Object tracked: {}".format(name_count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
 
                    # # delete detections that are not in allowed_classes
                    # bboxes = np.delete(bboxes, deleted_indx, axis=0)
                    # scores = np.delete(scores, deleted_indx, axis=0)

                    # encode yolo detections and feed to tracker
                    features = encoder(frame, bboxes)
                    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

                    #initialize color map
                    cmap = plt.get_cmap('tab20b')
                    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

                    # run non-maxima supression
                    boxs = np.array([d.tlwh for d in detections])
                    scores = np.array([d.confidence for d in detections])
                    classes = np.array([d.class_name for d in detections])
                    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
                    detections = [detections[i] for i in indices]       

                    # Call the tracker
                    tracker.predict()
                    tracker.update(detections)

                    # update tracks
                    frame_bbox_list = []
                    for track in tracker.tracks:
                        if not track.is_confirmed() or track.time_since_update > 1:
                            continue 
                        bbox = track.to_tlbr()
                        class_name = track.get_class()
                        
                        # draw bbox on screen
                        color = colors[int(track.track_id) % len(colors)]
                        color = [i * 255 for i in color]
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                        cv2.putText(frame, class_name + "-" + str(track.track_id),(int(bbox[0]), int(bbox[1]-10)),0, 0.75, (255,255,255),2)
                        frame_bbox_list.append(bbox)

                    result = np.asarray(frame)
                    result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    bbox_data[i] = frame_bbox_list
                    i += 1
                    video_out.write(result)
                    if(save_images):
                        cv2.imwrite(output_dir + str(frame_count) + ".jpg", result)
                    frame_count += 1

                    if((frame_count % 25) == 0):
                        percent_complete = (
                            (frame_count-start_count)/(end_count-start_count))*100
                        print(f"Created frame id {frame_count}, {frame_count/fps:0.2f} sec in video; Objects Cnt: {len(names)} completed:  {percent_complete:0.1f} %")

                    if(frame_count > end_count):
                        break

                else:
                    break

            video_in.release()  # done with original video
            video_out.release()

            print("Done: Created video: " + output_video)

        cv2.destroyAllWindows()
        
        return output_video, bbox_data

    def select_objects(self, bboxes, scores, classes, availabe_objects, class_names, allowed_classes):
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

