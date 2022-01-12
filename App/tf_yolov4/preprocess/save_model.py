import tensorflow as tf
from tf_yolov4.core.yolov4 import YOLO, decode, filter_boxes
import tf_yolov4.core.utils as utils


def save_tf(output_dir, weights):
    tiny = False  # 'is yolo-tiny or not'
    input_size = 416  # 'define input size of export model'
    score_thres = 0.2  # 'define score threshold'
    # 'define what framework do you want to convert (tf, trt, tflite)'
    framework = 'tf'
    model = 'yolov4'  # 'yolov3 or yolov4'

    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(tiny, model)

    input_layer = tf.keras.layers.Input([input_size,  input_size, 3])
    feature_maps = YOLO(input_layer, NUM_CLASS,  model,  tiny)
    bbox_tensors = []
    prob_tensors = []
    if tiny:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(
                    fm,  input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,  framework)
            else:
                output_tensors = decode(
                    fm,  input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,  framework)
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])
    else:
        for i, fm in enumerate(feature_maps):
            if i == 0:
                output_tensors = decode(
                    fm,  input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,  framework)
            elif i == 1:
                output_tensors = decode(
                    fm,  input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,  framework)
            else:
                output_tensors = decode(
                    fm,  input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE,  framework)
            bbox_tensors.append(output_tensors[0])
            prob_tensors.append(output_tensors[1])
    pred_bbox = tf.concat(bbox_tensors, axis=1)
    pred_prob = tf.concat(prob_tensors, axis=1)
    if framework == 'tflite':
        pred = (pred_bbox, pred_prob)
    else:
        boxes, pred_conf = filter_boxes(
            pred_bbox, pred_prob, score_threshold=score_thres, input_shape=tf.constant([input_size,  input_size]))
        pred = tf.concat([boxes, pred_conf], axis=-1)
    model = tf.keras.Model(input_layer, pred)
    utils.load_weights(model,  weights,  model,  tiny)
    # model.summary()
    model.save(output_dir)

    return output_dir
