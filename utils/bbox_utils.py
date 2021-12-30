import cv2
import numpy as np
from utils.video_utils import VideoUtils
import os
from sys import path

class Bbox:
    def __init__(self, x1 : int, y1 : int, x2 : int, y2 : int):
        self.left = min(x1, x2)
        self.right = max(x1, x2)
        self.top = min(y1, y2)
        self.bottom = max(y1, y2)
        self.width = abs(self.left - self.right)
        self.height = abs(self.bottom - self.top)
        self.center_x = int((self.left + self.right)/2)
        self.center_y = int((self.top + self.bottom)/2)

    # def __init__(self, x1, y1, x2, y2, image_height = 1, image_width = 1):
    #     self.left = int(min(x1, x2) * image_width)
    #     self.right = int(max(x1, x2) * image_width)
    #     self.top = int(min(y1, y2) * image_height)
    #     self.bottom = int(max(y1, y2) * image_height)
    #     self.width = abs(self.left - self.right)
    #     self.height = abs(self.bottom - self.top)
    #     self.center_x = int((self.left + self.right)/2)
    #     self.center_y = int((self.top + self.bottom)/2)

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

class Evaluate:

    [staticmethod]
    def GetMaxCorrelation(bb_gt, pred_bboxes, frame):
        max_iou = 0.0
        max_info = [frame, bb_gt, bb_gt, max_iou]
        found = False

        for idx, row in pred_bboxes.iterrows():
            bb = [row["bb_left"], row["bb_top"], row["bb_right"], row["bb_bottom"]]
            gt = Bbox(int(bb_gt[0]),int(bb_gt[1]),int(bb_gt[2]),int(bb_gt[3]))
            pred = Bbox(int(float(bb[0])),int(float(bb[1])),int(float(bb[2])),int(float(bb[3])))
            iou = gt.IOU(pred)
            if (max_iou < iou) & (0.5 < iou):
                found = True
                max_iou = iou
                max_info = [frame, bb_gt, bb, iou]

        return found, max_info

    [staticmethod]
    def AddIouToFrame(img, df_info):
        for index, row in df_info.iterrows():
            bb_gt = row['gt bbox']
            text = "IOU: {iou:.2f}".format(iou = row['iou'])
            bb = Bbox(bb_gt[0],bb_gt[1],bb_gt[2],bb_gt[3])
            color = (0, 255, 255)
            cv2.putText(img, text,(bb.left + 10, bb.bottom - 10),0, 0.5, color,1)
        return img

    [staticmethod]
    def AddIouToVideo(output_dir, orig_v_full_path, df_info):
        if (not orig_v_full_path):
            raise Exception(f"File not found: {orig_v_full_path}")

        # Open original video
        video_in = cv2.VideoCapture(orig_v_full_path)
        if video_in.isOpened():
            fps, total_frames, frame_size = VideoUtils.GetVideoData(video_in)
            count = 0
            # setting CV_CAP_PROP_POS_FRAMES at count
            video_in.set(cv2.CAP_PROP_POS_FRAMES, count)
            output_dir = os.path.dirname(orig_v_full_path)
            full_filename = os.path.join(output_dir, 'iou_' + os.path.basename(orig_v_full_path))
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

                    # All bounding box info for this frame
                    # Add all annotations to the frame
                    df_img = df_info[df_info['frame idx'] == count]
                    img = Evaluate.AddIouToFrame(img, df_img)
                    i += 1
                    video_out.write(img)
                    count += 1

                else:
                    break

            video_in.release()  # done with original video
            video_out.release()

            print("Done: Created video: " + full_filename)