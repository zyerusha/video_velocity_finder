import cv2
import numpy as np

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