import cv2
import numpy as np

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
