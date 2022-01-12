import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageUtils:
    def __init__(self):
        return None

    [staticmethod]

    def ColorGenerator(value):
        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
        color = colors[int(value) % len(colors)]
        color = [i * 255 for i in color]
        return color

    [staticmethod]

    def DrawBbox(frame, bb_left, bb_top, bb_right, bb_bottom, color, txt_msg=None, thickness=2, fontScale=0.5, fontFace=cv2.FONT_HERSHEY_SIMPLEX):
        top = int(bb_top)
        left = int(bb_left)
        bottom = int(bb_bottom)
        right = int(bb_right)
        center_x = int((left + right)/2)
        center_y = int((top + bottom)/2)

        # draw Bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

        # draw msg
        if txt_msg:
            size_txt = cv2.getTextSize(txt_msg, fontFace, fontScale, thickness)
            _x11 = left
            _y11 = top
            _x12 = _x11 + max(size_txt[0][0], (right-left))
            _y12 = _y11 - size_txt[0][1] - 5
            cv2.rectangle(frame, (_x11, _y11), (_x12, _y12), color, cv2.FILLED)
            cv2.putText(frame, txt_msg, (_x11, _y11 - 2), fontFace, fontScale, (255, 255, 255), thickness)

        return frame

    def StackImages(self, scale, imgArray):
        """From https://github.com/murtazahassan/Learn-OpenCV-in-3-hours/chapter6.py"""

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
