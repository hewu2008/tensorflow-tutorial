#encoding:utf8

import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

low_threshold = 100
high_threshold = 200

def low_threshold_slider(value):
    global low_threshold
    low_threshold = value

def high_threshold_slider(value):
    global high_threshold
    high_threshold = value

if __name__ == "__main__":
    kernal_3x3 = np.array([[-1, -1, -1],
                           [-1, 8, 1],
                           [-1, -1, -1]])
    kernal_5x5 = np.array([[-1, -1, -1, -1, -1],
                           [-1, 1, 2, 1, -1],
                           [-1, 2, 4, 2, -1],
                           [-1, 1, 2, 1, -1],
                           [-1, -1, -1, -1, -1]])


    img = cv2.imread('E:/Project/Tensorflow/tutorial/data/longan/5.bmp', 0)
    cv2.namedWindow("threshold")
    cv2.createTrackbar('low_threshold', 'threshold', low_threshold, 255, low_threshold_slider)
    cv2.createTrackbar('high_threshold', 'threshold', high_threshold, 255, high_threshold_slider)

    while True:
        threshold = cv2.inRange(img, low_threshold, high_threshold)
        cv2.imshow('threshold', threshold)
        if cv2.waitKey(10) != -1:
            break
    cv2.destroyAllWindows()