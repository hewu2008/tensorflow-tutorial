#encoding:utf8

import cv2
import numpy

if __name__ == "__main__":
    img = cv2.imread('5.bmp', 0)
    dst = cv2.Canny(img, 200, 300)
    cv2.imshow('canny', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()