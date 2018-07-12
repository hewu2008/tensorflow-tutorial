# encoding: utf8

import os
import numpy
import cv2

if __name__ == "__main__":
    random_byte_array = bytearray(os.urandom(120000))
    flat_numpy_array = numpy.array(random_byte_array)

    gray_image = flat_numpy_array.reshape(300, 400)
    cv2.imwrite("random_gray.png", gray_image)

    bgr_image = flat_numpy_array.reshape(100, 400, 3)
    cv2.imwrite("random_color.png", bgr_image)
