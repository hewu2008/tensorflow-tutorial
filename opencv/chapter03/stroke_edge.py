# encoding:utf8

import cv2
import numpy

def strokeEdges(src, dst, blurKsize=7, edgeKsize=5):
    if blurKsize >= 3:
        blurredSrc = cv2.medianBlur(src, blurKsize)
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize=edgeKsize)
    cv2.imshow('graysrc', graySrc)

    normalizedInverseAlpha = (1.0 / 255) * (255 - graySrc)
    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels, dst)


if __name__ == "__main__":
    src = cv2.imread('color1_small.jpg')
    dst = numpy.zeros(src.shape,  dtype="uint8")
    strokeEdges(src, dst)

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)

    cv2.waitKey()
    cv2.destroyAllWindows()