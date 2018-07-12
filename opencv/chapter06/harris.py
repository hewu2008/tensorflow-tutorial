# encoding:utf8

import cv2
import numpy as np

if __name__ == "__main__":
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 23, 0.04)
        frame[dst > 0.01 * dst.max()] = [0, 0, 255]
        cv2.imshow("harris", frame)
        if cv2.waitKey(10) != -1:
            break
    camera.release()
    cv2.destroyAllWindows()