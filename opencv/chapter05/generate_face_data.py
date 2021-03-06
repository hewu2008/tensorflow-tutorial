
# encoding:utf8

import cv2
import os


def generate():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    camera = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            f = cv2.resize(gray[y: y + h, x: x + w], (200, 200))
            cv2.imwrite("%s.pgm" % count, f)
            count += 1

        cv2.imshow("camera", frame)

        if cv2.waitKey(10) != -1:
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    for i in range(0, 26):
        fn = "E:/Project/Tensorflow/tutorial/data/result/%d.pgm" % i
        if os.path.exists(fn):
            img = cv2.imread(fn, 0)
            cv2.imshow("%d" %i, img)
    cv2.waitKey()
    cv2.destroyAllWindows()