# encoding:utf8

import cv2


def detect(filename):
    face_cascade = cv2.CascadeClassifier('D:/opencv32/build/etc/haarcascades/haarcascade_frontalface_default.xml')
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print('found face num %d' % len(faces))
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.namedWindow("Vikings Detected")
    cv2.imshow("Vikings Detected", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print('found face num %d' % len(faces))
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow("camera", frame)
        if cv2.waitKey(10) != -1:
            break
    cv2.destroyAllWindows()
