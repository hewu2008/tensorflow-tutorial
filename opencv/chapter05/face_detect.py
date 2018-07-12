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


def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

if __name__ == "__main__":
    face_cascade = cv2.CascadeClassifier('D:/opencv32/build/etc/haarcascades/haarcascade_frontalface_default.xml')
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # print('found face num %d' % len(faces))
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow("camera", frame)
        if cv2.waitKey(10) != -1:
            break
    cv2.destroyAllWindows()
