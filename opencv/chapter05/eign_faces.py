# encoding:utf8
import os
import cv2
import numpy as np


def read_faces_and_label(path):
    X, y = [], []
    c = 0
    for dirname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filePath = os.path.join(dirname, filename)
            X.append(np.asarray(cv2.imread(filePath, cv2.IMREAD_GRAYSCALE), dtype=np.uint8))
            y.append(c)
    return [X, y]


if __name__ == "__main__":
    names = ['hewu']

    [X, y] = read_faces_and_label("E:/Project/Tensorflow/tutorial/data/faces")
    model = cv2.face.EigenFaceRecognizer_create()
    model.train(np.asarray(X), np.asarray(y))

    face_cascade = cv2.CascadeClassifier('D:/opencv32/build/etc/haarcascades/haarcascade_frontalface_default.xml')
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi = gray[x:x+w, y:y+h]
            try:
                roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)
                param = model.predict(roi)
                print("Label %s, Confidenct %.2f" % (param[0], param[1]))
                cv2.putText(frame, names[0], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            except:
                continue
        cv2.imshow("camera", frame)
        if cv2.waitKey(10) != -1:
            break
    cv2.destroyAllWindows()