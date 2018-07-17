# encoding:utf8

import cv2
import numpy as np

data_path = ""


def path(cls, i):
    return '%s/%s%d.pgm' % (data_path, cls, i)


pos, neg = "pos-", "neg-"

detect = cv2.xfeatures2d.SIFT_create()
extract = cv2.xfeatures2d.SIFT_create()

flan_params = dict(algorithm=1, tree=5)
flann = cv2.FlannBasedMatcher(flan_params, {})

bow_kmeans_trainer = cv2.BOWKMeansTrainer(40)
extract_bow = cv2.BOWImgDescriptorExtractor(extract, flann)


def extract_sift(fn):
    im = cv2.imread(fn, 0)
    return extract.compute(im, detect.detect(im))[1]


for i in range(8):
    bow_kmeans_trainer.add(extract_sift(path(pos, i)))
    bow_kmeans_trainer.add(extract_sift(path(neg, i)))

voc = bow_kmeans_trainer.cluster()
extract_bow.setVocabulary(voc)


def bow_features(fn):
    im = cv2.imread(fn, 0)
    return extract_bow.compute(im, detect.detect(im))


train_data, train_label = [], []
for i in range(20):
    train_data.extend(bow_features(path(pos, i)))
    train_label.append(1)
    train_data.extend(bow_features(path(neg, i)))
    train_label.append(-1)

svm = cv2.ml.SVM_create()
svm.train(np.array(train_data), cv2.ml.ROW_SAMPLE, np.array(train_label))


def predict(fn):
    f = bow_features(fn)
    p = svm.predict(f)
    print(fn, "\t", p[1][0][0])
    return p


if __name__ == "__main__":
    car = ""
    not_car = ""

    car_img = cv2.imread(car)
    not_car_img = cv2.imread(not_car)

    car_predict = predict(car)
    not_car_predict = predict(not_car)

    font = cv2.FONT_HERSHEY_SIMPLEX

    if car_predict[1][0][0] == 1.0:
        cv2.putText(car_img, 'Car Detected', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if not_car_predict[1][0][0] == -1.0:
        cv2.putText(not_car_img, "Car Not detected", (10, 30), font, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("BOW + SVM success", car_img)
    cv2.imshow("BOW + SVM failure", not_car_img)

    cv2.waitKey()
    cv2.destroyAllWindows()
