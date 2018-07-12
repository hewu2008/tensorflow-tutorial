# encoding:utf8

import cv2
import numpy as np
from os.path import join

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