# encoding:utf8

import cv2
import numpy as np


def resize(img, scaleFactor):
    return cv2.resize(img, (int(img.shape[1] * (1 / scaleFactor)), int(img.shape[0] * (1 / scaleFactor))), interpolation=cv2.INTER_LINEAR)


def pyramid(image, scale=0.5, minSize=(200, 80)):
    yield image

    while True:
        image = resize(image, scale)
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        yield image


def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def non_max_supression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    scores = boxes[:, 4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    idxs = np.argsort(scores)[::-1]
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]  # 最大评分的下标
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))


data_path = "E:/Project/Tensorflow/tutorial/data/TrainImages"
SAMPLES = 100


def path(cls, i):
    return '%s/%s%d.pgm' % (data_path, cls, i)


def get_flann_matcher():
    flan_params = dict(algorithm=1, tree=5)
    return cv2.FlannBasedMatcher(flan_params, {})


def get_bow_extractor(extract, flann):
    return cv2.BOWImgDescriptorExtractor(extract, flann)


def get_extract_detect():
    return cv2.xfeatures2d.SIFT_create(), cv2.xfeatures2d.SIFT_create()


def extract_sift(fn, extractor, detector):
    im = cv2.imread(fn)
    return extractor.compute(im, detector.detect(im))[1]


def bow_features(img, extract_bow, detector):
    return extract_bow.compute(img, detector.detect(img))


def car_detector():
    pos, neg = "pos-", "neg-"
    detect, extract = get_extract_detect()
    flann = get_flann_matcher()
    print("Building BOWKMeansTrainer...")
    bow_kmeans_trainer = cv2.BOWKMeansTrainer(100)
    extract_bow = cv2.BOWImgDescriptorExtractor(extract, flann)

    print("adding features to trainer")
    for i in range(SAMPLES):
        print(i)
        bow_kmeans_trainer.add(extract_sift(path(pos, i), extract, detect))
        bow_kmeans_trainer.add(extract_sift(path(neg, i), extract, detect))

    print("cluster and set vocabulary")
    voc = bow_kmeans_trainer.cluster()
    extract_bow.setVocabulary(voc)

    traindata, trainlabels = [], []
    print("adding to train data")
    for i in range(SAMPLES):
        print(i)
        traindata.extend(bow_features(cv2.imread(path(pos, i), 0), extract_bow, detect))
        trainlabels.append(1)
        traindata.extend(bow_features(cv2.imread(path(neg, i), 0), extract_bow, detect))
        trainlabels.append(-1)

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setGamma(0.5)
    svm.setC(30)
    svm.setKernel(cv2.ml.SVM_RBF)

    svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))
    return svm, extract_bow


def in_range(number, test, thresh = 0.2):
    return abs(number - test) < thresh


if __name__ == "__main__":
    test_image = "E:/Project/Tensorflow/tutorial/data/timg.jpg"

    svm, extractor = car_detector()
    detect = cv2.xfeatures2d.SIFT_create()

    w, h = 200, 200
    img = cv2.imread(test_image)

    rectangle = []
    counter = 1
    scaleFactor = 2
    scale = 1
    font = cv2.FONT_HERSHEY_PLAIN

    for resized in pyramid(img, scaleFactor):
        print('resized: %d, %d' % (resized.shape[0], resized.shape[1]))
        scale = float(img.shape[1]) / float(resized.shape[1])
        for (x, y, roi) in sliding_window(resized, 100, (w, h)):
            print('%d, %d, roi(%d, %d)' % (x, y, roi.shape[0], roi.shape[1]))
            if roi.shape[1] != w or roi.shape[0] != h:
                continue
            try:
                bf = bow_features(roi, extractor, detect)
                _, result = svm.predict(bf)
                a, res = svm.predict(bf, flags = cv2.ml.STAT_MODEL_RAW_OUTPUT)
                print("Class %d, Score: %f" % (result[0][0], res[0][0]))
                score = res[0][0]
                if result[0][0] == 1:
                    if score < -1.0:
                        rx, ry, rx2, ry2 = int(x * scale), int(y * scale), int((x+w)*scale), int((y+h)*scale)
                        rectangle.append([rx, ry, rx2, ry2, abs(score)])
            except:
                pass
            counter += 1

    windows = np.array(rectangle)
    boxes = non_max_supression_fast(windows, 0.25)

    for (x, y, x2, y2, score) in boxes:
        print(x, y, x2, y2, score)
        cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 1)
        cv2.putText(img, "%f" % score, (int(x), int(y)), (int(x2), int(y2)), (0, 255, 0), 0)

    cv2.imshow("img", img)
    cv2.waitKey()
    cv2.destroyAllWindows()



