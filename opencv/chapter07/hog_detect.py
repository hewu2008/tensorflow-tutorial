# encoding:utf8

import cv2


def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih


def draw_person(image, person):
    x, y, w, h = person
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

def donothing(x):
    pass

if __name__ == "__main__":
    img = cv2.imread("E:/Project/Tensorflow/tutorial/data/2126335.jpg")
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cv2.namedWindow("people detection")
    cv2.createTrackbar("r", "people detection", 0, 255, donothing)
    found, w = hog.detectMultiScale(img)
    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and is_inside(r, q):
                break
            else:
                found_filtered.append(r)

    for person in found_filtered:
        draw_person(img, person)

    cv2.imshow('people detection', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
