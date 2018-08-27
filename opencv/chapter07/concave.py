# encoding:utf8

import cv2

if __name__ == "__main__":
    fn = "E:/1.bmp"

    image = cv2.imread(fn)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    img = cv2.drawContours(color, contours, -1, (0, 255, 0), 2)

    cv2.imshow('contours', img)
    cv2.imshow("image", image)

    cv2.waitKey()
    cv2.destroyAllWindows()