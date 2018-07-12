# encoding:utf8

import cv2

if __name__ == "__main__":
    camera = cv2.VideoCapture(0)
    while True:
        ret, img = camera.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        surf = cv2.xfeatures2d.SURF_create(1000)
        keypoints, descriptor = surf.detectAndCompute(gray, None)

        img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                                color=(51, 163, 236))

        cv2.imshow("sift_keypoints", img)
        if cv2.waitKey(10) != -1:
            break
    camera.release()
    cv2.destroyAllWindows()