# encoding:utf8

import cv2

clicked = False


def onMouse(event, x, y, flag, param):
    global clicked
    if event == cv2.EVENT_FLAG_LBUTTON:
        clicked = True


if __name__ == "__main__":
    cameraCapture = cv2.VideoCapture(0)
    cv2.namedWindow("MyWindow")
    cv2.setMouseCallback('MyWindow', onMouse)

    print("Showing Camera feed, Click Window or Pause any key to Stop.")
    success, frame = cameraCapture.read()
    while success and cv2.waitKey(1) == -1 and not clicked:
        cv2.imshow('MyWindow', frame)
        success, frame = cameraCapture.read()

    cv2.destroyAllWindows()
    cameraCapture.release()

