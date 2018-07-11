# encoding:utf8

import cv2
from opencv.chapter02.managers import WindowManager, CaptureManager
from opencv.chapter03.filters import *
from opencv.chapter03.stroke_edge import *

class Cameo(object):
    def __init__(self):
        self._windowManager = WindowManager("Cameo", self.onKeypress)
        self._captureManager = CaptureManager(cv2.VideoCapture(0), self._windowManager, True)
        self._curveFilter = EmbossFilter()

    def run(self):
        self._windowManager.createWindow()
        while self._windowManager.isWindowCreated:
            self._captureManager.enterFrame()
            frame = self._captureManager.frame
            strokeEdges(frame, frame)
            self._curveFilter.apply(frame, frame)
            self._captureManager.exitFrame()
            self._windowManager.processEvents()

    def onKeypress(self, keycode):
        if keycode == 32:
            self._captureManager.writeImage('screenshot.png')
        elif keycode == 9:
            if not self._captureManager.isWritingVideo:
                self._captureManager.startWriteVideo("screencast.avi")
            else:
                self._captureManager.stopWriteVideo()
        elif keycode == 27:
            self._windowManager.destoryWindow()


if __name__ == "__main__":
    Cameo().run()