import multiprocessing as mp

import keyboard
import numpy
from PyQt5.QtCore import QThread, pyqtSignal

from factories import CameraFactory
from model.AlertModel import WarnAlert
from model.ProccessModel import BasicCameraProcess


class Worker(QThread):
    FrontImage = pyqtSignal(numpy.ndarray)
    RearImage = pyqtSignal(numpy.ndarray)
    Alert = pyqtSignal(WarnAlert)

    command = mp.Value('i', 0)
    repeat_times = mp.Value('i', 0)

    def run(self):

        self.command.value = 1

        DetectScrawRightCamera = CameraFactory.CameraFactory(CameraFactory.TextDetectScrawRightCamera)
        RightCamera = BasicCameraProcess(self.command, DetectScrawRightCamera, self.RearImage, self.Alert, self.repeat_times)

        DetectScrawLeftCamera = CameraFactory.CameraFactory(CameraFactory.TextDetectScrawLeftCamera)
        LeftCamera = BasicCameraProcess(self.command, DetectScrawLeftCamera, self.FrontImage, self.Alert, self.repeat_times)

        Cameras = [LeftCamera, RightCamera]

        for Camera in Cameras:
            Camera.runCamera()

        self.ThreadActive = True

        while self.ThreadActive:
            for Camera in Cameras:
                Camera.getFrame()
                Camera.getAlert()

        for Camera in Cameras:
            Camera.endCamera()

        self.quit()

    def stop(self):
        self.repeat_times.value = 0
        self.command.value = 0
        self.ThreadActive = False
