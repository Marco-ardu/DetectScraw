from PyQt5.QtCore import QThread, pyqtSignal

import numpy
import multiprocessing as mp

from model.AlertModel import WarnAlert
from model.ProccessModel import BasicCameraProccess
from factories import CameraFactory, AlertFactory
import yaml

with open('config.yml', 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)


class Worker(QThread):
    FrontImage = pyqtSignal(numpy.ndarray)
    LeftImage = pyqtSignal(numpy.ndarray)
    RightImage = pyqtSignal(numpy.ndarray)
    Alert = pyqtSignal(WarnAlert)

    command = mp.Value('i', 0)

    def run(self):

        self.command.value = 1

        # FatigueCam = CameraFactory.CameraFactory(CameraFactory.TextFatigueCamera)
        CombinedCam = CameraFactory.CameraFactory(CameraFactory.TextCombinedCamera)
        LeftCamera = BasicCameraProccess(self.command, CombinedCam, config["LEFT_CAMERA_ID"], self.LeftImage, self.Alert)
        RightCamera = BasicCameraProccess(self.command, CombinedCam, config["RIGHT_CAMERA_ID"], self.RightImage, self.Alert)

        # YoloCam = CameraFactory.CameraFactory(CameraFactory.TextYoloCamera)
        # FrontAlert = AlertFactory.AlertFactory(AlertFactory.AlertText_PedestrianFront)
        # FrontCamera = BasicCameraProccess(self.command, YoloCam, FrontAlert, self.FrontImage, self.Alert)

        # PedestrianCam = CameraFactory.CameraFactory(CameraFactory.TextPedestrianCamera)
        # RightAlert = AlertFactory.AlertFactory(AlertFactory.AlertText_PedestrianRear)
        # RearCamera = BasicCameraProccess(self.command, PedestrianCam, RightAlert, self.RightImage, self.Alert)

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
        self.command.value = 0
        self.ThreadActive = False
