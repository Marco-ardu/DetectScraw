from PyQt5.QtCore import QThread, pyqtSignal

import numpy
import multiprocessing as mp

from model.AlertModel import WarnAlert
from model.ProccessModel import BasicCameraProccess
from factories import CameraFactory, AlertFactory


class Worker(QThread):
    FrontImage = pyqtSignal(numpy.ndarray)
    RearImage = pyqtSignal(numpy.ndarray)
    DriverImage = pyqtSignal(numpy.ndarray)
    Alert = pyqtSignal(WarnAlert)

    command = mp.Value('i', 0)

    def run(self):

        self.command.value = 1

        FatigueCam = CameraFactory.CameraFactory(CameraFactory.TextFatigueCamera)
        DriverAlert = AlertFactory.AlertFactory(AlertFactory.AlertText_DriverFocus)
        DriverCamera = BasicCameraProccess(self.command, FatigueCam, DriverAlert, self.DriverImage, self.Alert)

        YoloCam = CameraFactory.CameraFactory(CameraFactory.TextYoloCamera)
        FrontAlert = AlertFactory.AlertFactory(AlertFactory.AlertText_PedestrianFront)
        FrontCamera = BasicCameraProccess(self.command, YoloCam, FrontAlert, self.FrontImage, self.Alert)

        PedestrianCam = CameraFactory.CameraFactory(CameraFactory.TextPedestrianCamera)
        RearAlert = AlertFactory.AlertFactory(AlertFactory.AlertText_PedestrianRear)
        RearCamera = BasicCameraProccess(self.command, PedestrianCam, RearAlert, self.RearImage, self.Alert)

        Cameras = [DriverCamera, FrontCamera, RearCamera]
        
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


