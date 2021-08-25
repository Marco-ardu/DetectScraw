from PyQt5.QtCore import QThread, pyqtSignal

import multiprocessing as mp
import numpy

from model import AlertModel, ProccessModel


class Worker(QThread):
    FrontImage = pyqtSignal(numpy.ndarray)
    RearImage = pyqtSignal(numpy.ndarray)
    DriverImage = pyqtSignal(numpy.ndarray)
    Alert = pyqtSignal(AlertModel.WarnAlert)

    command = mp.Value('i', 1)

    def run(self):

        self.command.value = 1

        DriverCamera = ProccessModel.DriverCamera(self.command, AlertModel.AlertText_DriverFocus, self.DriverImage, self.Alert)
        FrontCamera = ProccessModel.FrontCamera(self.command, AlertModel.AlertText_PedestrianFront, self.FrontImage, self.Alert)
        RearCamera = ProccessModel.RearCamera(self.command, AlertModel.AlertText_PedestrianRear, self.RearImage, self.Alert)

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


