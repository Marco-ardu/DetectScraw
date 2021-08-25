from abc import ABC, abstractclassmethod
from model.AlertModel import WarnAlert
from PyQt5.QtCore import pyqtSignal
import multiprocessing as mp
import queue
from cameraFunc import YoloCamera, FatigueCam, PedestrianCamera
from model import AlertModel
import CAMERA_CONFIG

class ICameraProcess(ABC):
    @abstractclassmethod
    def runCamera(self):
        return NotImplemented

    @abstractclassmethod
    def getFrame(self):
        return NotImplemented

    @abstractclassmethod
    def endCamera(self):
        return NotImplemented

class BasicCameraProccess(ICameraProcess):
    def __init__(self, command: mp.Value, ImageSignal: pyqtSignal, AlertSignal:pyqtSignal, **kargs) -> None:
        super().__init__()
        self.command = command
        self.ImageSignal = ImageSignal
        self.AlertSignal = AlertSignal
        self.WarnAlert = WarnAlert()

        self.alert = mp.Value('i', 0)
        self.queue = mp.Queue(4)
        self.proccess = None

    def getFrame(self):
        try:
            frame = self.queue.get_nowait()
            self.ImageSignal.emit(frame)
            return frame
        except queue.Empty or queue.Full:
            pass        

    def getAlert(self):
        alert_level = self.alert.value

        if alert_level == CAMERA_CONFIG.NO_ALERT_SIGNAL:
            return False
        elif alert_level == CAMERA_CONFIG.YELLOW_ALERT_SIGNAL:
            self.WarnAlert.yellowAlert()
        elif alert_level == CAMERA_CONFIG.RED_ALERT_SIGNAL:
            self.WarnAlert.redAlert()        

        self.AlertSignal.emit(self.WarnAlert)
        self.alert.value = CAMERA_CONFIG.NO_ALERT_SIGNAL

    def endCamera(self):
        self.queue.close()
        self.proccess.kill()
        

class FrontCamera(BasicCameraProccess):       
        
    def runCamera(self):
        self.WarnAlert = AlertModel.AlertFactory(AlertModel.AlertText_PedestrianFront)
        self.proccess = mp.Process(target=YoloCamera.runYoloCamera, args=(self.queue, self.command, self.alert))        
        self.proccess.start()


class RearCamera(BasicCameraProccess):       

    def runCamera(self):
        self.WarnAlert = AlertModel.AlertFactory(AlertModel.AlertText_PedestrianRear)
        self.proccess = mp.Process(target=PedestrianCamera.runPedestrianCamera, args=(self.queue, self.command, self.alert))        
        self.proccess.start()    

class DriverCamera(BasicCameraProccess):      

    def runCamera(self):
        self.WarnAlert = AlertModel.AlertFactory(AlertModel.AlertText_DriverFocus)
        self.proccess = mp.Process(target=FatigueCam.runFatigueCam, args=(self.queue, self.command, self.alert))        
        self.proccess.start()

