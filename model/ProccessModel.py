from abc import ABC, abstractclassmethod
from model.AlertModel import WarnAlert
from PyQt5.QtCore import pyqtSignal
import multiprocessing as mp
import queue
import yaml
from model import AlertModel

with open('config.yml', 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)  


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
    def __init__(self, command: mp.Value, camera: object, alert:AlertModel.WarnAlert, ImageSignal: pyqtSignal, AlertSignal:pyqtSignal) -> None:
        super().__init__()
        self.command = command
        self.ImageSignal = ImageSignal
        self.AlertSignal = AlertSignal
        self.WarnAlert = alert

        self.alert = mp.Value('i', 0)
        self.queue = mp.Queue(4)
        self.proccess = mp.Process(target=camera, args=(self.queue, self.command, self.alert))   

    def runCamera(self):
        self.proccess.start()

    def getFrame(self):
        try:
            frame = self.queue.get_nowait()
            self.ImageSignal.emit(frame)
            return frame
        except queue.Empty or queue.Full:
            pass        

    def getAlert(self):
        alert_level = self.alert.value

        if alert_level == config["NO_ALERT_SIGNAL"]:
            return False
        elif alert_level == config["YELLOW_ALERT_SIGNAL"]:
            self.WarnAlert.yellowAlert()
        elif alert_level == config["RED_ALERT_SIGNAL"]:
            self.WarnAlert.redAlert()        

        self.AlertSignal.emit(self.WarnAlert)
        self.alert.value = config["NO_ALERT_SIGNAL"]

    def endCamera(self):
        self.queue.close()
        self.proccess.kill()
        

