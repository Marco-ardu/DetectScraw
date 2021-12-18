from abc import ABC, abstractclassmethod
from model.AlertModel import WarnAlert
from factories import AlertFactory
from PyQt5.QtCore import pyqtSignal
import multiprocessing as mp
import queue
import yaml

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
    def __init__(
            self, command: mp.Value, camera: object, camera_id: str, ImageSignal: pyqtSignal, AlertSignal: pyqtSignal) -> None:
        super().__init__()
        self.command = command
        self.ImageSignal = ImageSignal
        self.AlertSignal = AlertSignal


        self.alert = mp.Value('i', 99)
        self.queue = mp.Queue(4)
        self.proccess = mp.Process(target=camera, args=(
            self.queue, self.command, self.alert, camera_id))

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
        if self.alert.value == 99: return

        WarnAlert = AlertFactory.AlertList[self.alert.value]
        WarnAlert.redAlert()

        self.AlertSignal.emit(WarnAlert)
        self.alert.value = 99

    def endCamera(self):
        self.queue.close()
        self.proccess.kill()
