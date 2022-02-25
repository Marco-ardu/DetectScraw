import multiprocessing as mp
import queue
from abc import ABC, abstractmethod

import yaml
from PyQt5.QtCore import pyqtSignal

from factories import AlertFactory

with open('config.yml', 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)


class ICameraProcess(ABC):
    @abstractmethod
    def runCamera(self):
        return NotImplemented

    @abstractmethod
    def getFrame(self):
        return NotImplemented

    @abstractmethod
    def endCamera(self):
        return NotImplemented


class BasicCameraProcess(ICameraProcess):
    def __init__(self, command: mp.Value, camera, ImageSignal: pyqtSignal,
                 AlertSignal: pyqtSignal, repeat_times) -> None:
        super().__init__()
        self.command = command
        self.ImageSignal = ImageSignal
        self.AlertSignal = AlertSignal
        self.repeat_times = repeat_times
        self.alert = mp.Value('i', 99)
        self.queue = mp.Queue(4)
        self.proccess = mp.Process(target=camera, args=(
            self.queue, self.command, self.alert, self.repeat_times))

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
        if self.alert.value == 99:
            return

        WarnAlert = AlertFactory.AlertList[self.alert.value]
        WarnAlert.redAlert()

        self.AlertSignal.emit(WarnAlert)
        self.alert.value = 99

    def endCamera(self):
        self.queue.close()
        self.proccess.kill()
