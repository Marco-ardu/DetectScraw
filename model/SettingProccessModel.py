import multiprocessing as mp
import queue
from abc import ABC, abstractmethod

import yaml
from PyQt5.QtCore import pyqtSignal


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


class SettingCameraProcess(ICameraProcess):
    def __init__(self, command: mp.Value, camera, ImageSignal: pyqtSignal,
                 Mxid, repeat_times, lenPos, exp_time, sens_ios, old_value) -> None:
        super().__init__()
        self.command = command
        self.ImageSignal = ImageSignal
        self.repeat_times = repeat_times
        self.Mxid = Mxid
        self.queue = mp.Queue(4)
        self.lenPos = lenPos
        self.exp_time = exp_time
        self.sens_ios = sens_ios
        self.old_value = old_value
        self.proccess = mp.Process(target=camera, args=(
            self.queue, self.command, self.Mxid, self.repeat_times, self.lenPos, self.exp_time, self.sens_ios, self.old_value))

    def runCamera(self):
        self.proccess.start()

    def getFrame(self):
        try:
            frame = self.queue.get_nowait()
            self.ImageSignal.emit(frame)
            return frame
        except queue.Empty or queue.Full:
            pass

    def endCamera(self):
        self.queue.close()
        self.proccess.kill()
