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
                 Mxid, repeat_times, new_value, old_value, left_right, status) -> None:
        super().__init__()
        self.command = command
        self.ImageSignal = ImageSignal
        self.repeat_times = repeat_times
        self.Mxid = Mxid
        self.queue = mp.Queue(4)
        self.new_value = new_value
        self.old_value = old_value
        self.left_right = left_right
        self.status = status
        self.proccess = mp.Process(target=camera, args=(
            self.queue, self.command, self.Mxid, self.repeat_times, self.new_value, self.old_value, self.left_right, self.status))

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
