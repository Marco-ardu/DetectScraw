import multiprocessing as mp
import queue
from abc import ABC, abstractmethod

import yaml
from PyQt5.QtCore import pyqtSignal
from cameraFunc import DetectScrawCamera

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
    def __init__(self, ImageSignal: pyqtSignal, AlertSignal: pyqtSignal, Mxid, new_value, status, barcode, command) -> None:
        super().__init__()
        self.camera = DetectScrawCamera.run_Scraw_Camera
        self.ImageSignal = ImageSignal
        self.AlertSignal = AlertSignal
        self.old_value = {'lenPos_old': mp.Value('Q', 156), 'exp_time_old': mp.Value('Q', 20000),
                          'sens_ios_old': mp.Value('Q', 800)}
        self.new_value = new_value
        self.status = status
        self.Mxid = Mxid
        self.queue = mp.Queue(4)
        self.barcode = barcode
        self.result = mp.Queue(4)
        self.alert = mp.Value('i', 99)
        self.command = command
        self.repeat_times = mp.Value('i', 0)

        self.left_right = ''

    def runCamera(self):
        self.command.value = 1
        self.proccess = mp.Process(target=self.camera, args=(
            self.queue, self.command, self.alert, self.Mxid, self.repeat_times, self.new_value, self.old_value,
            self.left_right, self.status, self.barcode, self.result))
        self.proccess.start()

    def getFrame(self):
        try:
            frame = self.queue.get_nowait()
            self.ImageSignal.emit(frame)
            return frame
        except queue.Empty or queue.Full:
            pass

    def getAlert(self):
        pass
        # if self.alert.value == 99:
        #     return
        #
        # WarnAlert = AlertFactory.AlertList[self.alert.value]
        # WarnAlert.redAlert()
        #
        # self.AlertSignal.emit(WarnAlert)
        # self.alert.value = 99

    def endCamera(self):
        self.repeat_times.value = 0
        self.command.value = 0
        self.old_value = {'lenPos_old': mp.Value('Q', 156), 'exp_time_old': mp.Value('Q', 20000),
                          'sens_ios_old': mp.Value('Q', 800)}
        self.new_value = {'lenPos_new': mp.Value('Q', 156), 'exp_time_new': mp.Value('Q', 20000),
                          'sens_ios_new': mp.Value('Q', 800)}
        self.status = {'auto_exp_status': mp.Value('i', 1), 'auto_focus_status': mp.Value('i', 1)}
        self.proccess.terminate()


class LeftCameraProcess(BasicCameraProcess):
    def __init__(self, ImageSignal: pyqtSignal, AlertSignal: pyqtSignal, Mxid, new_value, status, barcode, command) -> None:
        super().__init__(ImageSignal, AlertSignal, Mxid, new_value, status, barcode, command)
        self.left_right = 'left'


class RightCameraProcess(BasicCameraProcess):
    def __init__(self, ImageSignal: pyqtSignal, AlertSignal: pyqtSignal, Mxid, new_value, status, barcode, command) -> None:
        super().__init__(ImageSignal, AlertSignal, Mxid, new_value, status, barcode, command)
        self.left_right = 'right'

