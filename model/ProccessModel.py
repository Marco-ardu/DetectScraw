import multiprocessing as mp
import queue
import traceback
from abc import ABC, abstractmethod

import yaml
from PyQt5.QtCore import pyqtSignal
from loguru import logger

from cameraFunc import DetectScrawCamera
from factories.AlertFactory import AlertEnum

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
    def __init__(self, ImageSignal: pyqtSignal, AlertSignal: pyqtSignal, Mxid, new_value, status, barcode,
                 command, left_location, left_isQualified, right_location, right_isQualified) -> None:
        super().__init__()
        self.camera = DetectScrawCamera.run_Scraw_Camera
        self.ImageSignal = ImageSignal
        self.AlertSignal = AlertSignal
        self.old_value = {'lenPos_old': mp.Value('Q', 156), 'exp_time_old': mp.Value('Q', 20000),
                          'sens_ios_old': mp.Value('Q', 800)}
        self.new_value = new_value
        self.status = status
        self.Mxid = Mxid
        self.barcode = barcode
        self.command = command
        self.queue = mp.Queue(4)
        # self.save_frame = None
        self.send_result, self.recv_result = mp.Pipe()
        self.alert = mp.Value('i', 99)
        self.repeat_times = mp.Value('i', 0)
        self.right_location = right_location
        self.right_isQualified = right_isQualified
        self.left_location = left_location
        self.left_isQualified = left_isQualified
        self.direction = ''

    def runCamera(self):
        self.command.value = 1
        self.proccess = mp.Process(target=self.camera, args=(
            self.queue, self.command, self.alert, self.Mxid, self.repeat_times, self.new_value, self.old_value,
            self.direction, self.status, self.barcode, self.send_result))
        self.proccess.start()

    def getFrame(self):
        try:
            frame = self.queue.get_nowait()
            self.ImageSignal.emit(frame)
            return frame
        except queue.Empty or queue.Full:
            pass

    def getAlert(self):
        if self.alert.value == int(AlertEnum.NoAlert):
            return
        self.AlertSignal.emit(AlertEnum(self.alert.value))
        self.alert.value = int(AlertEnum.NoAlert)

    def endCamera(self):
        self.repeat_times.value = 0
        self.command.value = 0
        self.old_value = {'lenPos_old': mp.Value('Q', 156), 'exp_time_old': mp.Value('Q', 20000),
                          'sens_ios_old': mp.Value('Q', 800)}
        self.new_value = {'lenPos_new': mp.Value('Q', 156), 'exp_time_new': mp.Value('Q', 20000),
                          'sens_ios_new': mp.Value('Q', 800)}
        self.status = {'auto_exp_status': mp.Value('i', 1), 'auto_focus_status': mp.Value('i', 1)}
        logger.info('stop {} camera'.format(self.direction))
        self.proccess.terminate()

    def setAlert(self):
        logger.info('{} {} {} {}'.format(self.left_location.value, self.right_location.value, self.left_isQualified.value, self.right_isQualified.value))
        if self.left_location.value == 2 and self.right_location.value == 2:
            self.alert.value = 4
        elif self.left_location.value == 2 and self.right_location.value == 1:
            self.alert.value = 5
        elif self.left_location.value == 1 and self.right_location.value == 2:
            self.alert.value = 6
        else:
            if self.left_isQualified.value == 1 and self.right_isQualified.value == 1:
                self.alert.value = 3
            elif self.left_isQualified.value == 2 and self.right_isQualified.value == 1:
                self.alert.value = 1
            elif self.left_isQualified.value == 1 and self.right_isQualified.value == 2:
                self.alert.value = 2
            else:
                self.alert.value = 0


class LeftCameraProcess(BasicCameraProcess):
    def __init__(self, ImageSignal: pyqtSignal, AlertSignal: pyqtSignal, Mxid, new_value, status, barcode,
                 command, left_location, left_isQualified, right_location, right_isQualified) -> None:
        super().__init__(ImageSignal, AlertSignal, Mxid, new_value, status, barcode, command, left_location, left_isQualified, right_location, right_isQualified)
        self.direction = 'left'

    def parse_left_result(self):
        result = self.recv_result.recv()
        logger.info(result)
        left_count = result['count']
        left_res = result['res']
        left_location = 3 in left_count
        left_pass = left_location and left_res[left_count.index(3)]
        if left_location:
            self.left_location.value = 1
            if left_pass:
                # self.left_send_save_frame.send(self.save_frame)
                self.left_isQualified.value = 1
            else:
                # self.left_send_save_frame.send(self.save_frame)
                self.left_isQualified.value = 2
        else:
            self.left_location.value = 2

        logger.info(
            '{} {} {} {}'.format(self.left_location.value, self.right_location.value, self.left_isQualified.value,
                                 self.right_isQualified.value))


class RightCameraProcess(BasicCameraProcess):
    def __init__(self, ImageSignal: pyqtSignal, AlertSignal: pyqtSignal, Mxid, new_value, status, barcode,
                 command, left_location, left_isQualified, right_location, right_isQualified) -> None:
        super().__init__(ImageSignal, AlertSignal, Mxid, new_value, status, barcode, command, left_location, left_isQualified, right_location, right_isQualified)
        self.direction = 'right'

    def parse_right_result(self):
        result = self.recv_result.recv()
        logger.info(result)
        right_count = result['count']
        right_res = result['res']
        right_location = 3 in right_count
        right_pass = right_location and right_res[right_count.index(3)]
        if right_location:
            self.right_location.value = 1
            if right_pass:
                # self.right_send_save_frame.send(self.save_frame)
                self.right_isQualified.value = 1
            else:
                # self.right_send_save_frame.send(self.save_frame)
                self.right_isQualified.value = 2
        else:
            self.right_location.value = 2
        logger.info('{} {} {} {}'.format(self.left_location.value, self.right_location.value, self.left_isQualified.value, self.right_isQualified.value))
