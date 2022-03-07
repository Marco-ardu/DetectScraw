import multiprocessing as mp
import time

import keyboard
import numpy
from PyQt5.QtCore import QThread, pyqtSignal

from demo_utils import save_yml
from factories import CameraFactory
from model.AlertModel import WarnAlert
from model.ProccessModel import BasicCameraProcess


class Worker(QThread):
    FrontImage = pyqtSignal(numpy.ndarray)
    RearImage = pyqtSignal(numpy.ndarray)
    command = mp.Value('i', 0)
    repeat_times = mp.Value('i', 0)
    lenPos_new = mp.Value('Q', 156)
    exp_time_new = mp.Value('Q', 20000)
    sens_ios_new = mp.Value('Q', 800)
    lenPos_old = mp.Value('Q', 156)
    exp_time_old = mp.Value('Q', 20000)
    sens_ios_old = mp.Value('Q', 800)
    setting_status = mp.Value('i', 1)
    auto_exp_status = mp.Value('i', 1)
    auto_focus_status = mp.Value('i', 1)
    barcode = mp.Queue(1)

    def __init__(self, Mxid):
        super().__init__()
        self.old_value = {'lenPos_old': self.lenPos_old, 'exp_time_old': self.exp_time_old,
                          'sens_ios_old': self.sens_ios_old}
        self.new_value = {'lenPos_new': self.lenPos_new, 'exp_time_new': self.exp_time_new,
                          'sens_ios_new': self.sens_ios_new}
        self.status = {'setting_status': self.setting_status, 'auto_exp_status': self.auto_exp_status,
                       'auto_focus_status': self.auto_focus_status}
        self.Mxid = Mxid
        self.save_yml = save_yml

    def run(self):
        self.command.value = 1
        time.sleep(6)
        self.Mxids = self.Mxid()

        DetectScrawLeftCamera = CameraFactory.CameraFactory(CameraFactory.TextDetectScrawCamera)
        LeftCamera = BasicCameraProcess(self.command, DetectScrawLeftCamera, self.FrontImage, self.Mxids[0],
                                        self.repeat_times, self.new_value, self.old_value, 'left', self.status,
                                        self.barcode)
        DetectScrawRightCamera = CameraFactory.CameraFactory(CameraFactory.TextDetectScrawCamera)
        RightCamera = BasicCameraProcess(self.command, DetectScrawRightCamera, self.RearImage, self.Mxids[1],
                                         self.repeat_times, self.new_value, self.old_value, 'right', self.status,
                                         self.barcode)

        Cameras = [LeftCamera, RightCamera]

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
        self.repeat_times.value = 0
        self.command.value = 0
        self.lenPos_new.value = 156
        self.exp_time_new.value = 20000
        self.sens_ios_new.value = 800
        self.lenPos_old.value = 156
        self.exp_time_old.value = 20000
        self.sens_ios_old.value = 800
        self.setting_status.value = 1
        self.auto_exp_status.value = 1
        self.auto_focus_status.value = 1
        self.ThreadActive = False

    # def save(self, config):
    #     self.save_yml(config)

    # def set_lenPos(self, value):
    #     self.lenPos_new.value = value
    #
    # def set_exp_time(self, value):
    #     self.exp_time_new.value = value
    #
    # def set_sens_ios(self, value):
    #     self.sens_ios_new.value = value
    #
    # def change_status(self, value):
    #     self.setting_status.value = value
    #
    # def barcode_value(self, value):
    #     self.barcode.put(value)
    #
    # def get_barcode(self):
    #     print(self.barcode.get())
    #     # if not self.barcode.empty():
    #     #     text = self.barcode.get_nowait()
    #     #     print(text)
    #
    # def change_exp_status(self, value):
    #     self.auto_exp_status.value = value
    #
    # def change_focus_status(self, value):
    #     self.auto_focus_status.value = value
