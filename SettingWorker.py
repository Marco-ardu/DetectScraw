import multiprocessing as mp

import numpy
from PyQt5.QtCore import QThread, pyqtSignal

from demo_utils import save_yml
from factories import CameraFactory
from model.SettingProccessModel import SettingCameraProcess


class SettingWorker(QThread):
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
    status = mp.Value('i', 1)

    def __init__(self, Mxid):
        super().__init__()
        self.old_value = {'lenPos_old': self.lenPos_old, 'exp_time_old': self.exp_time_old,
                          'sens_ios_old': self.sens_ios_old}
        self.new_value = {'lenPos_new': self.lenPos_new, 'exp_time_new': self.exp_time_new,
                          'sens_ios_new': self.sens_ios_new}
        self.Mxid = Mxid
        self.save_yml = save_yml

    def run(self):
        self.command.value = 1
        LeftCamera = CameraFactory.CameraFactory(CameraFactory.TextCamera)
        Left_Camera = SettingCameraProcess(self.command, LeftCamera, self.FrontImage, self.Mxid[0],
                                           self.repeat_times, self.new_value,
                                           self.old_value, 'left', self.status)
        RightCamera = CameraFactory.CameraFactory(CameraFactory.TextCamera)
        Right_Camera = SettingCameraProcess(self.command, RightCamera, self.RearImage, self.Mxid[1],
                                            self.repeat_times, self.new_value,
                                            self.old_value, 'right', self.status)

        Cameras = [Left_Camera, Right_Camera]

        for Camera in Cameras:
            Camera.runCamera()
        self.ThreadActive = True

        while self.ThreadActive:
            for Camera in Cameras:
                Camera.getFrame()

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
        self.status.value = 1
        self.ThreadActive = False

    def save(self, config):
        self.save_yml(config)

    def set_lenPos(self, value):
        self.lenPos_new.value = value

    def set_exp_time(self, value):
        self.exp_time_new.value = value

    def set_sens_ios(self, value):
        self.sens_ios_new.value = value

    def change_status(self, value):
        self.status.value = value
