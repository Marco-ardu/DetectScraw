import multiprocessing as mp
import time

import keyboard
import numpy
from PyQt5.QtCore import QThread, pyqtSignal
from loguru import logger
from demo_utils import save_yml
from factories.AlertFactory import AlertEnum
from model.AlertModel import WarnAlert
from model.ProccessModel import LeftCameraProcess, RightCameraProcess
from setDirection import isExist


class Worker(QThread):
    FrontImage = pyqtSignal(numpy.ndarray)
    RearImage = pyqtSignal(numpy.ndarray)
    Alert = pyqtSignal(AlertEnum)

    def __init__(self):
        super().__init__()
        self.new_value = {'lenPos_new': mp.Value('Q', 156), 'exp_time_new': mp.Value('Q', 20000),
                          'sens_ios_new': mp.Value('Q', 800)}
        self.status = {'auto_exp_status': mp.Value('i', 1), 'auto_focus_status': mp.Value('i', 1)}
        self.command = mp.Value('i', 0)
        self.barcode = mp.Queue(4)
        self.Mxid = isExist
        self.save_yml = save_yml

    def run(self):
        time.sleep(4)
        self.Mxids = self.Mxid()
        try:
            LeftCamera = LeftCameraProcess(self.FrontImage, self.Alert, self.Mxids[0], self.new_value, self.status, self.barcode, self.command)
            RightCamera = RightCameraProcess(self.RearImage, self.Alert, self.Mxids[1], self.new_value, self.status, self.barcode, self.command)
            Cameras = [LeftCamera, RightCamera]

            for Camera in Cameras:
                Camera.runCamera()

            self.ThreadActive = True

            while self.ThreadActive:
                for Camera in Cameras:
                    Camera.getFrame()
                    Camera.getAlert()

            for Camera in Cameras:
                logger.info('stop camera')
                Camera.endCamera()

            self.quit()
        except Exception as e:
            logger.error(e)
            self.run()

    def stop(self):
        self.ThreadActive = False
