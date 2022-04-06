import multiprocessing as mp
import time
import traceback

import cv2
import numpy
from PyQt5.QtCore import QThread, pyqtSignal
from loguru import logger

from demo_utils import isExist, save_to_picture, save_yml
from factories.AlertFactory import AlertEnum
from model.ProccessModel import LeftCameraProcess, RightCameraProcess


class Worker(QThread):
    FrontImage = pyqtSignal(numpy.ndarray)
    RearImage = pyqtSignal(numpy.ndarray)
    Alert = pyqtSignal(AlertEnum)

    def __init__(self):
        super().__init__()

        self.left_new_value = {'lenPos_new': mp.Value('Q', 156), 'exp_time_new': mp.Value('Q', 20000),
                               'sens_ios_new': mp.Value('Q', 800)}
        self.right_new_value = {'lenPos_new': mp.Value('Q', 156), 'exp_time_new': mp.Value('Q', 20000),
                                'sens_ios_new': mp.Value('Q', 800)}
        self.left_status = {'auto_exp_status': mp.Value('i', 2), 'auto_focus_status': mp.Value('i', 2)}
        self.right_status = {'auto_exp_status': mp.Value('i', 2), 'auto_focus_status': mp.Value('i', 2)}
        self.command = mp.Value('i', 0)
        self.right_location = mp.Value('i', 2)
        self.right_isQualified = mp.Value('i', 2)
        self.left_location = mp.Value('i', 2)
        self.left_isQualified = mp.Value('i', 2)
        self.left_send_barcode, self.left_recv_barcode = mp.Pipe()
        self.right_send_barcode, self.right_recv_barcode = mp.Pipe()
        self.Mxid = isExist
        self.save_yml = save_yml
        self.ThreadActive = True
        self.leftActive = False
        self.rightActive = False
        self.bar_code = ''
        self.save_left_frame = ''
        self.save_right_frame = ''

    def run(self):
        self.ThreadActive = True
        logger.info('loading camera')
        time.sleep(6)
        self.Mxids = self.Mxid()
        try:
            LeftCamera = LeftCameraProcess(self.FrontImage, self.Alert, self.Mxids[0], self.left_new_value, self.left_status,
                                           self.left_recv_barcode, self.command, self.left_location,
                                           self.left_isQualified, self.right_location, self.right_isQualified)
            RightCamera = RightCameraProcess(self.RearImage, self.Alert, self.Mxids[1], self.right_new_value, self.right_status,
                                             self.right_recv_barcode, self.command, self.left_location,
                                             self.left_isQualified, self.right_location, self.right_isQualified)
            Cameras = [LeftCamera, RightCamera]

            for Camera in Cameras:
                Camera.runCamera()

            while self.ThreadActive:
                if LeftCamera.recv_result.poll():
                    logger.info('left poll')
                    self.leftActive = True
                    LeftCamera.parse_left_result()
                left_frame = LeftCamera.getFrame()
                if left_frame is not None:
                    self.save_left_frame = left_frame
                LeftCamera.getAlert()
                if RightCamera.recv_result.poll():
                    logger.info('right poll')
                    self.rightActive = True
                    RightCamera.parse_right_result()
                right_frame = RightCamera.getFrame()
                if right_frame is not None:
                    self.save_right_frame = right_frame
                RightCamera.getAlert()
                if self.leftActive and self.rightActive:
                    self.leftActive = False
                    self.rightActive = False
                    RightCamera.setAlert()
                    if self.left_location.value == 1 and self.right_location.value == 1:
                        save_to_picture(self.left_isQualified.value == 1 and self.right_isQualified.value == 1,
                                        cv2.hconcat([self.save_left_frame, self.save_right_frame]),
                                        numbering=self.bar_code
                                        )

            for Camera in Cameras:
                Camera.endCamera()

            self.quit()
        except Exception as e:
            print(traceback.print_exc())
            logger.error(e)

    def stop(self):
        self.ThreadActive = False
