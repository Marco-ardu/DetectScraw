from re import S
from PyQt5.QtCore import QThread, pyqtSignal

import multiprocessing as mp
import queue
import cv2
import numpy

import DETECTION_CONFIG

from cameraFunc import YoloCamera, FatigueCam, PedestrianCamera

from Model import WarnAlert, PedestrianAlert, DriverAlert


class Worker(QThread):
    FrontImage = pyqtSignal(numpy.ndarray)
    RearImage = pyqtSignal(numpy.ndarray)
    DriverImage = pyqtSignal(numpy.ndarray)
    Alert = pyqtSignal(WarnAlert)

    command = mp.Value('i', 1)

    def run(self):
        self.command = mp.Value('i', 1)
        driver_alert = mp.Value('i', 0)
        front_alert_value = mp.Value('i', 0)
        rear_alert_value = mp.Value('i', 0)
        driver_queue = mp.Queue(4)
        front_queue = mp.Queue(4)
        rear_queue = mp.Queue(4)

        driver_proccess = mp.Process(target=FatigueCam.runFatigueCam, args=(driver_queue, self.command,driver_alert, ))
        front_proccess = mp.Process(target=YoloCamera.runYoloCamera, args=(front_queue, self.command, front_alert_value))
        rear_proccess = mp.Process(target=PedestrianCamera.runPedestrianCamera, args=(rear_queue, self.command, rear_alert_value))

        driver_proccess.start()
        front_proccess.start()
        rear_proccess.start()

        self.ThreadActive = True

        while self.ThreadActive:
            try:
                driver_frame = driver_queue.get_nowait()
                self.DriverImage.emit(driver_frame)
            except queue.Empty or queue.Full:
                pass

            try:
                front_frame = front_queue.get_nowait()
                self.FrontImage.emit(front_frame)
            except queue.Empty or queue.Full:
                pass

            try:
                rear_frame = rear_queue.get_nowait()
                self.RearImage.emit(rear_frame)
            except queue.Empty or queue.Full:
                pass

            if front_alert_value.value != DETECTION_CONFIG.NO_ALERT_SIGNAL:
                p = PedestrianAlert()
                p = AlertFactory(p, front_alert_value.value)
                self.Alert.emit(p)
                front_alert_value.value = DETECTION_CONFIG.NO_ALERT_SIGNAL

            if rear_alert_value.value != DETECTION_CONFIG.NO_ALERT_SIGNAL:
                p = PedestrianAlert()
                p = AlertFactory(p, rear_alert_value.value)
                self.Alert.emit(p)
                rear_alert_value.value = DETECTION_CONFIG.NO_ALERT_SIGNAL

            if driver_alert.value != DETECTION_CONFIG.NO_ALERT_SIGNAL:
                d = DriverAlert()
                d = AlertFactory(d, driver_alert.value)
                self.Alert.emit(d)
                driver_alert.value = DETECTION_CONFIG.NO_ALERT_SIGNAL

        driver_queue.close()
        front_queue.close()
        rear_queue.close()

        # normally dont just kill
        driver_proccess.kill()
        front_proccess.kill()
        rear_proccess.kill()

        self.quit()

    def stop(self):
        self.command.value = 0
        self.ThreadActive = False


def AlertFactory(WarnAlert, alert_level):
    if alert_level == DETECTION_CONFIG.YELLOW_ALERT_SIGNAL:
        WarnAlert.yellowAlert()
    elif alert_level == DETECTION_CONFIG.RED_ALERT_SIGNAL:
        WarnAlert.redAlert()

    return WarnAlert
