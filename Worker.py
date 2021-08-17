from re import S
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage

import multiprocessing as mp
import queue
import cv2

import config

from cameraFunc import YoloCamera, FatigueCam

from Model import WarnAlert, PedestrianAlert, DriverAlert



class Worker(QThread):
    FrontImage = pyqtSignal(QImage)
    RearImage = pyqtSignal(QImage)
    Alert = pyqtSignal(WarnAlert)

    command = mp.Value('i', 1)

    def run(self):
        self.command = mp.Value('i', 1)
        front_alert_value = mp.Value('i', 0)
        rear_alert_value = mp.Value('i', 0)
        front_queue = mp.Queue(4)
        rear_queue = mp.Queue(4)

        front_proccess = mp.Process(target=FatigueCam.runFatigueCam, args=(front_queue, self.command,front_alert_value, ))
        rear_proccess = mp.Process(target=YoloCamera.runYoloCamera, args=(rear_queue, self.command, rear_alert_value))        
        front_proccess.start()
        rear_proccess.start()
        self.ThreadActive = True
        
        while self.ThreadActive:
            try:
                front_frame = front_queue.get_nowait()
                front_Pic = frameToPic(front_frame)
                self.FrontImage.emit(front_Pic)
            except queue.Empty or queue.Full:
                pass

            try:
                rear_frame = rear_queue.get_nowait()
                rear_Pic = frameToPic(rear_frame)
                self.RearImage.emit(rear_Pic)
            except queue.Empty or queue.Full:
                pass

            if rear_alert_value.value != config.NO_ALERT_SIGNAL:
                p = PedestrianAlert()
                p = AlertFactory(p, rear_alert_value.value)
                self.Alert.emit(p)
                rear_alert_value.value = config.NO_ALERT_SIGNAL

            if front_alert_value.value != config.NO_ALERT_SIGNAL:
                d = DriverAlert()
                d = AlertFactory(d, front_alert_value.value)
                self.Alert.emit(d)
                front_alert_value.value = config.NO_ALERT_SIGNAL

        front_queue.close()
        rear_queue.close()       

        # normally dont just kill
        front_proccess.kill()
        rear_proccess.kill()           

        self.quit()

    def stop(self):
        self.command.value = 0
        self.ThreadActive = False      
        

def frameToPic(frame):
    Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)
    Pic = ConvertToQtFormat.scaled(400,250, Qt.KeepAspectRatio)
    return Pic


def AlertFactory(WarnAlert, alert_level):
    if alert_level == config.YELLOW_ALERT_SIGNAL:
        WarnAlert.yellowAlert()
    elif alert_level == config.RED_ALERT_SIGNAL:
        WarnAlert.redAlert()

    return WarnAlert
