from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtMultimedia import QSound
from PyQt5.QtWidgets import (
    QMainWindow
)
from PyQt5.QtCore import QTimer, Qt, pyqtSlot
from PyQt5.uic import loadUi
from ui.ui_qtcart import Ui_MainWindow

import numpy as np
import cv2

import PRODUCTION_CONFIG


class ViewWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self.defaultStyleSheet = "background-color: black; font-family:微軟正黑體; font-size:40pt;font-weight: bold; color:white"
        self.defaultWarnMessage = "警示訊息"
        self.defaultFrontLabelText = "前鏡頭"
        self.defaultRearLabelText = "後鏡頭"
        self.defaultDriverLabelText = "駕駛鏡頭"

    def setup(self, controller):
        self.btnStart.clicked.connect(controller.btnStart_clicked)
        self.btnStop.clicked.connect(controller.btnStop_clicked)
        self.qs = QSound('sound/welcome.wav', parent=self.labelMessage)
        if PRODUCTION_CONFIG.PRODUCTION is True:
            self.showMaximized()
            # self.LabelFront.setStyleSheet("background-color: yellow")
            # self.LabelRear.setStyleSheet("background-color: red")
            self.qs.play()

    def setDefaultView(self):

        self.LabelFront.clear()
        self.LabelRear.clear()
        self.LabelDriver.clear()

        self.LabelFront.setText(self.defaultFrontLabelText)
        self.LabelRear.setText(self.defaultRearLabelText)
        self.LabelDriver.setText(self.defaultDriverLabelText)
        self.labelMessage.setText(self.defaultWarnMessage)

    @pyqtSlot(np.ndarray)
    def UpdateFrontSlot(self, Image):
        self.setImg(Image, self.LabelFront)

    @pyqtSlot(np.ndarray)
    def UpdateRearSlot(self, Image):
        self.setImg(Image, self.LabelRear)

    @pyqtSlot(np.ndarray)
    def UpdateDriverSlot(self, Image):
        self.setImg(Image, self.LabelDriver)

    def keyPressEvent(self, event):
        key = event.key()
        print(key)

    def runAlert(self, WarnAlert):
        if not self.qs.isFinished():
            return

        self.labelMessage.setText(WarnAlert.warn_message)
        sound_file = WarnAlert.warn_file
        self.qs = QSound(sound_file, parent=self.labelMessage)
        self.qs.play()

        for i in range(0, 1800, 600):
            QTimer.singleShot((0.5 * i), lambda: self.labelMessage.setStyleSheet(
                f"background-color: {WarnAlert.warn_color}; font-family:微軟正黑體; font-size:40pt;font-weight: bold;"))
            QTimer.singleShot(
                i, lambda: self.labelMessage.setStyleSheet(self.defaultStyleSheet))

    def setImg(self, frame, label):

        Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Pic = QImage(Image.data, Image.shape[1],
                     Image.shape[0], QImage.Format_RGB888)

        if PRODUCTION_CONFIG.PRODUCTION is not True:
            h, w = label.size().height(), label.size().width()
            Pic = Pic.scaled(w, h, Qt.KeepAspectRatio)

        # 如果有需要再獨立 目前先放在這一併執行
        label.setPixmap(QPixmap.fromImage(Pic))
