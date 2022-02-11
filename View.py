import cv2
import numpy as np
import yaml
from PyQt5.QtCore import QTimer, Qt, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtMultimedia import QSound
from PyQt5.QtWidgets import (
    QMainWindow
)

from ui.uiqtcart_new import Ui_MainWindow

with open('config.yml', 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)


class ViewWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        self.defaultStyleSheet = "background-color: black; font-family:微軟正黑體; font-size:40pt;font-weight: bold; " \
                                 "color:white "
        self.defaultWarnMessage = "消息提示"
        self.defaultFrontLabelText = "左镜头"
        self.defaultRearLabelText = "右镜头"

    def setup(self, controller):
        self.btnStart.clicked.connect(controller.btnStart_clicked)
        self.btnStop.clicked.connect(controller.btnStop_clicked)
        self.qs = QSound('sound/welcome.wav', parent=self.labelMessage)
        if config["PRODUCTION"] is True:
            self.qs.play()

    @pyqtSlot()
    def setDefaultView(self):
        self.LabelFront.clear()
        self.LabelRear.clear()
        self.LabelFront.setText(self.defaultFrontLabelText)
        self.LabelRear.setText(self.defaultRearLabelText)
        self.labelMessage.setText(self.defaultWarnMessage)

    @pyqtSlot(np.ndarray)
    def UpdateFrontSlot(self, Image):
        self.setImg(Image, self.LabelFront)

    @pyqtSlot(np.ndarray)
    def UpdateRearSlot(self, Image):
        self.setImg(Image, self.LabelRear)

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

        if config["PRODUCTION"] is not True:
            h, w = label.size().height(), label.size().width()
            Pic = Pic.scaled(w, h, Qt.KeepAspectRatio)

        # 如果有需要再獨立 目前先放在這一併執行
        label.setPixmap(QPixmap.fromImage(Pic))
