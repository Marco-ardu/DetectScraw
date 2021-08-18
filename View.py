from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtMultimedia import QSound
from PyQt5.QtWidgets import (
    QMainWindow
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.uic import loadUi
from ui.ui_qtcart import Ui_MainWindow

import cv2

class ViewWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
      

    def setup(self, controller):
        #self.showMaximized()
        #self.LabelFront.adjustSize()
        self.actionStart.triggered.connect(controller.btnStart_clicked)
        self.actionStop.triggered.connect(controller.btnStop_clicked)
        self.btnStart.clicked.connect(controller.btnStart_clicked)
        self.btnStop.clicked.connect(controller.btnStop_clicked)
        self.qs = QSound('sound/welcome.wav', parent=self.labelMessage)
        self.qs.play()
        

    def UpdateFrontSlot(self, Image):
        self.setImg(Image, self.LabelFront)

    def UpdateRearSlot(self, Image):
        self.setImg(Image, self.LabelRear)

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
            QTimer.singleShot((0.5 * i), lambda: self.labelMessage.setStyleSheet(f"background-color: {WarnAlert.warn_color}"))
            QTimer.singleShot(i, lambda: self.labelMessage.setStyleSheet(""))        

    def setImg(self, frame, label):
        h, w = label.size().height(), label.size().width()
        Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)
        Pic = ConvertToQtFormat.scaled(w,h, Qt.KeepAspectRatio)

        #如果有需要再獨立 目前先放在這一併執行
        label.setPixmap(QPixmap.fromImage(Pic))
