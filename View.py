import time
from PyQt5.QtGui import QPixmap
from PyQt5.QtMultimedia import QSound
from PyQt5.QtWidgets import (
    QMainWindow
)
from PyQt5.QtCore import QTimer, QEventLoop
from PyQt5.uic import loadUi
from ui.ui_qtcart import Ui_MainWindow

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
        self.qs = QSound('sound/focus.wav', parent=self.labelMessage)
        

    def UpdateFrontSlot(self, Image):
        self.LabelFront.setPixmap(QPixmap.fromImage(Image))

    def UpdateRearSlot(self, Image):
        self.LabelRear.setPixmap(QPixmap.fromImage(Image))

    def keyPressEvent(self, event):
        key = event.key()
        print(key)
        print(self.qs.isFinished())
        self.qs.play()
        print(self.qs.isFinished())
        QTimer.singleShot(500, lambda: self.LabelPedestrian.setStyleSheet("background-color: yellow"))
        QTimer.singleShot(1000, lambda: self.LabelPedestrian.setStyleSheet(""))
        

    def runAlert(self, WarnAlert):
        
        if not self.qs.isFinished():
            return
        
        self.labelMessage.setText(WarnAlert.warn_message)
        sound_file = WarnAlert.warn_file
        self.qs = QSound(sound_file, parent=self.labelMessage)
        self.qs.play()

        for i in range(0, 1200, 600):
            QTimer.singleShot((0.5 * i), lambda: self.labelMessage.setStyleSheet(f"background-color: {WarnAlert.warn_color}"))
            QTimer.singleShot(i, lambda: self.labelMessage.setStyleSheet(""))        

