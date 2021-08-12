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
        self.loop = QEventLoop()
        

    def setup(self, controller):
        #self.showMaximized()
        #self.LabelFront.adjustSize()
        self.actionStart.triggered.connect(controller.btnStart_clicked)
        self.actionStop.triggered.connect(controller.btnStop_clicked)
        self.btnStart.clicked.connect(controller.btnStart_clicked)
        self.btnStop.clicked.connect(controller.btnStop_clicked)

    def UpdateFrontSlot(self, Image):
        self.LabelFront.setPixmap(QPixmap.fromImage(Image))

    def UpdateRearSlot(self, Image):
        self.LabelRear.setPixmap(QPixmap.fromImage(Image))

    def keyPressEvent(self, event):
        key = event.key()
        print(key)
        self.LabelPedestrian.setStyleSheet("background-color: yellow")
        QTimer.singleShot(2000, self.loop.quit)
        self.loop.exec_()
        self.LabelPedestrian.setStyleSheet("")

    def runAlert(self, alertStr):
        print(alertStr)
        sound_file = 'sound/focus.wav'
        QSound.play(sound_file)
        self.LabelDriver.setStyleSheet("background-color: yellow")
        QTimer.singleShot(2000, self.loop.quit)
        self.loop.exec_()
        self.LabelDriver.setStyleSheet("")
