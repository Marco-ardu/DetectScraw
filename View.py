import sys
from typing import SupportsRound
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QUrl, qsrand
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtMultimedia import QSound, QSoundEffect

from PyQt5.QtWidgets import (
    QApplication, QDialog, QMainWindow, QMessageBox
)
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

    def UpdateFrontSlot(self, Image):
        self.LabelFront.setPixmap(QPixmap.fromImage(Image))

    def UpdateRearSlot(self, Image):
        self.LabelRear.setPixmap(QPixmap.fromImage(Image))

    def keyPressEvent(self, event):
        key = event.key()
        sound_file = 'sound/pedestrian.wav'
        QSound.play(sound_file)
        print(key)
        self.LabelPedestrian.setStyleSheet("background-color: yellow")