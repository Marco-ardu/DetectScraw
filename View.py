import sys
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap

from PyQt5.QtWidgets import (
    QApplication, QDialog, QMainWindow, QMessageBox
)
from PyQt5.uic import loadUi
from ui_qtcart import Ui_MainWindow

class ViewWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

    def setup(self, controller):
        self.btnStart.clicked.connect(controller.btnStart_clicked)
        self.btnStop.clicked.connect(controller.btnStop_clicked)

    def ImageUpdateSlot(self, Image):
        self.ImageLabel.setPixmap(QPixmap.fromImage(Image))