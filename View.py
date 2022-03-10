import cv2
import numpy as np
import yaml
from PyQt5.QtCore import QTimer, Qt, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtMultimedia import QSound
from PyQt5.QtWidgets import (
    QMainWindow
)

from ui.ui_main_new import Ui_MainWindow

with open('config.yml', 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)


class ViewWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.defaultFrontLabelText = "左相机"
        self.defaultRearLabelText = "右相机"
        self.defaultSettingLensPos_value = 156
        self.defaultSettingExp_time_value = 20000
        self.defaultSettingSens_ios_value = 800

    def setup(self, controller):
        self.controller = controller
        self.btnStart.clicked.connect(controller.btnStart_clicked)
        self.btnStop.clicked.connect(controller.btnStop_clicked)
        self.ShutDown.clicked.connect(controller.shutdown)
        self.SaveButton.clicked.connect(controller.btn_save)
        self.lensPos_value.valueChanged.connect(controller.change_lenPos)
        self.exp_time_value.valueChanged.connect(controller.change_exp_time)
        self.sens_ios_value.valueChanged.connect(controller.change_sens_ios)
        self.leftCameraButton.toggled.connect(controller.change_checked_left)
        self.autoexp.clicked.connect(controller.change_auto_exp)
        self.autofocus.clicked.connect(controller.change_auto_focus)
        self.qs = QSound('sound/welcome.wav')
        if config["PRODUCTION"] is True:
            self.qs.play()

    @pyqtSlot()
    def setDefaultView(self):
        self.LabelFront.clear()
        self.LabelRear.clear()
        self.lensPos_value.setValue(self.defaultSettingLensPos_value)
        self.exp_time_value.setValue(self.defaultSettingExp_time_value)
        self.sens_ios_value.setValue(self.defaultSettingSens_ios_value)
        self.LabelFront.setText(self.defaultFrontLabelText)
        self.LabelRear.setText(self.defaultRearLabelText)
        self.leftCameraButton.setChecked(True)

    @pyqtSlot(np.ndarray)
    def UpdateFrontSlot(self, Image):
        self.setImg(Image, self.LabelFront)

    @pyqtSlot(np.ndarray)
    def UpdateRearSlot(self, Image):
        self.setImg(Image, self.LabelRear)

    def keyPressEvent(self, event):
        self.BarCodeValue.setFocus()
        self.BarCodeValue.editingFinished.connect(self.controller.barcode_edit)
        if event.key() == Qt.Key_Backspace:
            text = self.BarCodeValue.text()
            self.BarCodeValue.setText(text[:-1])
        elif event.key() == Qt.Key_Delete:
            self.BarCodeValue.clear()
        else:
            text = self.BarCodeValue.text()
            self.BarCodeValue.setText(text + event.text())

    def runAlert(self, WarnAlert):
        if not self.qs.isFinished():
            return

        # self.labelMessage.setText(WarnAlert.warn_message)
        sound_file = WarnAlert.warn_file
        self.qs = QSound(sound_file)
        self.qs.play()

    def setImg(self, frame, label):

        Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Pic = QImage(Image.data, Image.shape[1],
                     Image.shape[0], QImage.Format_RGB888)

        if config["PRODUCTION"] is not True:
            h, w = label.size().height(), label.size().width()
            Pic = Pic.scaled(w, h, Qt.KeepAspectRatio)

        # 如果有需要再獨立 目前先放在這一併執行
        label.setPixmap(QPixmap.fromImage(Pic))
