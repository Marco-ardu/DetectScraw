import cv2
import numpy as np
import yaml
from PyQt5.QtCore import QTimer, Qt, pyqtSlot
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QDialog
)

from ui.ui_settings import Ui_DialogSettings

with open('config.yml', 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)


class ViewSettingDialog(QDialog, Ui_DialogSettings):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.defaultSettingCameraText = "相机"
        self.defaultSettingLensPos_value = 156
        self.defaultSettingExp_time_value = 20000
        self.defaultSettingSens_ios_value = 800

    def setup(self, contorller):
        self.CameraList.clear()
        for i, j in enumerate(contorller.GetCameraList()):
            self.CameraList.insertItem(i, j)
        self.rebootCamera.clicked.connect(contorller.btn_reboot)
        self.ShutDownButton.clicked.connect(contorller.btn_stop)
        self.SaveButton.clicked.connect(contorller.btn_save)
        self.lensPos_value.valueChanged.connect(contorller.change_lenPos)
        self.exp_time_value.valueChanged.connect(contorller.change_exp_time)
        self.sens_ios_value.valueChanged.connect(contorller.change_sens_ios)

    @pyqtSlot()
    def setDefaultView(self):
        self.CameraImage.clear()
        self.lensPos_value.setValue(self.defaultSettingLensPos_value)
        self.exp_time_value.setValue(self.defaultSettingExp_time_value)
        self.sens_ios_value.setValue(self.defaultSettingSens_ios_value)

    @pyqtSlot(np.ndarray)
    def UpdateCamreaSlot(self, Image):
        self.setImg(Image, self.CameraImage)

    def setImg(self, frame, label):

        Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Pic = QImage(Image.data, Image.shape[1],
                     Image.shape[0], QImage.Format_RGB888)

        if config["PRODUCTION"] is not True:
            h, w = label.size().height(), label.size().width()
            Pic = Pic.scaled(w, h, Qt.KeepAspectRatio)

        # 如果有需要再獨立 目前先放在這一併執行
        label.setPixmap(QPixmap.fromImage(Pic))
