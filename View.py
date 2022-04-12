import cv2
import numpy as np
import yaml
from PyQt5.QtCore import Qt, pyqtSlot, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import QtGui
from PyQt5.QtMultimedia import QSound
from PyQt5.QtWidgets import (
    QMainWindow
)
from loguru import logger

from factories.AlertFactory import AlertEnum, AlertDict
from ui.ui_main import Ui_MainWindow

with open('config.yml', 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)


class ViewWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.defaultStyleSheet = "background-color: black; font-family:微軟正黑體; font-size:40pt;font-weight: bold; color:white"
        self.defaultFrontLabelText = "左相机"
        self.defaultRearLabelText = "右相机"
        self.defaultRemindLabelText = "消息提醒"
        self.defaultSettingLensPos_value = 156
        self.defaultSettingExp_time_value = 20000
        self.defaultSettingSens_ios_value = 800

    def setup(self, controller):
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("cam.ico"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        self.controller = controller
        self.sound_dict = self.setSounddict()
        self.btnStart.clicked.connect(controller.btnStart_clicked)
        self.btnStop.clicked.connect(controller.btnStop_clicked)
        self.ShutDown.clicked.connect(controller.shutdown)
        self.SaveButton.clicked.connect(controller.btn_save)
        self.left_lensPos_value.valueChanged.connect(controller.change_left_lenPos)
        self.left_exp_time_value.valueChanged.connect(controller.change_left_exp_time)
        self.left_sens_ios_value.valueChanged.connect(controller.change_left_sens_ios)
        self.right_lensPos_value.valueChanged.connect(controller.change_right_lenPos)
        self.right_exp_time_value.valueChanged.connect(controller.change_right_exp_time)
        self.right_sens_ios_value.valueChanged.connect(controller.change_right_sens_ios)
        self.autoexpleft.stateChanged.connect(controller.change_left_auto_exp)
        self.autofocusleft.stateChanged.connect(controller.change_left_auto_focus)
        self.autoexpright.stateChanged.connect(controller.change_right_auto_exp)
        self.autofocusright.stateChanged.connect(controller.change_right_auto_focus)
        self.left_lensPos_edit.textChanged.connect(controller.change_edit_value)
        self.left_sens_ios_edit.textChanged.connect(controller.change_edit_value)
        self.left_exp_time_edit.textChanged.connect(controller.change_edit_value)
        self.right_sens_ios_edit.textChanged.connect(controller.change_edit_value)
        self.right_exp_time_edit.textChanged.connect(controller.change_edit_value)
        self.right_lensPos_edit.textChanged.connect(controller.change_edit_value)
        self.BarCodeValue.setAttribute(Qt.WA_InputMethodEnabled, False)
        self.BarCodeValue.editingFinished.connect(self.controller.barcode_edit)
        self.btnAllScreen.clicked.connect(controller.setAllScreen)
        self.btnNoAllScreen.clicked.connect(controller.setNoAllScreen)
        self.btnOpenPath.clicked.connect(controller.clicked_Openpath)

    def setSounddict(self):
        sound_dict = {}
        for key in AlertDict:
            alert = AlertDict[key]
            sound_dict[key] = QSound(alert.warn_file)
        return sound_dict

    @pyqtSlot()
    def setDefaultView(self):
        self.LabelFront.clear()
        self.LabelRear.clear()
        self.left_lensPos_value.setValue(self.defaultSettingLensPos_value)
        self.left_exp_time_value.setValue(self.defaultSettingExp_time_value)
        self.left_sens_ios_value.setValue(self.defaultSettingSens_ios_value)
        self.right_lensPos_value.setValue(self.defaultSettingLensPos_value)
        self.right_exp_time_value.setValue(self.defaultSettingExp_time_value)
        self.right_sens_ios_value.setValue(self.defaultSettingSens_ios_value)
        self.LabelFront.setText(self.defaultFrontLabelText)
        self.LabelRear.setText(self.defaultRearLabelText)
        self.remind.setText(self.defaultRemindLabelText)
        self.remind.setStyleSheet(self.defaultStyleSheet) 
        self.left_exp_time_edit.setText(str(self.defaultSettingExp_time_value))
        self.right_exp_time_edit.setText(str(self.defaultSettingExp_time_value))
        self.left_sens_ios_edit.setText(str(self.defaultSettingSens_ios_value))
        self.right_sens_ios_edit.setText(str(self.defaultSettingSens_ios_value))
        self.left_lensPos_edit.setText(str(self.defaultSettingLensPos_value))
        self.right_lensPos_edit.setText(str(self.defaultSettingLensPos_value))

    @pyqtSlot(np.ndarray)
    def UpdateFrontSlot(self, Image):
        self.setImg(Image, self.LabelFront)

    @pyqtSlot(np.ndarray)
    def UpdateRearSlot(self, Image):
        self.setImg(Image, self.LabelRear)

    def keyPressEvent(self, event):
        self.BarCodeValue.setFocus()

    @pyqtSlot(AlertEnum)
    def runAlert(self, alertKey):
        for key in self.sound_dict:
            if not self.sound_dict[key].isFinished():
                return
        WarnAlert = AlertDict[alertKey]
        logger.info(alertKey)
        logger.info(WarnAlert.warn_file)
        current_sound = self.sound_dict[alertKey]
        current_sound.play()
        self.remind.setText(WarnAlert.warn_message)
        for i in range(0, 2400, 600):
            QTimer.singleShot((0.5 * i), lambda: self.remind.setStyleSheet(self.defaultStyleSheet.replace("black", WarnAlert.warn_color).replace('white', 'black')))
            QTimer.singleShot(i, lambda: self.remind.setStyleSheet(self.defaultStyleSheet))
        QTimer.singleShot(3000, lambda: self.remind.setStyleSheet(self.defaultStyleSheet.replace("black", WarnAlert.warn_color).replace('white', 'black')))

    def setImg(self, frame, label):
        Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Pic = QImage(Image.data, Image.shape[1],
                     Image.shape[0], QImage.Format_RGB888)
        h, w = label.size().height(), label.size().width()
        Pic = Pic.scaled(w, h, Qt.KeepAspectRatio)
        label.setPixmap(QPixmap.fromImage(Pic))
