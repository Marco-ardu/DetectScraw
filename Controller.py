import time

from PyQt5.QtWidgets import QApplication

from View import ViewWindow
from Worker import Worker
from setDirection import isExist


class MainController:
    def __init__(self, view: ViewWindow) -> None:
        self.view = view

    def start(self):
        self.view.setup(self)
        self.view.show()
        # self.view.showFullScreen()
        self.view.getMxid = isExist
        self.view.app = QApplication.instance()  # 实例化APP，获取app的指针
        self.view.Worker = Worker(self.view.getMxid)
        self.view.Worker.finished.connect(self.view.setDefaultView)

    def btnStart_clicked(self):
        self.view.btnStart.setEnabled(False)
        self.view.LabelFront.setText('加载中')
        self.view.LabelRear.setText('加载中')
        self.view.Worker.start()
        self.view.Worker.FrontImage.connect(self.view.UpdateFrontSlot)
        self.view.Worker.RearImage.connect(self.view.UpdateRearSlot)
        # self.view.Worker.Alert.connect(self.view.runAlert)

    def btnStop_clicked(self):
        self.view.Worker.stop()
        self.view.btnStart.setEnabled(True)

    def shutdown(self):
        self.view.Worker.stop()
        self.view.btnStart.setEnabled(True)
        self.view.app.quit()

    def getConfig(self):
        lensPos = self.view.lensPos_value.value()
        exp_time = self.view.exp_time_value.value()
        sens_ios = self.view.sens_ios_value.value()
        left_checked = self.view.leftCameraButton.isChecked()
        right_checked = self.view.rightCameraButton.isChecked()
        return [lensPos, exp_time, sens_ios, left_checked, right_checked, self.view.Worker.Mxids]

    def btn_save(self):
        try:
            self.view.Worker.save_yml(self.getConfig())
            if self.view.leftCameraButton.isChecked():
                self.view.rightCameraButton.setChecked(True)
            else:
                self.view.leftCameraButton.setChecked(True)
        except:
            pass

    def change_lenPos(self):
        self.view.Worker.lenPos_new.value = self.view.lensPos_value.value()

    def change_exp_time(self):
        self.view.Worker.exp_time_new.value = self.view.exp_time_value.value()

    def change_sens_ios(self):
        self.view.Worker.sens_ios_new.value = self.view.sens_ios_value.value()

    def change_checked_left(self):
        if self.view.leftCameraButton.isChecked():
            self.view.Worker.setting_status.value = 1
        else:
            self.view.Worker.setting_status.value = 2

    def barcode_edit(self):
        barcode = self.view.BarCodeValue.text().strip()
        if len(barcode) != 0:
            self.view.Worker.barcode.put(barcode)
            self.view.BarCodeValue.clear()
            self.view.Worker.barcode.get()

    def change_auto_exp(self):
        if self.view.autoexp.isChecked():
            self.view.Worker.auto_exp_status.value = 1
        else:
            self.view.Worker.auto_exp_status.value = 2

    def change_auto_focus(self):
        if self.view.autofocus.isChecked():
            self.view.Worker.auto_focus_status.value = 1
        else:
            self.view.Worker.auto_focus_status.value = 2
