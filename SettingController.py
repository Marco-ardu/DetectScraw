import time

from Setting_View import ViewSettingDialog
from setDirection import getCameraMxid
from SettingWorker import SettingWorker


class SettingController:
    def __init__(self, view: ViewSettingDialog) -> None:
        self.view = view

    def start(self):
        self.view.getMxid = getCameraMxid
        self.view.setup(self)
        self.view.show()
        self.startWorker()

    def GetCameraList(self):
        return self.view.getMxid()

    def getConfig(self):
        lensPos = self.view.lensPos_value.value()
        exp_time = self.view.exp_time_value.value()
        sens_ios = self.view.sens_ios_value.value()
        left_checked = self.view.leftCameraButton.isChecked()
        right_checked = self.view.rightCameraButton.isChecked()
        Mxid = self.view.CameraList.currentText()
        return [lensPos, exp_time, sens_ios, left_checked, right_checked, Mxid]

    def startWorker(self):
        self.view.Worker = SettingWorker(self.view.CameraList.currentText())
        self.view.Worker.finished.connect(self.view.setDefaultView)
        self.view.Worker.start()
        self.view.Worker.CameraImage.connect(self.view.UpdateCamreaSlot)

    def btn_reboot(self):
        if self.view.leftCameraButton.isChecked():
            self.view.rightCameraButton.setChecked(True)
        else:
            self.view.leftCameraButton.setChecked(True)
        self.view.Worker.stop()
        time.sleep(1)
        self.startWorker()

    def btn_stop(self):
        self.view.Worker.stop()
        self.view.CameraImage.setText(self.view.defaultSettingCameraText)

    def btn_save(self):
        self.view.Worker.save(self.getConfig())

    def change_lenPos(self):
        self.view.Worker.set_lenPos(self.view.lensPos_value.value())

    def change_exp_time(self):
        self.view.Worker.set_exp_time(self.view.exp_time_value.value())

    def change_sens_ios(self):
        self.view.Worker.set_sens_ios(self.view.sens_ios_value.value())
