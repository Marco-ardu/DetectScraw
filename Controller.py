from SettingWorker import SettingWorker
from View import ViewWindow
from Worker import Worker
from setDirection import isExist


class MainController:
    def __init__(self, view: ViewWindow) -> None:
        self.view = view

    def start(self):
        self.view.setup(self)
        # self.view.show()
        self.view.showFullScreen()
        self.view.getMxid = isExist
        self.view.Worker = Worker()
        self.view.Worker.finished.connect(self.view.setDefaultView)
        self.view.SettingWorker = SettingWorker(self.view.getMxid())
        self.view.SettingWorker.finished.connect(self.view.setDefaultView)

    def btnStart_clicked(self):
        self.view.btnStart.setEnabled(False)
        self.view.Worker.start()
        self.view.Worker.FrontImage.connect(self.view.UpdateFrontSlot)
        self.view.Worker.RearImage.connect(self.view.UpdateRearSlot)
        self.view.Worker.Alert.connect(self.view.runAlert)

    def btnStop_clicked(self):
        self.view.Worker.stop()
        self.view.SettingWorker.stop()
        self.view.btnStart.setEnabled(True)

    def startSetting(self):
        self.view.SettingWorker.start()
        self.view.SettingWorker.FrontImage.connect(self.view.UpdateFrontSlot)
        self.view.SettingWorker.RearImage.connect(self.view.UpdateRearSlot)

    def getConfig(self):
        lensPos = self.view.lensPos_value.value()
        exp_time = self.view.exp_time_value.value()
        sens_ios = self.view.sens_ios_value.value()
        left_checked = self.view.leftCameraButton.isChecked()
        right_checked = self.view.rightCameraButton.isChecked()
        return [lensPos, exp_time, sens_ios, left_checked, right_checked, self.view.SettingWorker.Mxid]

    def btn_save(self):
        self.view.SettingWorker.save(self.getConfig())
        self.view.btnStart.setEnabled(True)

    def change_lenPos(self):
        self.view.SettingWorker.set_lenPos(self.view.lensPos_value.value())

    def change_exp_time(self):
        self.view.SettingWorker.set_exp_time(self.view.exp_time_value.value())

    def change_sens_ios(self):
        self.view.SettingWorker.set_sens_ios(self.view.sens_ios_value.value())

    def change_checked_left(self):
        if self.view.leftCameraButton.isChecked():
            self.view.SettingWorker.change_status(1)
        else:
            self.view.SettingWorker.change_status(2)

