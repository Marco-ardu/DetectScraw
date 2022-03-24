import time
from PyQt5.QtWidgets import QApplication
from loguru import logger

from View import ViewWindow
from Worker import Worker

class MainController:
    def __init__(self, view: ViewWindow) -> None:
        self.view = view

    def start(self):
        self.view.setup(self)
        self.view.show() 
        # self.view.showFullScreen()
        self.view.app = QApplication.instance()  # 实例化APP，获取app的指针
        self.view.Worker = Worker()
        self.view.Worker.finished.connect(self.view.setDefaultView)

    def btnStart_clicked(self):
        self.view.btnStart.setEnabled(False)
        self.view.LabelFront.setText('加载中')
        self.view.LabelRear.setText('加载中')
        self.view.BarCodeValue.setFocus()
        self.view.Worker.start()
        self.view.Worker.FrontImage.connect(self.view.UpdateFrontSlot)
        self.view.Worker.RearImage.connect(self.view.UpdateRearSlot)
        self.view.Worker.Alert.connect(self.view.runAlert)

    def btnStop_clicked(self):
        self.view.Worker.stop()
        self.view.btnStart.setEnabled(True)

    def shutdown(self):
        self.view.Worker.stop()
        self.view.btnStart.setEnabled(True)
        logger.info('quit app')
        self.view.app.quit()

    def getConfig(self):
        left_lensPos = self.view.left_lensPos_value.value()
        left_exp_time = self.view.left_exp_time_value.value()
        left_sens_ios = self.view.left_sens_ios_value.value()
        right_lensPos = self.view.right_lensPos_value.value()
        right_exp_time = self.view.right_exp_time_value.value()
        right_sens_ios = self.view.right_sens_ios_value.value()
        return [left_lensPos, left_exp_time, left_sens_ios, right_lensPos, right_exp_time, right_sens_ios, self.view.Worker.Mxids]

    def btn_save(self):
        self.view.BarCodeValue.setFocus()
        try:
            self.view.Worker.save_yml(self.getConfig())
        except Exception as e:
            logger.error(e)

    def barcode_edit(self):
        barcode = self.view.BarCodeValue.text().strip()
        if len(barcode) != 0 and not self.view.btnStart.isEnabled():
            self.view.Worker.left_send_barcode.send(barcode)
            self.view.Worker.right_send_barcode.send(barcode)
            self.view.Worker.bar_code = barcode
            self.view.BarCodeValue.clear()

    def change_left_lenPos(self):
        self.view.BarCodeValue.setFocus()
        self.view.Worker.left_new_value['lenPos_new'].value = self.view.left_lensPos_value.value()

    def change_left_exp_time(self):
        self.view.BarCodeValue.setFocus()
        self.view.Worker.left_new_value['exp_time_new'].value = self.view.left_exp_time_value.value()

    def change_left_sens_ios(self):
        self.view.BarCodeValue.setFocus()
        self.view.Worker.left_new_value['sens_ios_new'].value = self.view.left_sens_ios_value.value()

    def change_right_lenPos(self):
        self.view.BarCodeValue.setFocus()
        self.view.Worker.right_new_value['lenPos_new'].value = self.view.right_lensPos_value.value()

    def change_right_exp_time(self):
        self.view.BarCodeValue.setFocus()
        self.view.Worker.right_new_value['exp_time_new'].value = self.view.right_exp_time_value.value()

    def change_right_sens_ios(self):
        self.view.BarCodeValue.setFocus()
        self.view.Worker.right_new_value['sens_ios_new'].value = self.view.right_sens_ios_value.value()

    def change_left_auto_exp(self):
        self.view.BarCodeValue.setFocus()
        if self.view.autoexpleft.isChecked():
            self.view.Worker.left_status['auto_exp_status'].value = 2
        else:
            self.view.Worker.left_status['auto_exp_status'].value = 1

    def change_left_auto_focus(self):
        self.view.BarCodeValue.setFocus()
        if self.view.autofocusleft.isChecked():
            self.view.Worker.left_status['auto_focus_status'].value = 2
        else:
            self.view.Worker.left_status['auto_focus_status'].value = 1
    
    def change_right_auto_exp(self):
        self.view.BarCodeValue.setFocus()
        if self.view.autoexpright.isChecked():
            self.view.Worker.right_status['auto_exp_status'].value = 2
        else:
            self.view.Worker.right_status['auto_exp_status'].value = 1

    def change_right_auto_focus(self):
        self.view.BarCodeValue.setFocus()
        if self.view.autofocusright.isChecked():
            self.view.Worker.right_status['auto_focus_status'].value = 2
        else:
            self.view.Worker.right_status['auto_focus_status'].value = 1