from View import ViewWindow
from Worker import Worker

class MainController:
    def __init__(self, view: ViewWindow) -> None:
        self.view = view

    def start(self):
        self.view.setup(self)
        self.view.show()
        self.view.Worker = Worker()
        self.view.Worker.finished.connect(self.view.setDefaultView)

    def btnStart_clicked(self):
        self.view.btnStart.setEnabled(False)
        self.view.Worker.start()
        self.view.Worker.FrontImage.connect(self.view.UpdateFrontSlot)
        self.view.Worker.RearImage.connect(self.view.UpdateRearSlot)
        self.view.Worker.DriverImage.connect(self.view.UpdateDriverSlot)
        self.view.Worker.Alert.connect(self.view.runAlert)

    def btnStop_clicked(self):
        self.view.Worker.stop()
        self.view.btnStart.setEnabled(True)




