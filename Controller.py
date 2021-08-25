from View import ViewWindow
from Model import Model, PedestrianAlert, WarnAlert
from Worker import Worker

class MainController:
    def __init__(self, model: Model, view: ViewWindow) -> None:
        self.model = model
        self.view = view

    def start(self):
        self.view.setup(self)
        self.view.show()
        self.view.Worker = Worker()

    def btnStart_clicked(self):
        self.view.btnStart.setEnabled(False)
        self.view.Worker.start()
        self.view.Worker.FrontImage.connect(self.view.UpdateFrontSlot)
        self.view.Worker.RearImage.connect(self.view.UpdateRearSlot)
        self.view.Worker.DriverImage.connect(self.view.UpdateDriverSlot)
        self.view.Worker.Alert.connect(self.view.runAlert)

    def btnStop_clicked(self):
        self.view.Worker.stop()
        self.view.setDefaultView()
        self.view.btnStart.setEnabled(True)




