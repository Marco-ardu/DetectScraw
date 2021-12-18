from NewView import ViewWindow
from NewWorker import Worker


class MainController:
    def __init__(self, view: ViewWindow) -> None:
        self.view = view

    def start(self):
        self.view.setup(self)
        self.view.show()
        self.view.prepareWorker(Worker)
        self.view.Worker.start()
