import sys
from PyQt5.QtWidgets import QApplication
from Controller import MainController
from View import ViewWindow
from Model import Model, PedestrianAlert

if __name__ == "__main__":
    app = QApplication(sys.argv)
    c = MainController(Model(), ViewWindow(), PedestrianAlert())
    c.start()
    sys.exit(app.exec())