from multiprocessing.spawn import freeze_support
import sys
from PyQt5.QtWidgets import QApplication
from Controller import MainController
from View import ViewWindow
from Model import Model, PedestrianAlert
import multiprocessing as mp

if __name__ == "__main__":
    mp.freeze_support()
    app = QApplication(sys.argv)
    c = MainController(Model(), ViewWindow())
    c.start()
    sys.exit(app.exec())