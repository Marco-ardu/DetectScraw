import sys
from PyQt5.QtWidgets import QApplication
from Controller import MainController
from View import ViewWindow
from Model import Model

if __name__ == "__main__":
    app = QApplication(sys.argv)
    c = MainController(Model(), ViewWindow())
    c.start()
    sys.exit(app.exec())