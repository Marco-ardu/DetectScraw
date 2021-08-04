import sys

from PyQt5.QtWidgets import (
    QApplication, QDialog, QMainWindow, QMessageBox
)

from Controller import Controller
from View import ViewWindow
from Model import Model

if __name__ == "__main__":
    app = QApplication(sys.argv)
    c = Controller(Model(), ViewWindow())
    c.start()
    sys.exit(app.exec())