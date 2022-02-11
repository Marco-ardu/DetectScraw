import multiprocessing as mp
import sys

from PyQt5.QtWidgets import QApplication
from loguru import logger

from Controller import MainController
from View import ViewWindow
from demo_utils import setLogPath

if __name__ == "__main__":
    setLogPath()
    logger.info('start')
    mp.freeze_support()
    app = QApplication(sys.argv)
    c = MainController(ViewWindow())
    c.start()
    sys.exit(app.exec())
