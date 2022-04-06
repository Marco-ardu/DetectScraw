import multiprocessing as mp
import sys

from PyQt5.QtWidgets import QApplication
from loguru import logger

from Controller import MainController
from View import ViewWindow
from demo_utils import setPath, DeleteLogsTxt

if __name__ == "__main__":
    setPath()
    DeleteLogsTxt()
    logger.info('start app')
    mp.freeze_support()
    app = QApplication(sys.argv)
    c = MainController(ViewWindow())
    c.start()
    sys.exit(app.exec())
