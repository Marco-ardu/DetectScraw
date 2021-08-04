from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage

import cv2
import time

class Worker(QThread):
    ImageUpdate = pyqtSignal(QImage)

    def run(self):
        self.ThreadActive = True
        i = 0
        while self.ThreadActive:
            i += 1
            img_id = i % 2 + 1
            frame = cv2.imread(f'test_img/{img_id}.jpg')
            Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #Image = cv2.flip(Image, 1)
            ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)
            Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
            self.ImageUpdate.emit(Pic)
            time.sleep(0.1)

    def stop(self):
        self.ThreadActive = False
        self.quit()

    @staticmethod
    def cvFrametoQtImage(frame):
        pass