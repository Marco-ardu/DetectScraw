import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage
import time
def runRearCamera(q, command):
    i = 0
    while command.value == 1:
        i += 1
        img_id = ( i + 1 ) % 2 + 1
        frame = cv2.imread(f'test_img/{img_id}.jpg')
        frame = cv2.resize(frame, (400,250))
        #rear_child_conn.send(frame)
        q.put_nowait(frame)
        time.sleep(0.1)

    print('end rear')