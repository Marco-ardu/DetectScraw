import multiprocessing as mp
import queue

import cv2


def runCam(frame_queue, command, alert):
    pass


def main():
    frame_queue = mp.Queue(4)
    command = mp.Value('i', 1)
    alert = mp.Value('i', 0)

    proccess = mp.Process(target=runCam, args=(frame_queue, command, alert,))
    proccess.start()

    while True:
        try:
            frame = frame_queue.get_nowait()
            cv2.imshow('frame', frame)
        except queue.Empty or queue.Full:
            pass

        if cv2.waitKey(1) == ord('q'):
            command.value = 0
            break

    proccess.kill()


if __name__ == '__main__':
    main()
