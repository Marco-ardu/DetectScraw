from abc import ABC, abstractclassmethod
class Model:
    indexOrder = 0

    def getFileName(IdofFile:int):
        img_id = IdofFile % 2 + 1
        return f'test_img/{img_id}.jpg'


class WarnAlert(ABC):
    warn_message = 'abstract warning'
    warn_color = None

    @classmethod
    def yellowAlert(cls):
        cls.warn_color = 'yellow'

    @classmethod
    def redAlert(cls):
        cls.warn_color = 'red'    


class PedestrianFrontAlert(WarnAlert):
    warn_message = '注意前方行人'
    warn_file = 'sound/pedestrian_front.wav'

class PedestrianRearAlert(WarnAlert):
    warn_message = '注意後方行人'
    warn_file = 'sound/pedestrian_rear.wav'

class DriverAlert(WarnAlert):
    warn_message = '駕駛注意'
    warn_file = 'sound/driver_focus.wav'

