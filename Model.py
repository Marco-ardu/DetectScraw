from abc import ABC, abstractclassmethod
class Model:
    indexOrder = 0

    def getFileName(IdofFile:int):
        img_id = IdofFile % 2 + 1
        return f'test_img/{img_id}.jpg'


class WarnAlert(ABC):
    status = False

    @abstractclassmethod
    def RedAlert(self):
        pass

    @abstractclassmethod
    def YellowAlert(self):
        pass

    def Ease(self):
        pass

class PedestrianAlert(WarnAlert):

    def Alert(self):
        self.status = True
    def Ease(self):
        self.status = False

