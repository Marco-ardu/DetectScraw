
class Model:
    indexOrder = 0

    def getFileName(IdofFile:int):
        img_id = IdofFile % 2 + 1
        return f'test_img/{img_id}.jpg'

