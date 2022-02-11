from abc import ABC


class WarnAlert(ABC):
    warn_message = 'abstract warning'
    warn_color = None

    @classmethod
    def yellowAlert(cls):
        cls.warn_color = 'yellow'

    @classmethod
    def redAlert(cls):
        cls.warn_color = 'red'


class FailedAlert(WarnAlert):
    warn_message = '不合格产品'
    warn_file = 'sound/不合格产品.mp3'


class QualifiedAlert(WarnAlert):
    warn_message = '合格产品'
    warn_file = 'sound/合格产品.mp3'


class ReminderLocationAlert(WarnAlert):
    warn_message = '请检查位置是否摆放正确'
    warn_file = 'sound/提醒位置.mp3'
