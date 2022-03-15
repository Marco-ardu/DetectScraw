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
    warn_file = 'sound/ALL_NG.wav'


class LeftFailedAlert(WarnAlert):
    warn_message = '不合格产品'
    warn_file = 'sound/left_NG.wav'


class RightFailedAlert(WarnAlert):
    warn_message = '不合格产品'
    warn_file = 'sound/right_NG.wav'


class QualifiedAlert(WarnAlert):
    warn_message = '合格产品'
    warn_file = 'sound/ALL_PASS.wav'


class LeftReminderLocationAlert(WarnAlert):
    warn_message = '请检查左边位置是否摆放正确'
    warn_file = 'sound/left_location.wav'


class RightReminderLocationAlert(WarnAlert):
    warn_message = '请检查右边位置是否摆放正确'
    warn_file = 'sound/right_location.wav'


class ReminderLocationAlert(WarnAlert):
    warn_message = '请检查位置是否摆放正确'
    warn_file = 'sound/ALL_location.wav'
