from enum import IntEnum

from model.AlertModel import FailedAlert, QualifiedAlert, ReminderLocationAlert, LeftFailedAlert, RightFailedAlert, \
    LeftReminderLocationAlert, RightReminderLocationAlert, ReminderCameraLoadAlert


class AlertEnum(IntEnum):
    NoAlert = 99
    AlertIndex_Failed = 0
    AlertIndex_LeftFailed = 1
    AlertIndex_RightFailed = 2
    AlertIndex_Qualified = 3
    AlertIndex_ReminderLocation = 4
    AlertIndex_LeftReminderLocation = 5
    AlertIndex_RightReminderLocation = 6
    AlertIndex_ReminderCameraLoad = 7

AlertDict = {
    AlertEnum.AlertIndex_Failed: FailedAlert(),
    AlertEnum.AlertIndex_LeftFailed: LeftFailedAlert(),
    AlertEnum.AlertIndex_RightFailed: RightFailedAlert(),
    AlertEnum.AlertIndex_Qualified: QualifiedAlert(),
    AlertEnum.AlertIndex_ReminderLocation: ReminderLocationAlert(),
    AlertEnum.AlertIndex_LeftReminderLocation: LeftReminderLocationAlert(),
    AlertEnum.AlertIndex_RightReminderLocation: RightReminderLocationAlert(),
    AlertEnum.AlertIndex_ReminderCameraLoad: ReminderCameraLoadAlert()
}
