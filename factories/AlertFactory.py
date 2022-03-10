from enum import IntEnum

from model.AlertModel import FailedAlert, QualifiedAlert, ReminderLocationAlert


class AlertEnum(IntEnum):
    AlertIndex_Failed = 0
    AlertIndex_Qualified = 1
    AlertIndex_ReminderLocation = 2


AlertDict = {
    AlertEnum.AlertIndex_Failed: FailedAlert(),
    AlertEnum.AlertIndex_Qualified: QualifiedAlert(),
    AlertEnum.AlertIndex_ReminderLocation: ReminderLocationAlert()
}
