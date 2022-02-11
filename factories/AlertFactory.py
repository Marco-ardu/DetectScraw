from model.AlertModel import FailedAlert, QualifiedAlert, ReminderLocationAlert

AlertIndex_Failed = 0
AlertIndex_Qualified = 1
AlertIndex_ReminderLocation = 2

AlertList = [
    FailedAlert(),
    QualifiedAlert(),
    ReminderLocationAlert()
]
