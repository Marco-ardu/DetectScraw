from model.AlertModel import PedestrianFrontAlert, PedestrianRearAlert, DriverAlert, NoPhoneAlert, NoHelmetAlert

AlertText_PedestrianFront = 'PedestrianFront'
AlertText_PedestrianRear = 'PedestrianRear'
AlertText_DriverFocus = 'DriverFocus'
AlertText_NoPhone = 'NoPhone'

def AlertFactory(AlertIndex):
    AlertDict = {
        AlertText_PedestrianFront : PedestrianFrontAlert(),
        AlertText_PedestrianRear : PedestrianRearAlert(),
        AlertText_DriverFocus : DriverAlert(),
        AlertText_NoPhone : NoPhoneAlert()
    }

    return AlertDict[AlertIndex]

AlertIndex_PedestrianFront = 0
AlertIndex_PedestrianRear = 1
AlertIndex_DriverFocus = 2
AlertIndex_NoPhone = 3
AlertIndex_NoHelmet = 4

AlertList = [
    PedestrianFrontAlert(),
    PedestrianRearAlert(),
    DriverAlert(),
    NoPhoneAlert(),
    NoHelmetAlert()
]