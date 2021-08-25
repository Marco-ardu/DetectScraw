from model.AlertModel import PedestrianFrontAlert, PedestrianRearAlert, DriverAlert

AlertText_PedestrianFront = 'PedestrianFront'
AlertText_PedestrianRear = 'PedestrianRear'
AlertText_DriverFocus = 'DriverFocus'

def AlertFactory(AlertIndex):
    AlertDict = {
        AlertText_PedestrianFront : PedestrianFrontAlert(),
        AlertText_PedestrianRear : PedestrianRearAlert(),
        AlertText_DriverFocus : DriverAlert()
    }    

    return AlertDict[AlertIndex]