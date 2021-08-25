from cameraFunc import YoloCamera, FatigueCam, PedestrianCamera

TextYoloCamera = 'YoloCamera'
TextPedestrianCamera = 'PedestrianCamera'
TextFatigueCamera = 'FatigueCamera'

def CameraFactory(CameraIndex):
    CameraDict = {
        TextFatigueCamera: FatigueCam.runFatigueCam,
        TextYoloCamera: YoloCamera.runYoloCamera,
        TextPedestrianCamera: PedestrianCamera.runPedestrianCamera
    }

    return CameraDict[CameraIndex]