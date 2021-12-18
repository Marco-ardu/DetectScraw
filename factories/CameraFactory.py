from cameraFunc import YoloCamera, FatigueCam, PedestrianCamera, CombinedCamera

TextYoloCamera = 'YoloCamera'
TextPedestrianCamera = 'PedestrianCamera'
TextFatigueCamera = 'FatigueCamera'
TextCombinedCamera = 'CombinedCamera'

def CameraFactory(CameraIndex):
    CameraDict = {
        TextFatigueCamera: FatigueCam.runFatigueCam,
        TextYoloCamera: YoloCamera.runYoloCamera,
        TextPedestrianCamera: PedestrianCamera.runPedestrianCamera,
        TextCombinedCamera: CombinedCamera.runCamera
    }

    return CameraDict[CameraIndex]