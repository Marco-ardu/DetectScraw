from cameraFunc import YoloCamera, FatigueCam, PedestrianCamera, CombinedCamera, DetectScrawRightCamera, \
    DetectScrawLeftCamera

TextYoloCamera = 'YoloCamera'
TextPedestrianCamera = 'PedestrianCamera'
TextFatigueCamera = 'FatigueCamera'
TextCombinedCamera = 'CombinedCamera'
TextDetectScrawRightCamera = 'DetectScrawRightCamera'
TextDetectScrawLeftCamera = 'DetectScrawLeftCamera'


def CameraFactory(CameraIndex):
    CameraDict = {
        TextFatigueCamera: FatigueCam.runFatigueCam,
        TextYoloCamera: YoloCamera.runYoloCamera,
        TextPedestrianCamera: PedestrianCamera.runPedestrianCamera,
        TextCombinedCamera: CombinedCamera.runCamera,
        TextDetectScrawLeftCamera: DetectScrawLeftCamera.run_Scraw_left,
        TextDetectScrawRightCamera: DetectScrawRightCamera.run_Scraw_right,
    }

    return CameraDict[CameraIndex]
