from cameraFunc import DetectScrawRightCamera, DetectScrawLeftCamera, SettingCamera

TextDetectScrawRightCamera = 'DetectScrawRightCamera'
TextDetectScrawLeftCamera = 'DetectScrawLeftCamera'
TextCamera = 'SettingCamera'


def CameraFactory(CameraIndex):
    CameraDict = {
        TextDetectScrawLeftCamera: DetectScrawLeftCamera.run_Scraw_left,
        TextDetectScrawRightCamera: DetectScrawRightCamera.run_Scraw_right,
        TextCamera: SettingCamera.runCamera,
    }

    return CameraDict[CameraIndex]
