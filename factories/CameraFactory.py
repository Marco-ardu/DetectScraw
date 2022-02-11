from cameraFunc import DetectScrawRightCamera, DetectScrawLeftCamera

TextDetectScrawRightCamera = 'DetectScrawRightCamera'
TextDetectScrawLeftCamera = 'DetectScrawLeftCamera'


def CameraFactory(CameraIndex):
    CameraDict = {
        TextDetectScrawLeftCamera: DetectScrawLeftCamera.run_Scraw_left,
        TextDetectScrawRightCamera: DetectScrawRightCamera.run_Scraw_right,
    }

    return CameraDict[CameraIndex]
