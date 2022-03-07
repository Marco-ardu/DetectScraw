from cameraFunc import DetectScrawCamera


TextDetectScrawCamera = 'DetectScrawCamera'


def CameraFactory(CameraIndex):
    CameraDict = {
        TextDetectScrawCamera: DetectScrawCamera.run_Scraw_Camera,
    }

    return CameraDict[CameraIndex]
