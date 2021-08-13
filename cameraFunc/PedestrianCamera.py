import cv2
import depthai as dai
import time

def runRearCamera(q, command):
    try:
        pipeline = dai.Pipeline()

        # Define a source - color camera
        cam_rgb = pipeline.createColorCamera()
        cam_rgb.setPreviewSize(600, 600)
        cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)

        # Create output
        xout_rgb = pipeline.createXLinkOut()
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        found, device_info = dai.Device.getDeviceByMxId("14442C1051EF97CD00")
        if not found:
            raise RuntimeError("device not found")
        print("Conected to " + device_info.getMxId())
        device = dai.Device(pipeline, device_info)
        # Output queue will be used to get the rgb frames from the output defined above
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        i = 0
        while command.value == 1:
            in_rgb = q_rgb.tryGet()
            if in_rgb is not None:
                frame = in_rgb.getCvFrame()
                q.put_nowait(frame)
    except Exception as e:
        print(f'rear error {e}')
    finally:
        print('end rear')