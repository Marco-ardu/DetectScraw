import depthai as dai
from loguru import logger


def getPipeline(name):
    # Create pipeline
    pipeline = dai.Pipeline()
    # Define sources and outputs
    camRgb = pipeline.create(dai.node.ColorCamera)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName(name)
    # Properties
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    camRgb.setIspScale(2, 3)
    # Linking
    camRgb.video.link(xoutRgb.input)
    controlIn = pipeline.create(dai.node.XLinkIn)
    controlIn.setStreamName("control")
    controlIn.out.link(camRgb.inputControl)
    return pipeline


def runCamera(frame_queue, command, device_mxid, repeat_times, lenPos, exp_time, sens_ios, old_value):
    # 管道已创建，现在将设备连接管道
    try:
        with dai.Device(getPipeline(device_mxid), dai.Device.getDeviceByMxId(device_mxid)[1], True) as device:
            device.startPipeline()
            q_rgb = device.getOutputQueue(name=device_mxid, maxSize=4, blocking=False)
            controlQueue = device.getInputQueue("control")
            ctrl = dai.CameraControl()
            ctrl.setManualFocus(lenPos.value)
            ctrl.setManualExposure(exp_time.value, sens_ios.value)
            controlQueue.send(ctrl)
            while command.value != 0:
                in_rgb = q_rgb.get()
                frame_queue.put_nowait(in_rgb.getCvFrame())
                if lenPos.value != old_value['lenPos_old'].value:
                    ctrl = dai.CameraControl()
                    ctrl.setManualFocus(lenPos.value)
                    old_value['lenPos_old'].value = lenPos.value
                    controlQueue.send(ctrl)
                if exp_time.value != old_value['exp_time_old'].value or sens_ios.value != old_value['sens_ios_old'].value:
                    ctrl = dai.CameraControl()
                    ctrl.setManualExposure(exp_time.value, sens_ios.value)
                    old_value['exp_time_old'].value = exp_time.value
                    old_value['sens_ios_old'].value = sens_ios.value
                    controlQueue.send(ctrl)
    except Exception as e:
        if repeat_times.value <= 10:
            repeat_times.value += 1
            runCamera(frame_queue, command, device_mxid, repeat_times, lenPos, exp_time, sens_ios, old_value)
        logger.error(f"Device {device_mxid} not found!\n" + str(e))
