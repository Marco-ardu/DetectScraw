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


def runCamera(frame_queue, command, device_mxid, repeat_times, new_value, old_value, left_right, status):
    # 管道已创建，现在将设备连接管道
    try:
        with dai.Device(getPipeline(device_mxid), dai.Device.getDeviceByMxId(device_mxid)[1], True) as device:
            device.startPipeline()
            q_rgb = device.getOutputQueue(name=device_mxid, maxSize=4, blocking=False)
            controlQueue = device.getInputQueue("control")
            ctrl = dai.CameraControl()
            ctrl.setManualFocus(new_value['lenPos_new'].value)
            ctrl.setManualExposure(new_value['exp_time_new'].value, new_value['sens_ios_new'].value)
            controlQueue.send(ctrl)
            flag = '1' if left_right == 'left' else '2'
            while command.value != 0:
                in_rgb = q_rgb.get()
                frame_queue.put_nowait(in_rgb.getCvFrame())
                if status.value == eval(flag):
                    if new_value['lenPos_new'].value != old_value['lenPos_old'].value:
                        ctrl = dai.CameraControl()
                        ctrl.setManualFocus(new_value['lenPos_new'].value)
                        old_value['lenPos_old'].value = new_value['lenPos_new'].value
                        controlQueue.send(ctrl)
                    if new_value['exp_time_new'].value != old_value['exp_time_old'].value or new_value['sens_ios_new'].value != old_value['sens_ios_old'].value:
                        ctrl = dai.CameraControl()
                        ctrl.setManualExposure(new_value['exp_time_new'].value, new_value['sens_ios_new'].value)
                        old_value['exp_time_old'].value = new_value['exp_time_new'].value
                        old_value['sens_ios_old'].value = new_value['sens_ios_new'].value
                        controlQueue.send(ctrl)
                else:
                    in_rgb = q_rgb.get()
                    frame_queue.put_nowait(in_rgb.getCvFrame())
    except Exception as e:
        if repeat_times.value <= 10:
            repeat_times.value += 1
            runCamera(frame_queue, command, device_mxid, repeat_times, new_value, old_value, left_right, status)
        logger.error(f"Device {device_mxid} not found!\n" + str(e))
