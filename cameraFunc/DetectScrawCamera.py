import queue
import traceback
from pathlib import Path

import blobconverter
import depthai as dai
import numpy as np
import yaml
from depthai_sdk import toTensorResult
from loguru import logger

from demo_utils import getNNPath, setLogPath, multiclass_nms, demo_postprocess
from visualize import vis

CLASSES = [
    "screw",
    "no_screw",
]

parentDir = Path(__file__).parent
blobconverter.set_defaults(output_dir=parentDir / Path("models"))
size = (320, 320)
nn_path = "cameraFunc/models/yolox_nano_components_openvino_2021.4_6shave.blob"
rgb_resolutions = {
    800: (800, 1280),
    720: (720, 1280),
    400: (400, 640),
}

mono_res_opts = {
    400: dai.MonoCameraProperties.SensorResolution.THE_400_P,
    720: dai.MonoCameraProperties.SensorResolution.THE_720_P,
    800: dai.MonoCameraProperties.SensorResolution.THE_800_P,
}

with open('config.yml', 'r') as stream:
    args = yaml.load(stream, Loader=yaml.FullLoader)


def run_Scraw_Camera(frame_queue, command, alert, device_mxid, repeat_times, new_value, old_value, left_right, status,
                     barcode, result):
    setLogPath()
    frames_qualified = ({})
    max_count = args['max_count']
    Barcode = None
    logger.info('start {} camera'.format(left_right))
    show_frame = None
    pipeline = dai.Pipeline()  # type: dai.Pipeline
    resolution = rgb_resolutions[args['resolution']]

    # ColorCamera
    cam = pipeline.createColorCamera()  # type: dai.node.ColorCamera
    cam.setPreviewSize(resolution[::-1])
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)

    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_xout = pipeline.createXLinkOut()  # type: dai.node.XLinkOut
    cam_xout.setStreamName("cam_out")
    cam.preview.link(cam_xout.input)
    controlIn = pipeline.createXLinkIn()
    controlIn.setStreamName("control")
    controlIn.out.link(cam.inputControl)

    # NeuralNetwork
    yoloDet = pipeline.createNeuralNetwork()  # type: dai.node.NeuralNetwork
    yoloDet.setBlobPath(str(getNNPath(nn_path)))
    yoloDet.setNumInferenceThreads(2)
    yoloDet.setNumPoolFrames(3)

    manip = pipeline.createImageManip()
    manip.setNumFramesPool(3)
    manip.initialConfig.setResizeThumbnail(320, 320)
    manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
    cam.preview.link(manip.inputImage)
    manip.out.link(yoloDet.input)
    yolox_det_nn_xout = pipeline.createXLinkOut()  # type: dai.node.XLinkOut
    yolox_det_nn_xout.setStreamName("yolox_det_nn")
    yoloDet.out.link(yolox_det_nn_xout.input)
    try:
        found, device_info = dai.Device.getDeviceByMxId(device_mxid)
        device = dai.Device(pipeline, device_info, True)
        mxid = device.getMxId()
        usb_speed = device.getUsbSpeed()
        print("   >>> MXID:", mxid)
        names_list = {'SUPER': 'USB3.0', 'HIGH': 'USB2.0'}
        print("   >>> USB speed:", names_list.get(usb_speed.name))
        cam_out = device.getOutputQueue("cam_out", 1, True)
        yolox_det_nn = device.getOutputQueue("yolox_det_nn", 30, False)
        controlQueue = device.getInputQueue("control")
        ctrl = dai.CameraControl()
        ctrl.setManualFocus(new_value['lenPos_new'].value)
        ctrl.setManualExposure(new_value['exp_time_new'].value, new_value['sens_ios_new'].value)
        controlQueue.send(ctrl)
        flag = '1' if left_right == 'left' else '2'
        while command.value != 0:
            in_rgb = cam_out.get()
            if in_rgb is not None:
                frame = in_rgb.getCvFrame()
                show_frame = frame.copy()
                yolox_det_data = yolox_det_nn.get()
                res = toTensorResult(yolox_det_data).get("output")
                predictions = demo_postprocess(res, (320, 320), p6=False)[0]
                boxes = predictions[:, :4]
                scores = predictions[:, 4, None] * predictions[:, 5:]
                boxes_xyxy = np.ones_like(boxes)
                boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
                boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
                boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
                boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0

                input_shape = np.array([320, 320])
                min_r = (input_shape / frame.shape[:2]).min()
                offset = (np.array(frame.shape[:2]) * min_r - input_shape) / 2
                offset = np.ravel([offset, offset])
                boxes_xyxy = (boxes_xyxy + offset[::-1]) / min_r
                dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.2, score_thr=0.2)
                if command.value == eval(flag):
                    if status['auto_focus_status'].value == 2:
                        logger.info('{} Camera Autoexposure enable'.format(left_right))
                        ctrl = dai.CameraControl()
                        ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.CONTINUOUS_VIDEO)
                        controlQueue.send(ctrl)
                        status['auto_focus_status'].value = 1
                    if status['auto_exp_status'].value == 2:
                        logger.info('{} Camera Autofocus enable, continuous'.format(left_right))
                        ctrl = dai.CameraControl()
                        ctrl.setAutoExposureEnable()
                        controlQueue.send(ctrl)
                        status['auto_exp_status'].value = 1
                    if new_value['lenPos_new'].value != old_value['lenPos_old'].value \
                            and status['auto_focus_status'].value != 2:
                        logger.info('set {} Camera lenPos: {}'.format(left_right, new_value['lenPos_new'].value))
                        ctrl = dai.CameraControl()
                        ctrl.setManualFocus(new_value['lenPos_new'].value)
                        old_value['lenPos_old'].value = new_value['lenPos_new'].value
                        controlQueue.send(ctrl)
                    if new_value['exp_time_new'].value != old_value['exp_time_old'].value \
                            or new_value['sens_ios_new'].value != old_value['sens_ios_old'].value \
                            and status['auto_exp_status'].value != 2:
                        logger.info('set {} Camera exp_time: {}， sens_ios: {}'.format(left_right,
                                                                                      new_value['exp_time_new'].value,
                                                                                      new_value['sens_ios_new'].value))
                        ctrl = dai.CameraControl()
                        ctrl.setManualExposure(new_value['exp_time_new'].value, new_value['sens_ios_new'].value)
                        old_value['exp_time_old'].value = new_value['exp_time_new'].value
                        old_value['sens_ios_old'].value = new_value['sens_ios_new'].value
                        controlQueue.send(ctrl)
                try:
                    if dets is not None:
                        final_boxes = dets[:, :4]
                        final_scores, final_cls_inds = dets[:, 4], dets[:, 5]
                        no_screw = final_boxes[(final_cls_inds == 1) & (final_scores > 0.5)]
                        screw = final_boxes[(final_cls_inds == 0) & (final_scores > 0.5)]
                        show_frame = vis(
                            frame,
                            final_boxes,
                            final_scores,
                            final_cls_inds,
                            conf=args['confidence_threshold'],
                            class_names=CLASSES,
                        )
                        if not barcode.empty():
                            max_count = 20
                            Barcode = barcode.get_nowait()
                            print(Barcode)
                            frames_qualified.setdefault(Barcode, {}).setdefault(left_right, {"res": [],
                                                                                             "count": []})
                        if max_count != 0 and Barcode is not None:
                            frames_qualified[Barcode][left_right]['res'].append(len(no_screw) == 0 and len(screw) == 3)
                            frames_qualified[Barcode][left_right]['count'].append(len(no_screw) + len(screw))
                            max_count -= 1
                        elif max_count == 0 and Barcode is not None:
                            result.put_nowait(frames_qualified)
                            print(result.get_nowait())
                            Barcode = None
                    frame_queue.put_nowait(show_frame)
                except queue.Full:
                    pass
    except Exception as e:
        print(traceback.print_exc())
        if repeat_times != 10:
            repeat_times.value += 1
            run_Scraw_Camera(frame_queue, command, alert, device_mxid, repeat_times, new_value, old_value, left_right,
                             status,
                             barcode, result)
        logger.error(f"Device {device_mxid} not found!\n" + str(e))
