import argparse
import json
import queue
import sys
from pathlib import Path

import blobconverter
import depthai as dai
import numpy as np
import yaml
from depthai_sdk import toTensorResult
from loguru import logger

from demo_utils import getNNPath, setLogPath
from visualize import vis

CLASSES = [
    "screw",
    "no_screw",
]

parentDir = Path(__file__).parent
blobconverter.set_defaults(output_dir=parentDir / Path("../models"))
size = (320, 320)
nn_path = "models/yolox_nano_components_add_nms_openvino_2021.4_6shave.blob"
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

json_path = ''
if getattr(sys, 'frozen', False):
    dirname = Path(sys.executable).resolve().parent
    json_path = dirname / 'mxid.json'
elif __file__:
    json_path = Path("mxid.json")
with open(json_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

with open('config.yml', 'r') as stream:
    args = yaml.load(stream, Loader=yaml.FullLoader)


def getDictKey_1(myDict, value):
    return [k for k, v in myDict.items() if v == value]


def run_Scraw_left(frame_queue, command, alert, repeat_times):
    setLogPath()
    show_frame = None
    focus = config.get('focus')
    exp_time = config.get('exp_time')
    sens_iso = config.get('sens_iso')
    device_id = getDictKey_1(config.get('cam'), 'left')[0]
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
        found, device_info = dai.Device.getDeviceByMxId(device_id)  # type: bool, dai.DeviceInfo
        device = dai.Device(pipeline, device_info)

        mxid = device.getMxId()
        cameras = device.getConnectedCameras()
        usb_speed = device.getUsbSpeed()
        print("   >>> MXID:", mxid)
        names_list = {'SUPER': 'USB3.0', 'HIGH': 'USB2.0'}
        print("   >>> USB speed:", names_list.get(usb_speed.name))
        cam_out = device.getOutputQueue("cam_out", 1, True)
        yolox_det_nn = device.getOutputQueue("yolox_det_nn", 30, False)
        controlQueue = device.getInputQueue("control")
        ctrl = dai.CameraControl()
        ctrl.setManualFocus(focus.get('left'))
        controlQueue.send(ctrl)
        ctrl = dai.CameraControl()
        ctrl.setManualExposure(exp_time.get('left'), sens_iso.get('left'))
        controlQueue.send(ctrl)
        while command.value != 0:
            in_rgb = cam_out.get()
            if in_rgb is not None:
                frame = in_rgb.getCvFrame()
                show_frame = frame.copy()
                yolox_det_data = yolox_det_nn.get()
                res = toTensorResult(yolox_det_data)
                bboxes = res.get("bboxes")
                scores = res.get("scores")
                selected_indices = res.get("selected_indices.0")
                selected_indices = selected_indices[
                    (selected_indices >= 0).all(1) & (selected_indices < scores.shape).all(1)
                    ]
                class_indices = selected_indices[:, 1]
                box_indices = selected_indices[:, 2]
                bboxes = bboxes[:, box_indices]
                scores = [scores[i][j][k] for i, j, k in selected_indices]
                if bboxes is not None:
                    boxes_xyxy = bboxes.squeeze(0)
                    input_shape = np.array(size)
                    min_r = (input_shape / frame.shape[:2]).min()
                    offset = (np.array(frame.shape[:2]) * min_r - input_shape) / 2
                    offset = np.ravel([offset, offset])
                    final_boxes = (boxes_xyxy + offset[::-1]) / min_r
                    final_cls_inds = class_indices
                    final_scores = np.array(scores)
                    no_screw = final_boxes[(final_cls_inds == 1) & (final_scores > 0.5)]
                    screw = final_boxes[(final_cls_inds == 0) & (final_scores > 0.5)]
                    show_frame = vis(
                        frame,
                        final_boxes,
                        scores,
                        final_cls_inds,
                        conf=args['confidence_threshold'],
                        class_names=CLASSES,
                    )
                try:
                    frame_queue.put_nowait(show_frame)
                except queue.Full:
                    pass
    except Exception as e:
        if repeat_times != 10:
            repeat_times.value += 1
            run_Scraw_left(frame_queue, command, alert, repeat_times)
        logger.error(f"Device {device_id} not found!\n" + str(e))
