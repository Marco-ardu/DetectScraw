#!/usr/bin/env python3
# coding=utf-8
import json
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
from loguru import logger

from demo_utils import cv2AddChineseText

tips = [
    "请使用英文输入法输入",
    "输入 \"q\" 退出当前画面,",
    "或者确定当前的默认配置是正确;",
    "输入 \",\" 和 \".\" 进行调焦;",
    "输入 \"i\" 和 \"o\" 调节曝光时间",
    "输入 \"j\" 和 \"k\" 调节感光度",
    "输入 \"l\" 设置当前相机为左相机;",
    "输入 \"r\" 设置当前相机为右相机;",
    "输入 \"Esc\" 退出主程序。",
]

select_ids = set()


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


def test_cam(deviceId=None, name="test", focus=None, exp_time=None, sens_iso=None, repeat_times=0):
    if focus is None:
        focus = 156
    if exp_time is None:
        exp_time = 20000
    if sens_iso is None:
        sens_iso = 800

    lensPos = focus
    lensMin = 0
    lensMax = 255

    exp_time = exp_time
    exp_min = 1
    exp_max = 33000

    sens_iso = sens_iso
    sens_min = 100
    sens_max = 1600

    EXP_STEP = 500  # us
    ISO_STEP = 50
    LENS_STEP = 3
    device_infos = dai.Device.getAllAvailableDevices()  # type: Any
    if len(device_infos) == 0:
        if repeat_times != 100:
            repeat_times += 1
            test_cam(deviceId, name, focus, exp_time, sens_iso, repeat_times)
        logger.error("No DepthAI devices found!")
    elif len(device_infos) == 1:
        if repeat_times != 100:
            repeat_times += 1
            test_cam(deviceId, name, focus, exp_time, sens_iso, repeat_times)
        logger.error("Only one DepthAI device found!")
    else:
        for i, device_info in enumerate(device_infos):
            print(f"[{i}] {device_info.getMxId()} [{device_info.state.name}]")
        device_info = next(
            filter(lambda info: info.getMxId() == deviceId, device_infos), None
        )
        if device_info is None:
            while True:
                val = input("Which DepthAI Device you want to use: ")
                try:
                    device_info = device_infos[int(val)]
                    break
                except:
                    logger.error('input values in [0, 1]')

    device = dai.Device(getPipeline(name), device_info)
    # Add callback to the output queue "frames" for all newly arrived frames (color, left, right)
    cam_out = device.getOutputQueue(name=name, maxSize=4, blocking=False)
    controlQueue = device.getInputQueue("control")
    ctrl = dai.CameraControl()
    ctrl.setManualFocus(lensPos)
    controlQueue.send(ctrl)
    while True:
        frame = cam_out.get().getCvFrame()
        # This is a different thread and you could use it to
        # run image processing algorithms here
        for i, v in enumerate(tips):
            frame = cv2AddChineseText(
                frame,
                text=v,
                position=(10, (1 + i) * 20),
                textSize=20,
                textColor=(255, 0, 0),
            )
        cv2.imshow(name, frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            cv2.destroyAllWindows()
            select_ids.add(device_info.getMxId())
            device.close()
            return {device_info.getMxId(): name}, {name: focus}, {name: exp_time}, {name: sens_iso}
        elif key == ord("l"):
            cv2.destroyAllWindows()
            select_ids.add(device_info.getMxId())
            device.close()
            return {device_info.getMxId(): "left"}, {"left": lensPos}, {"left": exp_time}, {"left": sens_iso}
        elif key == ord("r"):
            cv2.destroyAllWindows()
            select_ids.add(device_info.getMxId())
            device.close()
            return {device_info.getMxId(): "right"}, {"right": lensPos}, {"right": exp_time}, {"right": sens_iso}
        elif key == 27:  # Esc
            raise SystemExit
        elif key in [ord(","), ord(".")]:
            if key == ord(","):
                lensPos -= LENS_STEP
            if key == ord("."):
                lensPos += LENS_STEP
            lensPos = int(np.clip(lensPos, lensMin, lensMax))
            logger.info("Setting manual focus, lens position: {}".format(lensPos))
            ctrl = dai.CameraControl()
            ctrl.setManualFocus(lensPos)
            controlQueue.send(ctrl)
        elif key in [ord('i'), ord('o'), ord('j'), ord('k')]:
            if key == ord('i'):
                exp_time -= EXP_STEP
            if key == ord('o'):
                exp_time += EXP_STEP
            if key == ord('j'):
                sens_iso -= ISO_STEP
            if key == ord('k'):
                sens_iso += ISO_STEP
            exp_time = int(np.clip(exp_time, exp_min, exp_max))
            sens_iso = int(np.clip(sens_iso, sens_min, sens_max))
            logger.info("Setting manual exposure, time: {} iso: {}".format(exp_time, sens_iso))
            ctrl = dai.CameraControl()
            ctrl.setManualExposure(exp_time, sens_iso)
            controlQueue.send(ctrl)


def set_cam(filename="mxid.json"):
    repeat_times = 0
    file = Path(filename)
    config_dic_w = {}
    cam_dic_w = {}
    focus_dic_w = {}
    exp_time_dic_w = {}
    sens_iso_dic_w = {}
    cam_dic_r = {}
    if file.exists():
        config = json.loads(file.read_text(encoding="utf8"))
        focus_dic_r = config.setdefault("focus", {})  # type: dict
        exp_time_dic_r = config.setdefault("exp_time", {})  # type: dict
        sens_iso_r = config.setdefault("sens_iso", {})  # type: dict
    else:
        focus_dic_r = {}
        exp_time_dic_r = {}
        sens_iso_r = {}
    focus_dic_r.setdefault("left")  # left_lens
    focus_dic_r.setdefault("right")  # right_lens
    exp_time_dic_r.setdefault("left")  # left_exp_time
    exp_time_dic_r.setdefault("right")  # right_exp_time
    sens_iso_r.setdefault("left")  # left_sens_iso
    sens_iso_r.setdefault("right")  # right_sens_iso

    while len(cam_dic_w) < 2:
        for j, i in enumerate(focus_dic_r.items()):
            import time
            time.sleep(1)
            temp_cam, temp_focus, temp_exp_time, temp_sens_iso = test_cam(cam_dic_r.get(i[0]), *i,
                                                                          exp_time_dic_r.get(i[0]),
                                                                          sens_iso_r.get(i[0]), repeat_times)
            cam_dic_w.update(temp_cam)
            focus_dic_w.update(temp_focus)
            exp_time_dic_w.update(temp_exp_time)
            sens_iso_dic_w.update(temp_sens_iso)
    config_dic_w.update(
        {"cam": cam_dic_w, "focus": focus_dic_w, "exp_time": exp_time_dic_w, "sens_iso": sens_iso_dic_w})
    file.write_text(json.dumps(config_dic_w, indent=2), encoding="utf8")
    return config_dic_w


if __name__ == "__main__":
    print(set_cam())
