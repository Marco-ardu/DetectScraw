#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import os
import platform
import subprocess
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import ImageDraw, ImageFont, Image

__all__ = ["mkdir", "nms", "multiclass_nms", "demo_postprocess",
           "play_sound", "getNNPath", "cv2AddChineseText",
           "setLogPath", "audio_remind", "put_text", 'save_yml']

from loguru import logger


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate([valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def demo_postprocess(outputs, img_size, p6=False):
    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(hsize), np.arange(wsize))
        grid = np.stack((xv, yv), 2).reshape((1, -1, 2))
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


def save_yml(config_camera):
    with open('config.yml', 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    with open('config.yml', 'w') as stream:
        direction = 'left' if config_camera[3] else 'right'
        config['{}_camera_mxid'.format(direction)] = config_camera[5]
        config['{}_camera_lensPos'.format(direction)] = config_camera[0]
        config['{}_camera_exp_time'.format(direction)] = config_camera[1]
        config['{}_camera_sens_ios'.format(direction)] = config_camera[2]
        config = yaml.dump(config)
        stream.write(config)


def play_sound(path):
    if platform.system() == 'Windows':
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME)
    else:
        p = subprocess.Popen(
            "ffplay -nodisp -autoexit -hide_banner {}".format(path), shell=True
        )
        p.communicate()


def audio_remind(path):
    sound_thread_ = threading.Thread(target=play_sound, args=[path])
    sound_thread_.daemon = True
    sound_thread_.start()


def getNNPath(path):
    nnPath = path
    if getattr(sys, 'frozen', False):
        dirname = os.path.dirname(os.path.abspath(sys.executable))
        nnPath = os.path.join(dirname, nnPath)
    elif __file__:
        nnPath = os.path.join("./", nnPath)
    return nnPath


def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
    draw.text(position, text, textColor, font=fontStyle)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def setLogPath():
    logPath = ''
    if getattr(sys, 'frozen', False):
        dirname = Path(sys.executable).resolve().parent
        logPath = dirname / 'logs.txt'
    elif __file__:
        logPath = Path("./logs.txt")
    logger.add(logPath.as_posix())


def put_text(img, text, org, color=(255, 255, 255), bg=(0, 0, 0), font_scale=0.5, thickness=1):
    cv2.putText(img=img,
                text=text,
                org=org,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=bg,
                thickness=thickness + 3,
                lineType=cv2.LINE_AA,
                )
    cv2.putText(img=img,
                text=text,
                org=org,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
                )
