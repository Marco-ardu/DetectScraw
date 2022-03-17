#!/usr/bin/env python3
# coding=utf-8

import depthai as dai
import yaml


def getCameraMxid():
    return [i.getMxId() for i in dai.Device.getAllAvailableDevices()]


def isExist():
    with open('config.yml', 'r') as stream:
        args = yaml.load(stream, Loader=yaml.FullLoader)
    if args['left_camera_mxid'] is None or args['right_camera_mxid'] is None:
        return getCameraMxid()
    else:
        if args['left_camera_mxid'] not in getCameraMxid() or args['right_camera_mxid'] not in getCameraMxid():
            return getCameraMxid()
        else:
            return [args['left_camera_mxid'], args['right_camera_mxid']]

