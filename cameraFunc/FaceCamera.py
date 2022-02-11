import depthai as dai


def runFaceCamera(q, command):
    try:
        # Start defining a pipeline
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

        device_info = dai.Device.getAllAvailableDevices()[0]
        device = dai.Device(pipeline, device_info)
        print("Conected to " + device_info.getMxId())
        # Output queue will be used to get the rgb frames from the output defined above
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

        i = 0
        while command.value == 1:
            in_rgb = q_rgb.tryGet()
            if in_rgb is not None:
                frame = in_rgb.getCvFrame()
                q.put_nowait(frame)

            # i += 1
            # img_id = i % 2 + 1
            # frame = cv2.imread(f'test_img/{img_id}.jpg')
            # frame = cv2.resize(frame, (400,250))
            # q.put_nowait(frame)
            # time.sleep(0.1)
    except Exception as e:
        print(f'front error: {e}')
    finally:
        print('end front')
