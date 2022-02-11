import multiprocessing as mp
import queue
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
import yaml

with open('../config.yml', 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

parentDir = Path(__file__).parent
shaves = 6
size = (320, 320)
CLASSES = ["phone", "person", "head"]

nnPath = "models/yolox_nano_0_1_0_openvino_2021.4_6shave.blob"


def toTensorResult(packet):
    """
    Converts NN packet to dict, with each key being output tensor name and each value being correctly reshaped and converted results array

    Useful as a first step of processing NN results for custom neural networks

    Args:
        packet (depthai.NNData): Packet returned from NN node

    Returns:
        dict: Dict containing prepared output tensors
    """
    data = {}
    for tensor in packet.getRaw().tensors:
        if tensor.dataType == dai.TensorInfo.DataType.INT:
            data[tensor.name] = np.array(packet.getLayerInt32(tensor.name)).reshape(tensor.dims)
        elif tensor.dataType == dai.TensorInfo.DataType.FP16:
            data[tensor.name] = np.array(packet.getLayerFp16(tensor.name)).reshape(tensor.dims)
        elif tensor.dataType == dai.TensorInfo.DataType.I8:
            data[tensor.name] = np.array(packet.getLayerUInt8(tensor.name)).reshape(tensor.dims)
        else:
            print("Unsupported tensor layer type: {}".format(tensor.dataType))
    return data


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = round(box[0])
        y0 = round(box[1])
        x1 = round(box[2])
        y1 = round(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)


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


def runCam(frame_queue, command, camera_id):
    pipeline = dai.Pipeline()

    print("Creating Color Camera...")
    cam = pipeline.createColorCamera()  # type: dai.node.ColorCamera
    cam.setPreviewSize(1280, 720)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setFps(60)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_xout = pipeline.createXLinkOut()
    cam_xout.setStreamName("cam_out")
    cam.preview.link(cam_xout.input)

    # NeuralNetwork
    print(f"Creating {Path(nnPath).stem} Neural Network...")
    yoloDet = pipeline.createNeuralNetwork()  # type: dai.node.NeuralNetwork
    yoloDet.setBlobPath(str(nnPath))
    yoloDet.setNumInferenceThreads(2)
    yolox_det_nn_xout = pipeline.createXLinkOut()  # type: dai.node.XLinkOut
    yolox_det_nn_xout.setStreamName("yolox_det_nn")
    yoloDet.out.link(yolox_det_nn_xout.input)

    manip = pipeline.createImageManip()
    manip.initialConfig.setResizeThumbnail(320, 320, 114, 114, 114)
    manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
    cam.preview.link(manip.inputImage)
    manip.out.link(yoloDet.input)
    # cam.preview.link(yoloDet.input)

    found, device_info = dai.Device.getDeviceByMxId(camera_id)
    if not found:
        raise RuntimeError("device not found")

    device = dai.Device(pipeline, device_info)

    print("Starting pipeline...")
    cam_out = device.getOutputQueue("cam_out", 1, True)
    yolox_det_nn = device.getOutputQueue("yolox_det_nn", 4, False)

    while True:
        frame = cam_out.get().getCvFrame()

        frame_debug = frame.copy()

        yolox_det_data = yolox_det_nn.tryGet()
        if yolox_det_data is not None:
            res = toTensorResult(yolox_det_data).get("output")
            predictions = demo_postprocess(res, size, p6=False)[0]
            # predictions = res[0]
            boxes = predictions[:, :4]
            scores = predictions[:, 4, None] * predictions[:, 5:]

            boxes_xyxy = np.ones_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
            boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0

            input_shape = np.array(size)
            min_r = (input_shape / frame.shape[:2]).min()
            offset = (np.array(frame.shape[:2]) * min_r - input_shape) / 2
            offset = np.ravel([offset, offset])
            boxes_xyxy = (boxes_xyxy + offset[::-1]) / min_r

            dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.3, score_thr=0.3)

            if dets is not None:
                final_boxes = dets[:, :4]
                final_scores, final_cls_inds = dets[:, 4], dets[:, 5]
                frame_debug = vis(
                    frame_debug,
                    final_boxes,
                    final_scores,
                    final_cls_inds,
                    conf=0.3,
                    class_names=CLASSES,
                )

            frame_queue.put_nowait(frame_debug)


def main():
    frame_queue = mp.Queue(4)
    command = mp.Value('i', 1)
    alert = mp.Value('i', 0)
    camera_id = config["LEFT_CAMERA_ID"]
    print(camera_id)

    proccess = mp.Process(target=runCam, args=(frame_queue, command, camera_id,))
    proccess.start()

    while True:
        try:
            frame = frame_queue.get_nowait()
            cv2.imshow('frame', frame)
        except queue.Empty or queue.Full:
            pass

        if cv2.waitKey(1) == ord('q'):
            command.value = 0
            break

    proccess.kill()


if __name__ == '__main__':
    main()
