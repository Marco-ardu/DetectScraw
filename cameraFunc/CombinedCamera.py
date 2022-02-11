import multiprocessing as mp
import queue
from pathlib import Path

import cv2
import depthai as dai
import numpy as np
import yaml

from factories import AlertFactory

with open('config.yml', 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

VOC_CLASSES = ("helmet", "head")
nnPath = "cameraFunc/models/yolox_nano_0_1_0_openvino_2021.4_6shave.blob"
path_helmet_model = "cameraFunc/models/helmet_detection_yolox1_openvino_2021.4_6shave.blob"
parentDir = Path(__file__).parent
shaves = 6
size = (320, 320)
CLASSES = ["phone", "person", "head"]

__all__ = ["vis"]


def multiclass_nms_phone(boxes, scores, nms_thr, score_thr):
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


def demo_postprocess_phone(outputs, img_size, p6=False):
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
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        if class_names is not None:
            text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
            txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                img,
                (x0, y0 - int(1.5 * txt_size[1])),
                (x0 + txt_size[0] + 1, y0 + 1),
                # (x0 + y0 + 1),
                # (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
                txt_bk_color,
                -1
            )
            # cv2.putText(img, text, (x0, y0 - txt_size[1]), font, 0.4, txt_color, thickness=1)
            cv2.putText(img, text, (x0, y0), font, 0.4, txt_color, thickness=1)

    return img


_COLORS = np.array(
    [
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],

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


def multiclass_nms_helmet(boxes, scores, nms_thr, score_thr):
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


def demo_postprocess_helmet(outputs, img_size, p6=False):
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


def to_tensor_result(packet):
    data = {}
    for tensor in packet.getRaw().tensors:
        if tensor.dataType == dai.TensorInfo.DataType.INT:
            data[tensor.name] = np.array(packet.getLayerInt32(tensor.name)).reshape(
                tensor.dims  # [::-1]
            )
        elif tensor.dataType == dai.TensorInfo.DataType.FP16:
            data[tensor.name] = np.array(packet.getLayerFp16(tensor.name)).reshape(
                tensor.dims  # [::-1]
            )
        elif tensor.dataType == dai.TensorInfo.DataType.I8:
            data[tensor.name] = np.array(packet.getLayerUInt8(tensor.name)).reshape(
                tensor.dims  # [::-1]
            )
        else:
            print("Unsupported tensor layer type: {}".format(tensor.dataType))
    return data


def to_planar(arr: np.ndarray, input_size: tuple = None) -> np.ndarray:
    if input_size is None or tuple(arr.shape[:2]) == input_size:
        return arr.transpose((2, 0, 1))

    input_size = np.array(input_size)
    if len(arr.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(arr)
    r = min(input_size / img.shape[:2])
    resize_ = (np.array(img.shape[:2]) * r).astype(int)
    resized_img = cv2.resize(
        img,
        resize_[::-1],
        interpolation=cv2.INTER_LINEAR,
    )
    padding = (input_size - resize_) // 2
    padded_img[
    padding[0]: padding[0] + int(img.shape[0] * r),
    padding[1]: padding[1] + int(img.shape[1] * r),
    ] = resized_img
    image = padded_img.transpose(2, 0, 1)
    return image


def frame_norm(frame, bbox):
    return (
            np.clip(np.array(bbox), 0, 1)
            * np.array([*frame.shape[:2], *frame.shape[:2]])[::-1]
    ).astype(int)


def runCamera(frame_queue, command, alert, camera_id):
    pipeline = dai.Pipeline()

    print("Creating Color Camera...")
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(1280, 720)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setFps(10)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    cam_xout = pipeline.createXLinkOut()
    cam_xout.setStreamName("cam_out")
    cam.preview.link(cam_xout.input)

    # NeuralNetwork
    print(f"Creating {Path(nnPath).stem} Neural Network...")
    yoloDet_phone = pipeline.createNeuralNetwork()  # type: dai.node.NeuralNetwork
    yoloDet_phone.setBlobPath(str(nnPath))
    yoloDet_phone.setNumInferenceThreads(2)
    yolox_det_nn_xout_phone = pipeline.createXLinkOut()  # type: dai.node.XLinkOut
    yolox_det_nn_xout_phone.setStreamName("yolox_det_nn_phone")
    yoloDet_phone.out.link(yolox_det_nn_xout_phone.input)

    yoloDet_helmet = pipeline.createNeuralNetwork()
    yoloDet_helmet.setBlobPath(
        Path(path_helmet_model)
            .resolve()
            .absolute()
            .as_posix()
    )
    yolox_det_nn_xout_helmet = pipeline.createXLinkOut()
    yolox_det_nn_xout_helmet.setStreamName("yolox_det_nn_helmet")
    yoloDet_helmet.out.link(yolox_det_nn_xout_helmet.input)

    manip = pipeline.createImageManip()
    manip.initialConfig.setResizeThumbnail(320, 320, 114, 114, 114)
    manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
    cam.preview.link(manip.inputImage)
    manip.out.link(yoloDet_helmet.input)
    manip.out.link(yoloDet_phone.input)

    found, device_info = dai.Device.getDeviceByMxId(camera_id)
    if not found:
        raise RuntimeError("device not found")

    device = dai.Device(pipeline, device_info)
    print("Starting pipeline...")
    cam_out = device.getOutputQueue("cam_out", 1, True)
    yolox_det_nn_helmet = device.getOutputQueue("yolox_det_nn_helmet")
    yolox_det_nn_phone = device.getOutputQueue("yolox_det_nn_phone", 4, False)

    frame = None

    while command.value != 0:
        phone_exists = False
        helmet_count = 0
        people_count = 0

        yolox_det_data_helmet = yolox_det_nn_helmet.tryGet()
        frame = cam_out.get().getCvFrame()
        yolox_det_data_phone = yolox_det_nn_phone.tryGet()

        if yolox_det_data_phone is not None:
            res = toTensorResult(yolox_det_data_phone).get("output")
            predictions = demo_postprocess_phone(res, size, p6=False)[0]
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

            dets = multiclass_nms_phone(boxes_xyxy, scores, nms_thr=0.3, score_thr=0.3)

            if dets is not None:
                final_boxes = dets[:, :4]
                final_scores, final_cls_inds = dets[:, 4], dets[:, 5]
                for cls_ind in final_cls_inds:
                    object_name = CLASSES[int(cls_ind)]
                    if object_name == "phone":
                        phone_exists = True
                    elif object_name == "person":
                        people_count += 1

        if yolox_det_data_helmet is not None:
            res = to_tensor_result(yolox_det_data_helmet).get("output")
            predictions = demo_postprocess_helmet(res, (320, 320), p6=False)[0]

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

            dets = multiclass_nms_helmet(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.2)

            if dets is not None:
                final_boxes = dets[:, :4]
                final_scores, final_cls_inds = dets[:, 4], dets[:, 5]
                for cls_ind in final_cls_inds:
                    object_name = VOC_CLASSES[int(cls_ind)]
                    if object_name == "helmet": helmet_count += 1
            # print(f"phone: {phone_exists} people:{people_count} helm:{helmet_count}")
            if phone_exists:
                alert.value = AlertFactory.AlertIndex_NoPhone
            elif people_count > 1:
                alert.value = AlertFactory.AlertIndex_PedestrianRear
            elif helmet_count < 1 and people_count > 0:
                alert.value = AlertFactory.AlertIndex_NoHelmet
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass

    print("ended")


def main():
    frame_queue = mp.Queue(4)
    command = mp.Value('i', 1)
    alert = mp.Value('i', 99)
    camera_id = config["LEFT_CAMERA_ID"]
    print(camera_id)

    proccess = mp.Process(target=runCamera, args=(frame_queue, command, alert, camera_id,))
    proccess.start()

    while True:
        try:
            frame = frame_queue.get_nowait()
            cv2.imshow('frame', frame)
        except queue.Empty or queue.Full:
            pass

        if alert.value != 99:
            print(AlertFactory.AlertList[alert.value])

        if cv2.waitKey(1) == ord('q'):
            command.value = 0
            break

    proccess.kill()


if __name__ == '__main__':
    main()
