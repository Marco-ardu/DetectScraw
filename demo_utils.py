from json.tool import main
import os
import platform
import subprocess
import sys
import os
import threading
from datetime import datetime
from pathlib import Path
import depthai as dai
import cv2
import numpy as np
import yaml
from PIL import ImageDraw, ImageFont, Image
from loguru import logger
import zipfile
import time

__all__ = ["mkdir", "nms", "multiclass_nms", "demo_postprocess",
           "play_sound", "getNNPath", "cv2AddChineseText",
           "setLogPath", "audio_remind", "put_text", 'save_yml', 
           'save_to_picture', 'getCameraMxid', 'isExist']


whether_dict = {True: 'PASS', False: 'NG'}

def compression_pictures():
    target_path = 'images'
    all_content = os.listdir(target_path)
    png_paths = []
    for i in all_content:
        if i.split('.')[-1] == 'png':
            png_paths.append('images/'+ i)
    logger.info('All png numbers is {}'.format(len(png_paths)))
    if len(png_paths) > 1000:
        time_res = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = 'images/{}_images.7z'.format(time_res)
        f = zipfile.ZipFile(os.path.abspath(zip_path),'w',zipfile.ZIP_DEFLATED)
        for i in png_paths:
            f.write(i)
        f.close()
        logger.info('compression path is {}'.format(zip_path))
        for i in png_paths:
            os.remove(i)


'''获取文件的大小,结果保留两位小数，单位为MB'''
def get_FileSize(filePath):
    fsize = os.path.getsize(filePath)
    fsize = fsize/float(1024*1024)
    return round(fsize,2)


def zip_compress(to_zip,save_zip_name):#save_zip_name是带目录的，也可以不带就是当前目录
#1.先判断输出save_zip_name的上级是否存在(判断绝对目录)，否则创建目录
    save_zip_dir=os.path.split(os.path.abspath(save_zip_name))[0]#save_zip_name的上级目录
    print(save_zip_dir)
    if not os.path.exists(save_zip_dir):
        os.makedirs(save_zip_dir)
        print('创建新目录%s'%save_zip_dir)
    f = zipfile.ZipFile(os.path.abspath(save_zip_name),'w',zipfile.ZIP_DEFLATED)
# 2.判断要被压缩的to_zip是否目录还是文件，是目录就遍历操作，是文件直接压缩
    if not os.path.isdir(os.path.abspath(to_zip)):#如果不是目录,那就是文件
        if os.path.exists(os.path.abspath(to_zip)):#判断文件是否存在
            f.write(to_zip)
            f.close()
            logger.info('%s压缩为%s' % (to_zip, save_zip_name))
        else:
            logger.info('%s文件不存在'%os.path.abspath(to_zip))
    else:
        if os.path.exists(os.path.abspath(to_zip)):#判断目录是否存在，遍历目录
            zipList = []
            for dir,subdirs,files in os.walk(to_zip):#遍历目录，加入列表
                for fileItem in files:
                    zipList.append(os.path.join(dir,fileItem))
                    # print('a',zipList[-1])
                for dirItem in subdirs:
                    zipList.append(os.path.join(dir,dirItem))
                    # print('b',zipList[-1])
            #读取列表压缩目录和文件
            for i in zipList:
                f.write(i,i.replace(to_zip,''))#replace是减少压缩文件的一层目录，即压缩文件不包括to_zip这个目录
                # print('%s压缩到%s'%(i,save_zip_name))
            f.close()
            logger.info('%s压缩为%s' % (to_zip, save_zip_name))
        else:
            logger.info('%s文件夹不存在' % os.path.abspath(to_zip))

def DeleteLogsTxt():
    if os.path.exists('logs.txt'):
        size = get_FileSize('logs.txt')
        if size > 100:
            logger.info('log文件大于100MB')
            zip_compress('logs.txt', 'logs.zip')
            os.remove('logs.txt')
            file=open("logs.txt","a")

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_to_picture(whether, image, numbering):
    time_res = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent_dir = Path("images")
    parent_dir.mkdir(exist_ok=True)
    name = parent_dir / f"{time_res}_{numbering}"
    exist_files = list(parent_dir.glob(f"*_{numbering}_No*.png"))
    if whether:
        put_text(image, "PASS", (1075, 400), (0, 255, 0), font_scale=5, thickness=4)
    else:
        put_text(image, "NG", (1175, 400), (0, 0, 255), font_scale=5, thickness=4)
    if exist_files:
        latest_file = max(exist_files, key=os.path.getctime).stem.split("No")[-1].split("_")[0]
        logger.info('save picture {} time'.format(int(latest_file) + 1))
        cv2.imwrite(f"{name}_No{int(latest_file) + 1:0>2d}_{whether_dict[whether]}.png", image)
    else:
        logger.info('save picture first time')
        cv2.imwrite(f"{name}_No01_{whether_dict[whether]}.png", image)

def getCameraMxid():
    return [i.getMxId() for i in dai.Device.getAllAvailableDevices()]


def isExist():
    with open('config.yml', 'r') as stream:
        args = yaml.load(stream, Loader=yaml.FullLoader)
    MXIDS = getCameraMxid()
    logger.info(MXIDS)
    if args['left_camera_mxid'] is None or args['right_camera_mxid'] is None:
        return MXIDS
    else:
        if args['left_camera_mxid'] not in MXIDS or args['right_camera_mxid'] not in MXIDS:
            return MXIDS
        else:
            logger.info('get args ')
            return [args['left_camera_mxid'], args['right_camera_mxid']]



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
    # print(outputs,type(outputs))
    # print(outputs.shape)
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
        if config_camera[3]:
            config['{}_camera_mxid'.format(direction)] = config_camera[5][0]
        else:
            config['{}_camera_mxid'.format(direction)] = config_camera[5][1]
        config['{}_camera_lensPos'.format(direction)] = config_camera[0]
        config['{}_camera_exp_time'.format(direction)] = config_camera[1]
        config['{}_camera_sens_ios'.format(direction)] = config_camera[2]
        config = yaml.dump(config)
        stream.write(config)
        logger.info('save {} camera parameters'.format(direction))


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
