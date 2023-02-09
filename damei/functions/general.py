"""
functions of dm
"""
import math
import random
import collections
from contextlib import contextmanager

# import cv2
import numpy as np

try:
    import cv2
    import torch
except Exception as e:
    pass


def letterbox(
        img, new_shape=640, color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, roi=None):
    """
    resize image to new_shape with multiple of 32
    :param img: np.array [w, h, 3]
    :param new_shape: int or tuple, 640, [640, 320]
    :param color: background color
    :param auto: auto pad
    :param scaleFill: stretch pad
    :param scaleup: scale up and scale down if Ture else only scale down
    :param roi: use roi or not
    :return: img, ratio, (dw, dh), recover
    """
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]  720, 1280

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better tests mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # print(dw, dh, ratio)
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    # print(dw, dh)
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    # print(new_unpad)

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # roi
    if roi is not None:
        x1, x2 = roi[0] / shape[1] * new_unpad[0], roi[2] / shape[1] * new_unpad[0]  # convert from pixel to percet
        y1, y2 = roi[1] / shape[0] * new_unpad[1], roi[3] / shape[0] * new_unpad[1]
        img = img[int(y1):int(y2), int(x1):int(x2)]
        rest_h = img.shape[0] % 32
        rest_w = img.shape[1] % 32
        dh = 0 if rest_h == 0 else (32 - rest_h) / 2
        dw = 0 if rest_w == 0 else (32 - rest_w) / 2
        recover = [new_shape[0], new_unpad[1], int(x1) - dw, int(y1) - dh]
    else:
        recover = None

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh), recover


def plot_one_box_trace_pose_status(
        x, img, color=None, label=None, line_thickness=None, focus=False, trace=None, status=None,
        keypoints=None, kp_score=None):
    """
    draw box in img, support bbox, trace, pose and status, and focus
    :param x: bbox in xyxy format
    :param img: raw_img
    :param color: color in (R, G, B)
    :param label: label for target detection, i.e. cls or tid
    :param line_thickness: rt
    :param focus: is fill the inner bbox area
    :param trace: trace
    :param status: target status
    :param keypoints: tuple, [num_kps, 2], keypoints in pixel, support num_kps: 17 26 136
    :param kp_score: tuple, [num_kps, 1], score of every keypoint
    :return:
    """
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    # print('plot', img.shape, c1, c2, color, tl)
    if focus:  # 把框內的部分也填入
        focus_area = np.zeros((c2[1] - c1[1], c2[0] - c1[0], 3), dtype=np.uint8)
        focus_area[:, :, 0::] = color
        img = imgAdd(focus_area, img, x=c1[1], y=c1[0], alpha=0.75)

    if trace is not None:  # 如果传入了轨迹，就绘制轨迹，传入的轨迹是nparray (N, 2) N>=2，后面的2的xy
        # print('trace', trace)
        for i in range(len(trace) - 1):
            pt1 = tuple(trace[i])
            # pt2 = tuple(trace[i] + 1)  # 之前写的用在AILabelImage上的，没有报错啊
            pt2 = tuple(trace[i + 1])
            cv2.arrowedLine(img, pt1, pt2, color, int(2 * tl), cv2.LINE_8, 0, 0.3)

    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        if status is not None:  # 绘制状态，字体稍微小一点
            status = [status] if isinstance(status, str) else status
            # print(status)
            assert len(status) <= 2  # 最多两个状态类型，也可以是1个
            # print('s')
            spt1 = (int(x[2]), int(x[1]))  # status point1
            if len(status) == 1:
                max_status = status[0]
            else:
                max_status = status[0] if len(status[0]) >= len(status[1]) else status[1]  # 选择pose和action中最长的那个来计算size

            s_size = cv2.getTextSize(max_status, 0, fontScale=tl / 4, thickness=tf)[0]
            spt2 = (spt1[0] + s_size[0], spt1[1] + len(status) * s_size[1] + 15)
            cv2.rectangle(img, spt1, spt2, color, -1, cv2.LINE_AA)

            cv2.putText(
                img, status[0], (spt1[0], spt1[1] + s_size[1] + 4), 0, tl / 4, [255, 255, 255], thickness=tf,
                lineType=cv2.LINE_AA)
            if len(status) == 2:
                cv2.putText(
                    img, status[1], (spt1[0], spt1[1] + 2 * s_size[1] + 10), 0, tl / 4, [255, 255, 255], thickness=tf,
                    lineType=cv2.LINE_AA)
    # print(status, t_size, s_size, spt1, spt2, tl)

    if keypoints is not None:
        kp_preds = np.array(keypoints)
        kp_scores = np.array(kp_score)
        kp_num = len(kp_preds)

        if kp_num == 17:
            kpformat = 'coco'
            if kpformat == 'coco':
                l_pair = [
                    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                    (17, 11), (17, 12),  # Body
                    (11, 13), (12, 14), (13, 15), (14, 16)
                ]
                p_color = [
                    (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
                    # Nose, LEye, REye, LEar, REar
                    (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255),
                    (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                    (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255),
                    (77, 255, 127), (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
                line_color = [
                    (0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                    (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                    (77, 222, 255), (255, 156, 127),
                    (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]
            else:
                raise NotImplementedError
        elif kp_num == 136:
            l_pair = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),  # Body
                (17, 18), (18, 19), (19, 11), (19, 12),
                (11, 13), (12, 14), (13, 15), (14, 16),
                (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),  # Foot
                (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36),
                (36, 37), (37, 38),  # Face
                (38, 39), (39, 40), (40, 41), (41, 42), (43, 44), (44, 45), (45, 46), (46, 47), (48, 49), (49, 50),
                (50, 51), (51, 52),  # Face
                (53, 54), (54, 55), (55, 56), (57, 58), (58, 59), (59, 60), (60, 61), (62, 63), (63, 64), (64, 65),
                (65, 66), (66, 67),  # Face
                (68, 69), (69, 70), (70, 71), (71, 72), (72, 73), (74, 75), (75, 76), (76, 77), (77, 78), (78, 79),
                (79, 80), (80, 81),  # Face
                (81, 82), (82, 83), (83, 84), (84, 85), (85, 86), (86, 87), (87, 88), (88, 89), (89, 90), (90, 91),
                (91, 92), (92, 93),  # Face
                (94, 95), (95, 96), (96, 97), (97, 98), (94, 99), (99, 100), (100, 101), (101, 102), (94, 103),
                (103, 104), (104, 105),  # LeftHand
                (105, 106), (94, 107), (107, 108), (108, 109), (109, 110), (94, 111), (111, 112), (112, 113),
                (113, 114),  # LeftHand
                (115, 116), (116, 117), (117, 118), (118, 119), (115, 120), (120, 121), (121, 122), (122, 123),
                (115, 124), (124, 125),  # RightHand
                (125, 126), (126, 127), (115, 128), (128, 129), (129, 130), (130, 131), (115, 132), (132, 133),
                (133, 134), (134, 135)  # RightHand
            ]
            p_color = [
                (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255),
                (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255),
                (77, 255, 127),  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
                (77, 255, 255), (0, 255, 255), (77, 204, 255),  # head, neck, shoulder
                (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
                (77, 255, 255)]  # foot

            line_color = [
                (0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77),
                (77, 255, 77),
                (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
                (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36),
                (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)]
        elif kp_num == 26:
            l_pair = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # Head
                (5, 18), (6, 18), (5, 7), (7, 9), (6, 8), (8, 10),  # Body
                (17, 18), (18, 19), (19, 11), (19, 12),
                (11, 13), (12, 14), (13, 15), (14, 16),
                (20, 24), (21, 25), (23, 25), (22, 24), (15, 24), (16, 25),  # Foot
            ]
            p_color = [
                (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),  # Nose, LEye, REye, LEar, REar
                (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255),
                (191, 255, 77),  # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255),
                (77, 255, 127),  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
                (77, 255, 255), (0, 255, 255), (77, 204, 255),  # head, neck, shoulder
                (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
                (77, 255, 255)]  # foot

            line_color = [
                (0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77),
                (77, 255, 77),
                (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
                (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36),
                (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)]
        else:
            raise NotImplementedError

        # draw keypoints
        vis_thres = 0.05 if kp_num == 136 else 0.4
        part_line = {}
        for n in range(kp_scores.shape[0]):
            if kp_scores[n] <= vis_thres:
                continue
            cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
            part_line[n] = (cor_x, cor_y)
            if n < len(p_color):
                cv2.circle(img, (cor_x, cor_y), 3, p_color[n], -1)
            else:
                cv2.circle(img, (cor_x, cor_y), 1, (255, 255, 255), 2)
        # Draw limbs
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                if i < len(line_color):
                    cv2.line(img, start_xy, end_xy, line_color[i], 2 * int(kp_scores[start_p] + kp_scores[end_p]) + 1)
                else:
                    cv2.line(img, start_xy, end_xy, (255, 255, 255), 1)
    return img


def imgAdd(small_img, big_image, x, y, alpha=0.5):
    """
    draw small image into big image
    :param small_img: small img
    :param big_image: big img
    :param x: left position for drawing in pixel
    :param y: top position for drawing in pixel
    :param alpha: transparency
    :return: big img draw with small img
    """
    row, col = small_img.shape[:2]
    if small_img.shape[0] > big_image.shape[0] or small_img.shape[1] > big_image.shape[1]:
        raise NameError(f'imgAdd, the size of small img bigger than big img.')
    roi = big_image[x:x + row, y:y + col, :]
    roi = cv2.addWeighted(small_img, alpha, roi, 1 - alpha, 0)
    big_image[x:x + row, y:y + col] = roi
    return big_image


def imgMask(img1, img2, mask):
    """
    mask为1的像素取img1, mask为0的像素取img2
    :param img1: image 1 ndarray [h, w, c]
    :param img2: image 2 ndarray [h, w, c]
    :param mask: mask ndarray [h, w]
    :return:
    """
    pass


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, return_np=False):
    """
    input box1 and box2, return the iou.
    :param box1: torch.tensor or numpy.array, shape: (4,) normlized
    :param box2: torch.tensor or numpy.array, shape: (4,) normlized
    :param x1y1x2y2: True, if True, treat box as x1y1x2y2 format else xcycwh format
    :param GIoU: False, if True, cal GIoU
    :param DIoU: False, if True, cal DIou
    :param CIoU: False, if True, cal CIou
    :param return_np: Fasle, if True, return numpy.array type, else torch.tensor type
    :return: the IoU or GIoU or DIou OR CIoU of box1 and box2
    """

    if isinstance(box1, np.ndarray):
        box1 = torch.from_numpy(box1)
    if isinstance(box2, np.ndarray):
        box2 = torch.from_numpy(box2)

    box2 = box2.T
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    # print(inter, union)
    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v + 1e-16)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
    if return_np:
        iou = iou.numpy()
    # print(box1, box2, box1.shape, box2.shape, iou)
    return iou


def bbox_contains(bbox1, bbox2):
    """计算bbox1是否包含bbox2"""
    b1_x1, b1_y1, b1_x2, b1_y2 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
    inter_w = (np.min(b1_x2, b2_x2) - np.max(b1_x1, b2_x1))
    inter_w = inter_w if inter_w > 0 else 0
    inter_h = (np.min(b1_y2, b2_y2) - np.max(b1_y1, b2_y1))
    inter_h = inter_h if inter_h > 0 else 0
    inter_area = inter_w * inter_h
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    return inter_area / (b2_area + 1e-16)


def confusion2score(cm, round=5, with_background=False, return_type='list', names=None, percent=False):
    """
    input n*n confusion matrix, output: P R F1 score of each class and average ACC
    :param cm: confusion matrix, np.array, (n, n)
    :param round: default 5, decimal places
    :param with_backgroud: default False, if True, indicate the last row is background class
    :return: P, R, F1 and ACC
        P, R, F1: np.array (n,)
        acc: float32
    """
    confusion = cm
    assert confusion.shape[0] == confusion.shape[1], 'confusion matrix must be square'
    nc = confusion.shape[0]  # number of classes
    actual_nc = nc - 1 if with_background else nc
    # sum of each row
    P_sum = np.repeat(np.sum(confusion, axis=1) + 1e-10, nc).reshape(nc, nc)
    P = confusion / P_sum
    R_sum = np.repeat(np.sum(confusion, axis=0) + 1e-10, nc).reshape(nc, nc).transpose()
    R = confusion / R_sum

    # 取对角
    P = np.diagonal(P, offset=0)
    R = np.diagonal(R, offset=0)
    F1 = 2 * P * R / (P + R + 1e-10)

    # acc
    correct = np.diagonal(confusion, offset=0)
    acc = np.sum(correct) / (np.sum(confusion) + 1e-10)
    # print(confusion, P, R, F1, acc)
    if percent:  # 变为百分比
        P = P * 100
        R = R * 100
        F1 = F1 * 100
        acc = acc * 100

    if isinstance(round, int):
        P = np.round(P, round)
        R = np.round(R, round)
        F1 = np.round(F1, round)
        acc = np.round(acc, round)

    if with_background:
        P = P[:-1]
        R = R[:-1]
        F1 = F1[:-1]

    mean_P = np.round(np.mean(P), round)
    mean_R = np.round(np.mean(R), round)
    mean_F1 = np.round(np.mean(F1), round)

    if return_type == 'list':
        if names is None:
            names = ['class_%d' % i for i in range(actual_nc)]
        assert len(names) == actual_nc, f'names must be same as number of classes {actual_nc} != {len(names)}'
        ret = list()
        head = [''] + names + ['mean']
        ret.append(head)
        ret.append(['P'] + list(P) + [mean_P])
        ret.append(['R'] + list(R) + [mean_R])
        ret.append(['F1'] + list(F1) + [mean_F1])
        ret.append(['ACC'] + [acc])
        return ret

    elif return_type == 'dict':
        ret = collections.OrderedDict()
        ret['classes'] = names if names else ['class_%d' % i for i in range(actual_nc)]
        ret['Precision'] = P
        ret['Recall'] = R
        ret['F1 score'] = F1
        ret['Accuracy'] = acc
        ret['Mean Precision'] = mean_P
        ret['Mean Recall'] = mean_R
        ret['Mean F1 score'] = mean_F1
        return ret
    else:
        raise ValueError('return_type must be list or dict')

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def xyxy2xywh(x):
    """
    既要支持多个bbox转，也要支持单个bbox转
    :param x: bbox in [x1, y1, x2, y2] format of torch.tensor or np.array.
    :return: bbox in [xc, yc, w, h] format.
    """
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    # x = np.array(x)
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    if len(x.shape) == 1:
        y[0] = (x[0] + x[2]) / 2
        y[1] = (x[1] + x[3]) / 2
        y[2] = (x[2] - x[0]) / 2
        y[3] = (x[3] - x[1]) / 2
    elif len(x.shape) == 2:
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
    else:
        raise NotImplementedError(f'does not support dim: {len(x.shape)} shape: {x.shape}')
    return y


def xywh2xyxy(x, need_scale=False, im0=None):
    """
    :param x: bbox in [xc, yc, w, h] format of torch.tensor or np.array.
    :return: bbox in [x1, y1, x2, y2] format
    """
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    if need_scale:
        assert im0 is not None
        h, w = im0.shape[:2]
        y[:, 0] = y[:, 0] * w
        y[:, 1] = y[:, 1] * h
        y[:, 2] = y[:, 2] * w
        y[:, 3] = y[:, 3] * h
        y = y.type(torch.IntTensor) if isinstance(
            y, torch.Tensor) else y.astype(np.int)
    return y


def pts2bbox(pts):
    """
    points to bbox
    :param pts: list or ndarray, [n, 2]
    :return: bbox in ndarray
    """
    pts = np.array(pts)
    assert pts.ndim == 2
    bbox = np.array(
        [np.min(pts, axis=0)[0], np.min(pts, axis=0)[1],
         np.max(pts, axis=0)[0], np.max(pts, axis=0)[1]])
    return bbox


def mask2bbox(mask):
    """
    :param mask: mask, [w, h] or [w, h, 1]
    :return: bbox
    """
    assert 2 <= mask.ndim <= 3
    if len(mask.shape) == 3:
        if mask.shape[2] != 1:
            raise NameError(f'mask shape error, {mask.shape}')
        mask = mask[:, :, 0]
    if mask.max() - mask.min() == 0:
        return None
    mask = np.array(mask / (mask.max() - mask.min()), dtype=np.uint8)  # 转成0到1
    # columns = np.argmax(mask, axis=0)
    pts = np.argwhere(mask == 1)  # 返回的是索引
    bbox = np.array([pts.min(axis=0)[1], pts.min(axis=0)[0], pts.max(axis=0)[1], pts.max(axis=0)[0]])
    # print(pts, pts.shape, bbox)
    return bbox


def non_max_suppression(bboxes, scores, threshold):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()
            keep.append(i)

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)

        iou = inter / (areas[i] + areas[order[1:]] - inter)
        idx = (iou <= threshold).nonzero().squeeze()
        if idx.numel() == 0:
            break
        order = order[idx + 1]
    return torch.LongTensor(keep)
