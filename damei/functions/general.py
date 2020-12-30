"""
functions of dm
"""
import math
import torch
import numpy as np


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


def confusion2score(cm, round=5):
	"""
	input n*n confusion matrix, output: P R F1 score of each class and average ACC
	:param cm: confusion matrix, np.array, (n, n)
	:param round: default 5, decimal places
	:return: P, R, F1 and ACC
		P, R, F1: np.array (n,)
		acc: float32
	"""
	confusion = cm
	assert confusion.shape[0] == confusion.shape[1]
	nc = confusion.shape[0]
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
	if isinstance(round, int):
		P = np.round(P, round)
		R = np.round(R, round)
		F1 = np.round(F1, round)
		acc = np.round(acc, round)
	return P, R, F1, acc
