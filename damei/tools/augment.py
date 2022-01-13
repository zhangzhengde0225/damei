"""
Augment
"""
import os, sys
import numpy as np
import cv2
from copy import deepcopy

from ..functions.general import mask2bbox


class Augment(object):
	def __init__(self):
		"""
		注意：所有使用的mask, 形状是[h, w, 1]，宽高跟原始图相同，通道为1，取值范围二值化，即0和1
		"""
		pass

	def get_random_tzr_params(self, mask, existing_masks=[], thresh=0):
		"""
		传入二值化mask，输出随机尺寸、位置、缩放矩阵，如果同时传入存在的masks，保证遮挡率小于阈值
		:param mask: binary mask, ndarray [h, w, 1]
		:param existing_masks:
		:param thresh:
		:return:
		"""
		assert np.max(mask) == 1 and np.min(mask) == 0, f'mask must be binary, max: {np.max(mask)} min: {np.min(mask)}'
		mask = np.array(mask, dtype=np.uint8)
		h, w, c = mask.shape
		while True:
			# 随机尺寸
			size = np.clip(np.random.randn() / 6 * 2 + 1.3, 0.5, 2)
			angle = np.random.randint(0, 180)
			# 获取随机缩放和随机旋转矩阵
			matRotateZoom = self.get_rotation_zoom_matrix(mask, scale_factor=size, angle=angle)
			# 执行随机缩放和旋转
			new_mask = cv2.warpAffine(deepcopy(mask), matRotateZoom, (h, w))
			new_mask = new_mask[..., np.newaxis]
			# 随机位置
			new_mask, img, translation_params = self.random_position(new_mask, img)
			if new_mask is None:
				continue
			# b = [self.mask2bbox(new_mask)]
			# self.imshow(new_mask, bboxes=b, clss=[cls], show_name='new_mask')
			# self.imshow(img, bboxes=b, clss=[cls], show_name='img')
			# 判断交叉
			occ = [0]
			for j in range(len(existing_masks)):
				exist_mask = existing_masks[j]
				iop2 = self.cal_contains(mask1=new_mask, mask2=exist_mask)
				# intersection of percent 2, 新生成的mask遮挡已存在的mask的程度
				occ.append(iop2)
			if np.max(occ) <= thresh:
				# print(f'完全无遮挡: {occ}')
				return img, new_mask, matRotateZoom, translation_params

	def apply_transformation(self, ):
		# img = cv2.warpAffine(deepcopy(im0), matRotateZoom, (h, w))
		pass

	def translate_mask(self, mask, translation_params):
		x1, y1, x2, y2, bw, bh, stick_x, stick_y = translation_params
		new_mask = np.zeros_like(mask, dtype=np.uint8)  # [h, w, 1]
		cropped_bbox = mask[y1:y2, x1:x2, :]
		new_mask[stick_y:stick_y + bh, stick_x:stick_x + bw, :] = cropped_bbox
		return new_mask

	def get_rotation_zoom_matrix(self, mask, scale_factor=1, angle=0):
		"""
		根据给定的尺寸因子、旋转，返回旋转缩放矩阵
		:param mask: [h, w, c] c=1
		:param scale_factor: scale factor, default 1
		:param angle: rotation angle, default 0, rotation center: bbox center
		:return RotationMatrix2D
		"""
		sf = scale_factor
		bbox = mask2bbox(mask)
		xc, yc = bbox[1] + (bbox[3] - bbox[1]) / 2, bbox[0] + (bbox[2] - bbox[0]) / 2
		matRotateZoom = cv2.getRotationMatrix2D((xc, yc), angle, sf)
		return matRotateZoom

	def cal_contains(self, mask1, mask2):
		"""计算mask1和mask2中的目标的包含关系，即mask1与mask2的交集 除以 mask2的面积"""
		intersection = np.sum(mask2[mask1 == 1] == 1)  # 交集
		area2 = np.sum(mask2 == 1)
		correlation = intersection / (area2 + 1e-8)
		return correlation

	def random_position(self, mask, img):
		"""

		:param mask:
		:param img:
		:return:
		"""
		bbox = mask2bbox(mask)
		if bbox is None:
			return None, None, None
		h, w, c = mask.shape
		x1, y1, x2, y2 = bbox
		bw, bh = x2 - x1, y2 - y1  # bbox w h
		stick_x = np.random.randint(low=0, high=w - bw)
		stick_y = np.random.randint(low=0, high=h - bh)
		# translate mask
		translation_params = [x1, y1, x2, y2, bw, bh, stick_x, stick_y]
		new_mask = self.translate_mask(mask, translation_params)
		# translate img
		new_img = np.zeros((h, w, 3), dtype=np.uint8)
		new_img[stick_y:stick_y + bh, stick_x:stick_x + bw, :] = img[y1:y2, x1:x2, :]
		# print(f'maskshape: {mask.shape} sx: {stick_x} sy: {stick_y} bbox: {bbox} bw: {bw} bh: {bh}\n', cropped_bbox.shape, new_mask.shape)
		return new_mask, new_img, translation_params
