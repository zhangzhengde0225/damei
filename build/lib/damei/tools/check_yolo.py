"""
检查YOLO数据集
"""
import os
from pathlib import Path
import numpy as np
import cv2
from ..functions import general
from ..controls.color import ColorControl


class CheckYOLO(object):
	def __init__(self, dp):
		self.dp = dp
		self.suffixes = ['.jpg', '.png', '.bmp']

	def __call__(self, *args, **kwargs):
		dp = self.dp
		print(dp)
		self.classes = self.get_classes()

		num_color = len(self.classes) if self.classes is not None else 1000
		self.colors = ColorControl(num=num_color, random_color=True).color
		self.check(trte='train')

	def get_classes(self):
		dp = self.dp
		if os.path.exists(f'{dp}/classes.txt'):
			with open(f'{dp}/classes.txt', 'r') as f:
				classes = f.readlines()
			classes = [x.replace('\n', '') for x in classes]
		else:
			classes = None
		return classes

	def check(self, trte):
		p = f'{self.dp}/images/{trte}'
		pl = f'{self.dp}/labels/{trte}'
		if os.path.exists(p):
			pass
		else:
			print(f'Directory {p} does not exists, redirect to {self.dp}')
			p = self.dp
			pl = self.dp

		imgs = [f'{p}/{x}' for x in os.listdir(p) if Path(x).suffix in self.suffixes]
		imgs = sorted(imgs)
		# print(imgs)
		for i, imgp in enumerate(imgs):
			stem = Path(imgp).stem
			labelp = f'{pl}/{Path(imgp).stem}.txt'
			img = cv2.imread(imgp)
			h, w, c = img.shape

			with open(labelp, 'r') as f:
				label = f.readlines()
			label = np.array([x.split() for x in label], dtype=np.float32)

			classes = label[:, 0]
			bboxes = label[:, 1::]
			bboxes = general.xywh2xyxy(bboxes)
			for j in range(len(label)):
				cls = classes[j]
				bbox = bboxes[j]
				bbox[0] *= w
				bbox[1] *= h
				bbox[2] *= w
				bbox[3] *= h

				label = f'{self.classes[int(cls)]}' if self.classes is not None else f'{cls}'
				color = None

				general.plot_one_box_trace_pose_status(
					bbox, img, label=label, color=color)

			print(f'stem: {stem} img: {img.shape} lb: {label}')
			# cr = np.any(label[:, 0] == 1)
			cr = True
			if cr:
				if np.max(img.shape) > 1920:
					h, w, c = img.shape
					img = cv2.resize(img, (int(0.5 * w), int(0.5 * h)))
				cv2.imshow('xx', img)
				if cv2.waitKey(0) == ord('q'):
					exit()