"""
转换训练集，YOLOv5 format to COCO format
"""
import os, sys
import argparse
from pathlib import Path
import json
import shutil
import cv2
from tqdm import tqdm


class YOLO2COCO(object):
	def __init__(self, sp, tp=None):
		self.sp = sp  # source path
		self.tp = tp if tp is not None else f'{Path(sp).parent}/transformed_coco_format'

		# 检查输出文件夹
		if os.path.exists(self.tp):
			ipt = input(f'Target path: {self.tp} exists, \nremove and continue? [YES/no]: ')
			if ipt in ['Y', '', 'y', 'yes', 'YES']:
				shutil.rmtree(self.tp)
			else:
				print('Exit')
				exit()
		os.makedirs(self.tp)
		os.makedirs(f'{self.tp}/annotations')
		os.makedirs(f'{self.tp}/train2017')
		os.makedirs(f'{self.tp}/val2017')

		# 关于coco格式
		self.annotation_id = 1
		self.type = 'instances'
		self.categories = self.read_categories()
		self.info = {
			'year': 2021,
			'version': '1.0',
			'description': 'For object detection',
			'date_created': '2021-9-28',
		}
		self.licenses = [{
			'id': 1,
			'name': 'GNU General Public License v3.0',
			'url': '',
		}]

		# 支持的图像格式
		self.suffix = ['.jpg', '.bmp', '.png']

	def read_categories(self):
		file = f'{self.sp}/classes.txt'
		print(self.sp)
		assert os.path.exists(file), f'classes file {file} does not exists.'
		with open(file, 'r') as f:
			data = f.readlines()
		data = [x.replace('\n', '') for x in data]
		categories = []
		for i, cls_name in enumerate(data):
			cls = i + 1
			categories.append({
				'supercategory': cls_name,
				'id': cls,
				'name': cls_name
			})
		return categories

	def __call__(self, *args, **kwargs):
		sp = self.sp
		for trte in ['train', 'test']:
            print(f'Deal with {trte}')
            imgs = os.listdir(f'{sp}/images/{trte}')
            imgs = [f'{sp}/images/{trte}/{x}' for x in imgs if str(Path(x).suffix) in self.suffix]

            trval = trte if trte == 'train' else 'val'
            self.deal_single(img_paths=imgs, trval=trval)

	def deal_single(self, img_paths, trval):
		tp = self.tp
		bar = tqdm(img_paths)

		images = []
		annotations = []
		for i, img_path in enumerate(bar):
			img_id = i + 1
			stem = Path(img_path).stem
			txt_path = f'{str(Path(img_path).parent).replace("images", "labels")}/{stem}.txt'

			img = cv2.imread(img_path)
			h, w, c = img.shape

			new_stem = f'{img_id:0>12}'

			# 保存图像
			if Path(img_path).suffix.lower() == '.jpg':
				shutil.copyfile(img_path, f'{tp}/{trval}2017/{new_stem}.jpg')
			else:
				cv2.imwrite(f'{tp}/{trval}2017/{new_stem}.jpg')

			# 保存标注
			images.append({
				'date_captured': '2021',
				'file_name': f'{new_stem}.jpg',
				'id': img_id,
				'height': h,
				'width': w,
			})

			annot = self.label2annot(txt_path, h, w, img_id=img_id)
			assert len(annot) > 0, f'{txt_path} is empty.'
			annotations.extend(annot)

		json_data = {
			'info': self.info,
			'images': images,
			'licenses': self.licenses,
			'type': self.type,
			'annotations': annotations,
			'categories': self.categories
		}

		tp_json = f'{tp}/annotations/instances_{trval}2017.json'
		with open(tp_json, 'w', encoding='utf-8') as f:
			json.dump(json_data, f, ensure_ascii=False)

	def label2annot(self, txt_path, h, w, img_id):
		annotation = []
		with open(txt_path, 'r') as f:
			labels = f.readlines()
		labels = [x.replace('\n', '') for x in labels]
		for lb in labels:
			cls = lb.split()[0]
			bbox = lb.split()[1::]  # xc yc w h in fraction
			assert len(bbox) == 4
			segmentation, bbox, area = self._get_annotation(bbox, h, w)  # 转为x1y1wh了
			annotation.append({
				'segmentation': segmentation,
				'area': area,
				'iscrowd': 0,
				'image_id': img_id,
				'bbox': bbox,
				'category_id': int(cls) + 1,
				'id': self.annotation_id,
			})
			self.annotation_id += 1
		return annotation

	@staticmethod
	def _get_annotation(vertex_info, height, width):
		cx, cy, w, h = [float(i) for i in vertex_info]

		cx = cx * width
		cy = cy * height
		box_w = w * width
		box_h = h * height

		# left top
		x0 = max(cx - box_w / 2, 0)
		y0 = max(cy - box_h / 2, 0)

		# right bottomt
		x1 = min(x0 + box_w, width)
		y1 = min(y0 + box_h, height)

		segmentation = [[x0, y0, x1, y0, x1, y1, x0, y1]]
		bbox = [x0, y0, box_w, box_h]
		area = box_w * box_h
		return segmentation, bbox, area


if __name__ == '__main__':
	sp = '/home/zzd/datasets/crosswalk/fogged_train_data_v5_format'
	tp = "/home/zzd/datasets/crosswalk/fogged_train_data_coco_format"
	y2c = YOLO2COCO(sp, tp)
	y2c()
