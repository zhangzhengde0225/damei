import cv2


class Tools(object):
	def __init__(self):
		pass

	def test(self):
		print('z')

	def cap_video_save(self, path, new_size=None, rotate=False, save_dir=None):
		cap = cv2.VideoCapture(path)
		assert cap.isOpened(), f'Failed to open {path}'
		w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = cap.get(cv2.CAP_PROP_FPS) % 100
		print(f'{path}, w: {w}, h: {h}, fps: {fps}')
		record_flag = False
		idx = 0
		while True:
			ret, frame = cap.read()
			if rotate:
				frame = cv2.rotate(frame, cv2.ROTATE_180)
			if new_size:
				frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_CUBIC)
			cv2.imshow('x', frame)
			if cv2.waitKeyEx(1) == ord('q'):
				cap.release()
				break
			if cv2.waitKeyEx(1) == ord('r'):
				record_flag = True
			if record_flag and save_dir is not None:
				print(f'\rrecording ... ', end='')
				cv2.imwrite(f'{save_dir}/{idx:0>6}.jpg', frame)

			idx += 1
			print(f' {idx:0>6}.jpg')

	def yolo2coco(self, sp, tp=None):
		"""
		transform datatset of yolov5 format to coco format
		:param sp: source path, path to yolov5 dataset dir, which contains images, label dir and classes.txt
		:param sp: target path, output path
		"""
		from .yolo2coco import YOLO2COCO
		# print(f'sp: {sp}\ntp: {tp}')
		y2c = YOLO2COCO(sp, tp=tp)
		y2c()

	def check_coco(self, json_path, img_dir=None, line_thicknes=3, only_show_id=None):
		"""check coco dataset
		:param json_path, path to coco json file, for example: xxx/annotations/instances_train2017.json
		:param img_dir, image dir
		:param line_thicknes: thickness for rectangle of bbox
		:param only_show_id: only show the img with this id if is None
		"""
		from .check_coco import CheckCOCO
		cc = CheckCOCO(json_path, img_dir=img_dir, line_thickness=line_thicknes)
		cc(only_show_id=only_show_id)

	def check_YOLO(self, dp):
		from .check_yolo import CheckYOLO
		cy = CheckYOLO(dp=dp)
		cy()
