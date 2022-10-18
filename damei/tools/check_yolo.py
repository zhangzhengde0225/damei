"""
检查YOLO数据集
"""
import os
from pathlib import Path

import cv2
import numpy as np

from ..controls.color import ColorControl
from ..functions import general


class CheckYOLO(object):
    def __init__(self, dp=None):
        self.dp = dp
        self.suffixes = ['.jpg', '.png', '.bmp', '.tif', '.tiff', '.jpeg']

    def __call__(self, trte=None, **kwargs):
        dp = self.dp
        trte = trte if trte else 'train'
        print(dp)
        self.classes = kwargs.pop('classes', self.get_classes())

        num_color = len(self.classes) if self.classes is not None else 1000
        self.colors = ColorControl(num=num_color, random_color=True).color
        self.check(trte=trte, **kwargs)

    def get_classes(self):
        dp = self.dp
        if os.path.exists(f'{dp}/classes.txt'):
            with open(f'{dp}/classes.txt', 'r') as f:
                classes = f.readlines()
            classes = [x.replace('\n', '') for x in classes]
        else:
            classes = None
        return classes

    def check(self, trte=None, p=None, pl=None, names=None, **kwargs):
        p = p if p else f'{self.dp}/images/{trte}'
        pl = pl if pl else f'{self.dp}/labels/{trte}'
        save_dir = kwargs.get('save_dir', None)
        show = kwargs.get('show', True)
        names = names if names else self.classes
        if os.path.exists(p):
            pass
        else:
            print(f'Directory {p} does not exists, redirect to {self.dp}')
            p = self.dp
            pl = self.dp

        imgs = [f'{p}/{x}' for x in os.listdir(p) if (Path(x).suffix in self.suffixes) and (x.startswith('.') is False)]
        imgs = sorted(imgs)
        # print(imgs)
        print(f'all classes: {names} len {len(imgs)}')
        for i, imgp in enumerate(imgs):
            assert os.path.exists(imgp), f'{imgp} not exists.'
            stem = Path(imgp).stem
            labelp = f'{pl}/{Path(imgp).stem}.txt'
            assert os.path.exists(labelp), f'{labelp} not exists.'
            img = cv2.imread(imgp)
            h, w, c = img.shape

            with open(labelp, 'r') as f:
                label = f.readlines()
            label = np.array([x.split() for x in label], dtype=np.float32)

            if len(label) == 0:
                classes = []
                lbs = []
            else:
                classes = label[:, 0]
                bboxes = label[:, 1::]
                bboxes = general.xywh2xyxy(bboxes)
                lbs = []
                for j in range(len(label)):
                    cls = classes[j]
                    bbox = bboxes[j]
                    bbox[0] *= w
                    bbox[1] *= h
                    bbox[2] *= w
                    bbox[3] *= h

                    label = f'{names[int(cls)]}' if names is not None else f'{int(cls)}'
                    lbs.append(label)
                    color = None

                    general.plot_one_box_trace_pose_status(
                        bbox, img, label=label, color=color)

            print(f'Stem: {stem}. Image shape: {img.shape}. Num_targets: {len(classes)}. Labels: {lbs}')
            # cr = np.any(label[:, 0] == 1)
            if save_dir is not None:
                save_path = f'{save_dir}/plotted_{stem}.jpg'
                print(f'plot to {save_path}')
                cv2.imwrite(save_path, img)

            if show:
                if np.max(img.shape) > 1920:
                    h, w, c = img.shape
                    img = cv2.resize(img, (int(0.5 * w), int(0.5 * h)))
                cv2.imshow('xx', img)
                if cv2.waitKey(0) == ord('q'):
                    exit()
