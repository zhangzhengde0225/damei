"""
基于动态步长、动态尺寸和随机尺度的动态背景算法。
作用是：给定一张大的原始背景图，实现随机采样，生成多张背景图。
用于数据增强。
"""
import os, sys
import numpy as np
import cv2
from copy import deepcopy
from pathlib import Path
import shutil
import damei as dm

logger = dm.getLogger('dynamic_background.py')


class DynamicBackground(object):
    def __init__(self, sp=None, window_size=640, stride_ratio=3 / 4, suffix='.jpg'):
        self.win_wh = (window_size, window_size) if isinstance(window_size, int) else window_size  # window size
        self.stride_ratio = stride_ratio
        self.sp = sp
        self.suffix = suffix

    def __call__(self, *args, **kwargs):
        sp = kwargs.get('sp', self.sp)
        assert sp, 'source path is None'
        assert isinstance(sp, str), 'source path is not str'
        if os.path.isfile(sp):
            imgs = [sp]
        elif os.path.isdir(sp):
            imgs = [f'{sp}/{x}' for x in os.listdir(sp) if x.endswith(self.suffix)]
        else:
            raise Exception(f'{sp} is not a file or dir')
        logger.info(f'Deal with {len(imgs)} images in {sp}')
        count = 0
        for imgp in imgs:
            num_sub_bg = self.gen_from_single_img(imgp, **kwargs)
            count += num_sub_bg
        logger.info(f'Done! Total {count} sub-backgrounds')

    def gen_from_single_img(self, imgp, win_wh=None, save_dir=None, show=True, save=False, skip_exist=True, ):
        """
        单图动态，输入一张图，输出动态步长、等比数列尺度的背景图
        """
        save_dir = save_dir if save_dir else f'{Path(imgp).parent}/{Path(imgp).stem}'
        suffix = Path(imgp).suffix
        if os.path.exists(save_dir):
            if skip_exist:
                return len(os.listdir(save_dir))
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        img = cv2.imread(imgp)
        win_wh = win_wh if win_wh else self.win_wh

        # 计算nx, ny
        h, w, c = img.shape
        raw_wh = [w, h]
        stride_ratio = self.stride_ratio
        bboxes, num_sub_bgs = self.cal_bboxes(raw_wh, win_wh, stride_ratio)  # 不同尺度的list，每个元素是[n, 4]
        logger.info(f'Deal with img: {imgp} shape: {img.shape} num_sub_bg: {len(bboxes)} '
                    f'num_of_each_scale: {num_sub_bgs}')

        # screen
        if len(bboxes) > 200:
            bboxes = bboxes[num_sub_bgs[0]::]
        if show or save:
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox  # 这个bbox是指切片的位置
                sub_bg = deepcopy(img)[y1:y2, x1:x2, :]
                print(f'\r[{i + 1:>3}/{len(bboxes)}] bbox: {bbox} shape: {sub_bg.shape}', end='')
                if show:
                    cv2.imshow('xx', sub_bg)
                    if cv2.waitKey(0) == ord('q'):
                        cv2.destroyAllWindows()
                        break
                if save:
                    cv2.imwrite(f'{save_dir}/win_{i:0>6}{suffix}', sub_bg)
            print('')
        return len(bboxes)

    def cal_bboxes(self, raw_wh, win_wh, stride_ratio):
        # 计算一张图，所有尺度下的bboxes
        new_win_whs, new_nss, new_ss = self.multi_scale(raw_wh=raw_wh, win_wh=win_wh, stride_ratio=stride_ratio)

        bboxes = np.zeros((0, 4), dtype=np.int32)
        num_sub_bgs = []
        for i, win_wh in enumerate(new_win_whs):
            a, b = win_wh
            nx, ny = new_nss[i]  # nx, ny
            sx, sy = new_ss[i]  # sx, sy
            tmp_bboxes = self.cal_bboxes_single_scale(nx, ny, sx, sy, a, b)  # 一个尺度就有多个bboxes
            bboxes = np.concatenate((bboxes, tmp_bboxes), axis=0)
            num_sub_bgs.append(len(tmp_bboxes))
            # print(f'{len(tmp_bboxes)} ', end='')
        return bboxes, num_sub_bgs

    def cal_bboxes_single_scale(self, nx, ny, sx2, sy2, a, b):
        # 计算单一尺度下所有生成子图的bbox
        bboxes = []
        for i in range(int(nx)):
            for j in range(int(ny)):
                x1 = i * sx2
                x2 = i * sx2 + a
                y1 = j * sy2
                y2 = j * sy2 + b
                bboxes.append([x1, y1, x2, y2])
        bboxes = np.array(bboxes, dtype=np.int32)
        return bboxes

    def multi_scale(self, raw_wh, win_wh, stride_ratio):
        # 变化window size，直到nx ny中最小的为1.
        ra, rb = win_wh  # raw a, raw b, a和b是窗口的宽和高
        new_win_whs = []
        new_ns = []
        new_strides = []
        m = 1  # 这么多次
        q = 1.25
        while True:
            ram = ra * np.power(q, m - 1)  # 等比数列
            rbm = rb * np.power(q, m - 1)
            if ram > raw_wh[0] or rbm > raw_wh[1]:
                break
            new_stride = (ram * stride_ratio, rbm * stride_ratio)
            nx, ny = self.cal_nxny(raw_wh, win_wh=(ram, rbm), stride=new_stride)
            if np.min([nx, ny]) < 1:
                break

            sx2, sy2 = self.cal_new_stride(raw_wh, win_wh=(ram, rbm), ns=(nx, ny))
            if sx2 is None or sy2 is None:
                break

            # minnxy = np.min([nx, ny])
            # print(
            #    f'm: {m:>2} ram: {int(ram):>4} rbm: {int(rbm):>4} power: {np.power(q, m-1):.2f} nx: {nx} ny: {ny} '
            #    f'min: {minnxy} sx2: {int(sx2)} sy2: {int(sy2)}')

            new_win_whs.append([ram, rbm])
            new_ns.append([nx, ny])
            new_strides.append([sx2, sy2])

            m += 1
            if m == 15:
                break
        new_win_whs = np.array(new_win_whs, dtype=np.int32)
        new_ns = np.array(new_ns, dtype=np.int32)
        new_strides = np.array(new_strides, dtype=np.int32)
        # print(f'new_win_whs: {new_win_whs}')
        return new_win_whs, new_ns, new_strides

    def cal_nxny(self, raw_wh, win_wh, stride):
        w, h = raw_wh
        a, b = win_wh
        sx, sy = stride
        nx = np.round((w - a) / sx) + 1  # 6
        ny = np.round((h - b) / sy) + 1  # 4
        return nx, ny

    def cal_new_stride(self, raw_wh, win_wh, ns, epsilon=1e-8):
        w, h = raw_wh
        a, b = win_wh
        nx, ny = ns
        if a > w or b > h:
            raise NameError(f'raw_wh: {raw_wh} win_wh: {win_wh} at least one of win wh larger than raw wh')
        if nx < 1 or ny < 1:
            raise NameError(f'nx: {nx} ny: {ny}, at least one of them less than 1')
        sx2 = (w - a) / (nx - 1 + epsilon) if nx != 1 else 0
        sy2 = (h - b) / (ny - 1 + epsilon) if ny != 1 else 0
        return sx2, sy2


if __name__ == '__main__':
    p = f'/home/zzd/datasets/transmission_line/background'
    db = DynamicBackground(sp=p)
