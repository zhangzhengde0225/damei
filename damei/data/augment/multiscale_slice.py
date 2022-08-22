import os, sys
import numpy as np
import cv2
import copy
import damei as dm
from pathlib import Path
import json

logger = dm.getLogger('multiscale_slice.py')


class AugmentMSS(object):
    """
    A multi-scale slice augmentation method based on slide window.
    """

    def __init__(self, sp=None, tp=None):
        self.sp = sp
        self.tp = tp

    def __call__(self, *args, **kwargs):
        if not self.sp:
            raise Exception('source path is None')
        elif isinstance(self.sp, str):
            if os.path.isfile(self.sp):
                imgs = [self.sp]
            elif os.path.isdir(self.sp):
                imgs = [f'{self.sp}/{x}' for x in os.listdir(self.sp) if x.endswith('.jpg')]
                # print(imgs, len(imgs))
            else:
                raise Exception(f'{self.sp} is not a file or dir')
        elif isinstance(self.sp, list):
            imgs = self.sp
        else:
            raise Exception('source path is not str')
        # for imgp in imgs:
        #     self.single_img_slice(imgp)
        if len(imgs) == 0:
            raise Exception('imgs is empty, please check your source path')
        self.single_img_slice(imgs, **kwargs)

    def single_img_slice(self, imgp, **kwargs):
        slices_object = MultiScaleSlice(
            img_paths=imgp,
            **kwargs,
        )
        tp = self.tp
        if tp is None:
            imgp0 = imgp if isinstance(imgp, str) else imgp[0]
            tp = f'{Path(imgp0).parent}/augment_mss'
        if not os.path.exists(tp):
            os.makedirs(tp)
        count = 0
        for i, (img, slice_path, im0, anno) in enumerate(slices_object):
            if anno is None:
                # print('img is None')
                continue
            # 保存切片
            slice_stem = Path(slice_path).stem
            cv2.imwrite(f'{tp}/{slice_stem}{Path(slice_path).suffix}', img)
            with open(f'{tp}/{slice_stem}.json', 'w') as f:
                json.dump(anno, f, indent=4)
            count += 1
            print(f'\rmss: slice with targets: {count}, all slices {i}, {img.shape} {Path(slice_path).stem}', end='')
        print()


class MultiScaleSlice(object):
    """
    多级尺寸切片，加载器，是一个生成器
    """

    def __init__(self, img_paths, min_wh=640, max_wh=None, stride_ratio=0.5, pss_factor=1.25,
                 need_annotation=True, anno_fmt='json', sort='min2max', maxl=4):
        """
        :param img_paths: 图片路径
        :param min_wh: 最小尺寸，即初始尺寸
        :param max_wh: 最大尺寸，默认是原始图像的最大尺寸
        :param stride_ratio: 切片移动的步长，默认的当前尺度的1/2
        :param pss_factor: 尺寸按等比数列缩放的等比因子，默认是1.25
        :param need_annotation: 是否需要标注
        :param anno_fmt: 标注格式，默认是json
        :param sort: 排序方式，默认是min2max，最小到最大
        :param maxl: 图像宽高最大支持位数，默认4，即9999*9999分辨率的图像
        """
        assert len(img_paths) > 0, f'img_paths is empty, please check your source path: {img_paths}'
        # 关于图片的切片窗口的参数
        self.min_wh = (min_wh, min_wh) if isinstance(min_wh, int) else min_wh
        self.max_wh = max_wh
        self.stride_ratio = stride_ratio
        self.pss_factor = pss_factor
        self.sort = sort
        self.maxl = maxl  # 图像宽高最大支持位数，默认4，即9999*9999分辨率的图像

        # 关于返回真值的参数
        self.need_annotation = need_annotation
        self.anno_format = anno_fmt

        self._current_img = None
        self.img_idx = 0  # 用来记录当前图片的索引

        if isinstance(img_paths, str):
            img_paths = [img_paths]
        self._img_paths = img_paths
        self._init_sw()  # 初始化，注，仅仅初始化第一张

    def _init_sw(self):
        """
        初始化每张大图的每个尺度下的所有切片窗口
        """

        # img_paths = sorted(img_paths)

        slice_windows = []
        num_slice_windows = []

        # assert os.path.exists(img_path), f'{img_path} not exists'
        # img = cv2.imread(img_path)  # 读取图像分析很慢，需要优化
        img = self.cimg
        raw_wh = img.shape[:2]

        bboxes, num_sub_figs = self.cal_bboxes(
            raw_wh=raw_wh,
            win_wh=self.min_wh,
            stride_ratio=self.stride_ratio,
        )
        slice_windows.extend(bboxes)

        num_sub_figs_sum = np.sum(num_sub_figs)
        num_slice_windows.append(num_sub_figs_sum)
        # print(f'\ranaylizing [{i+1:>3}/{len(img_paths)}]{imgp} ... {num_sub_figs_sum}', end='')
        slice_windows = np.array(slice_windows)

        if self.sort == 'min2max':
            pass
        elif self.sort == 'max2min':
            slice_windows = slice_windows[::-1]
            num_slice_windows = num_slice_windows[::-1]
        else:
            raise NotImplementedError(f'{self.sort} is not implemented')

        self._slice_windows = slice_windows
        self._num_slice_windows = num_slice_windows
        # self._img_paths = img_paths

    def __iter__(self):
        self.img_idx = 0  # 用来记录当前图片的索引
        self.slice_idx = -1  # 用来记录切片图像的索引
        return self

    def __next__(self):
        """
        迭代器，每次返回从一张大图中的一个切片，每次next仅取1个切片。
        :return:
            slicee: 切片，ndarray, (h, w, c)
            slice_imgp: 原始图片路径，格式：原始图像路径_(窗口高宽)_(滑动窗口x1y1x2y2).后缀
            im0: 原始图片
        """
        self.slice_idx += 1  # 初始是0，每次加1

        # if self.img_idx >= len(self._img_paths):  # img_idx在在切片索引超过总切片时增加1
        #     raise StopIteration
        im0 = self.cimg
        if im0 is None:
            raise StopIteration
        imgp = self.img_paths[self.img_idx]
        sw = self.sw[self.slice_idx]  # slice window x1, y1, x2, y2
        w, h = sw[2] - sw[0], sw[3] - sw[1]

        slice_imgp = f'{Path(imgp).parent}/{Path(imgp).stem}_' \
                     f'{h:0>{self.maxl}},{w:0>{self.maxl}}_' \
                     f'{",".join([f"{x:0>{self.maxl}}" for x in sw])}' \
                     f'{Path(imgp).suffix}'

        slicee = im0[sw[1]:sw[3], sw[0]:sw[2], :]  # 滑动窗口的切片，(h, w, 3), h和w是不同尺度的高宽
        # print(img.shape, slicee.shape, sw)
        # print(self.count, len(self.sw), self.sw.shape, self.nsw)

        # self.count += 1
        if self.need_annotation:
            anno = self.get_annotation(imgp, sw)
            return slicee, slice_imgp, im0, anno

        return slicee, slice_imgp, im0

    def __len__(self):
        """不太准确，这里只是当前目标的窗口"""
        return len(self._slice_windows)

    @property
    def cimg(self):
        """
        读取当前原图
        """
        if self._current_img is None:  # 读取第一张
            # img_path = self.img_paths[self.raw_img_idx]
            img_path = self.img_paths[self.img_idx]
            self._current_img = cv2.imread(img_path)
            h, w, c = self._current_img.shape
            assert len(str(h)) <= self.maxl, f'raw image is too large, {h} {len(str(h))}'
            assert len(str(w)) <= self.maxl, f'raw image is too large, {w}'
            return self._current_img
        else:  # 需要判断是否切换原始图片
            called_num_slice_windows = np.sum(self.nsw[:self.img_idx + 1:])
            # print(f'count: {self.count} called: {called_num_slice_windows}')
            if self.slice_idx < called_num_slice_windows:  # 不切换
                return self._current_img
            else:  # 切换，
                self.img_idx += 1
                if self.img_idx >= len(self.img_paths):
                    return None
                # 切换的同时slice_idx置零
                img_path = self.img_paths[self.img_idx]
                self._current_img = cv2.imread(img_path)
                self.slice_idx = 0
                h, w, c = self._current_img.shape
                assert len(str(h)) <= self.maxl, f'raw image is too large, {h}'
                assert len(str(w)) <= self.maxl, f'raw image is too large, {w}'
                return self._current_img

    @property
    def sw(self):
        """slice windows 切片窗口 (n, 4)"""
        return self._slice_windows

    @property
    def nsw(self):
        """num slice windows """
        return self._num_slice_windows

    @property
    def img_paths(self):
        """
        img paths
        """
        return self._img_paths

    def get_annotation(self, imgp, sw):
        if self.anno_format == 'json':
            label_path = f'{Path(imgp).parent}/{Path(imgp).stem}.json'
            assert os.path.exists(label_path), f'{label_path} not exists'
            slice_anno = self.get_json_slice_anno(label_path, sw)
            return slice_anno
        else:
            raise NotImplementedError(f'anno_format {self.anno_format} not implemented')

    def get_json_slice_anno(self, label_path, sw, iou_thresh_target_vs_window=0.3,
                            iou_thresh_clipped_target_vs_target=0.3):
        """
        读取labelme格式的json文件，获取滑动窗口内的标注，如果没有任何目标，返回None
        """
        with open(label_path, 'r') as f:
            anno = json.load(f)

        # 这部分仅仅针对labelme的格式有效
        new_anno = dict()
        new_anno['version'] = anno['version']
        new_anno['flags'] = anno['flags']
        imgp = Path(anno['imagePath'])
        w, h = sw[2] - sw[0], sw[3] - sw[1]
        img_path = f'{Path(imgp).parent}/{Path(imgp).stem}_' \
                   f'{h:0>{self.maxl}},{w:0>{self.maxl}}_' \
                   f'{",".join([f"{x:0>{self.maxl}}" for x in sw])}' \
                   f'{Path(imgp).suffix}'
        new_anno['imagePath'] = img_path
        new_anno['imageData'] = None
        new_anno['imageHeight'] = int(h)
        new_anno['imageWidth'] = int(w)
        new_anno['shapes'] = []
        has_target = False

        for shape in anno['shapes']:
            new_shape = dict()

            a, b = sw[0], sw[1]  # 原坐标系原点到新坐标系原点的平移距离，切片窗口的左上角坐标

            # 原坐标系x和新坐标系的变换：x' = x - a

            # print(shape.keys())  # dict_keys(['label', 'points', 'shape_type', 'group_id', 'flags'])
            new_shape['label'] = shape['label']
            new_shape['shape_type'] = shape['shape_type']
            new_shape['group_id'] = shape['group_id']
            new_shape['flags'] = shape['flags']
            points = np.array(shape['points'], dtype=np.int32)  # (n, 2)
            bbox = dm.general.pts2bbox(points)  # (x1, y1, x2, y2) 目标的bbox
            iou = dm.general.bbox_iou(bbox, sw, x1y1x2y2=True, return_np=True)  # iou with slice window

            if iou == 0.:  # 与当前窗口无交集的目标不要
                continue
            # 判断窗口和目标谁大
            area_sw = (sw[2] - sw[0]) * (sw[3] - sw[1])
            area_bbox = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            if iou <= iou_thresh_target_vs_window and area_sw < area_bbox:
                continue

            # 变换
            new_points = points - np.array([a, b], dtype=np.int32)  # 平移变换，目标的所有点在新坐标系中的坐标
            win_w, win_h = sw[2] - sw[0], sw[3] - sw[1]
            # 限制在窗口内, clip掉超出窗口的点
            raw_points = new_points.copy()
            new_points[:, 0][new_points[:, 0] < 0] = 0
            new_points[:, 0][new_points[:, 0] > win_w] = win_w
            new_points[:, 1][new_points[:, 1] < 0] = 0
            new_points[:, 1][new_points[:, 1] > win_h] = win_h

            bbox_raw_pts = dm.general.pts2bbox(raw_points)
            bbox_new_pts = dm.general.pts2bbox(new_points)
            if dm.general.bbox_iou(bbox_raw_pts, bbox_new_pts, x1y1x2y2=True,
                                   return_np=True) < iou_thresh_clipped_target_vs_target:
                continue

            has_target = True
            # print(new_points.shape, iou, label_path, sw, new_points, new_shape['label'], bbox)
            new_shape['points'] = new_points.tolist()
            new_anno['shapes'].append(new_shape)

        if has_target:
            return new_anno
        else:
            return None

    def cal_bboxes(self, raw_wh, win_wh, stride_ratio):
        """
        计算一张图，所有尺度下的bboxes
        :param raw_wh: 原始图像的尺寸
        :param win_wh: 切片的尺寸
        :param stride_ratio: 切片移动的步长，默认的当前尺度的1/2
        :return:
            bboxes: 切片的bboxes, ndarray (n, 4), n个子窗，每个子窗是一个bbox，4是x1y1x2y2
            num_sub_figs: 子窗的数量, list, 例如：[160， 104， 60， 40， 24， 15， 6， 3] 对应每个尺度下的子窗数量
        """
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

    def multi_scale(self, raw_wh, win_wh, stride_ratio):
        """
        变化window size，直到nx ny中最小的为1.
        :param raw_wh: 原始图像的尺寸
        :param win_wh: 初始window size
        :param stride_ratio: 切片移动的步长，默认的当前尺度的1/2
        :return:
            new_win_whs: 变化后的window size
            new_nss: 变化后的nx, ny
            new_strides: 变化后的sx, sy
        """

        ra, rb = win_wh  # raw a, raw b, a和b是窗口的宽和高
        new_win_whs = []
        new_ns = []
        new_strides = []
        m = 1  # 这么多次
        q = self.pss_factor
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

            # 这个写法宽高相反了
            # new_win_whs.append([ram, rbm])
            # new_ns.append([nx, ny])
            # new_strides.append([sx2, sy2])
            # 这个写法才对
            new_win_whs.append([rbm, ram])
            new_ns.append([ny, nx])
            new_strides.append([sy2, sx2])

            m += 1
            if m == 15:
                logger.warn('multi_scale: m == 15, break')
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

    def cal_bboxes_single_scale(self, nx, ny, sx2, sy2, a, b):
        """
        计算单一尺度下所有生成子图的bbox
        """

        bboxes = []
        for i in range(int(nx)):
            for j in range(int(ny)):
                x1 = i * sx2
                x2 = i * sx2 + a
                y1 = j * sy2
                y2 = j * sy2 + b
                bboxes.append([x1, y1, x2, y2])
                # print(f'{x2-x1} {y2-y1}')
        bboxes = np.array(bboxes, dtype=np.int32)
        return bboxes
