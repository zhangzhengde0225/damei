"""
从小样本标注产生，标注是points类型
"""
import os, sys
from pathlib import Path
import json
import numpy as np
from copy import deepcopy
try:
    import cv2
except:
    pass
import random
import time
import shutil
# from tqdm import tqdm
# from shapely.geometry import Polygon
import damei as dm
from collections import OrderedDict
import argparse

logger = dm.get_logger('augment_dataset.py')


class AugmentData(object):
    def __init__(self, dp,
                 tp=None,
                 adp=None,
                 anno_fmt='labelme',
                 out_fmt='YOLOfmt',
                 suffix='.jpg',
                 use_noise_background=False,
                 out_size=640,
                 save_mask=False,
                 *args,
                 **kwargs
                 ):
        """
        :param dp: source dataset path, str, e.g. /home/xxx/datasets/xxx_dataset
        :param tp: target dataset path, str, e.g. /home/xxx/datasets/xxx_dataset_augmented_v5_format
        :param adp: additional background path
        """
        self.dp = dp
        self.anno_fmt, self.out_fmt = self._init_io_format(anno_fmt, out_fmt)
        self.tp = tp if tp else f'{Path(self.dp).parent}/{Path(self.dp).stem}_augmented_{self.out_fmt}'
        self.adp = adp
        self.suffix = suffix
        self.use_noise_background = use_noise_background
        self.save_mask = save_mask
        self.ad_bgs = self._init_additional_backgrounds()
        self.ns = (out_size, out_size) if isinstance(out_size, int) else out_size  # new_size, w, h

    def _init_io_format(self, anno_fmt, out_fmt):
        if anno_fmt == 'labelme':
            pass
        # elif anno_fmt == 'xxx':  # reserve for future
        #     pass
        else:
            raise NotImplementedError(f'{anno_fmt} not supported')
        if out_fmt == 'YOLOfmt':
            pass
        else:
            raise NotImplementedError(f'{out_fmt} not supported')
        return anno_fmt, out_fmt

    def _init_additional_backgrounds(self):
        """init additional background images"""
        imgps = []
        if self.adp is None:
            return imgps
        for root, dirs, files in os.walk(self.adp):
            imps = [f'{root}/{x}' for x in files if x.endswith(self.suffix)]
            imgps.extend(imps)
        # print(f'additional backgrounds: {len(imgps)}')
        logger.info(f'Additional backgrounds: {len(imgps)}')
        return imgps

    def __call__(self, *args, **kwargs):
        # classes, static, lb_per_img, labeled_imgs = self.analyse()  # classes是类别数目，static是每类有多少个标注的字典
        # print(f'Analyzing datasets in {self.dp} ...')
        num_augment_images = kwargs.get('num_augment_images', 8000)
        train_ratio = kwargs.get('train_ratio', 0.8)
        samples_ratio_per_cls = kwargs.get('samples_ratio_per_class', None)
        # erase_ratio = kwargs.get('erase_ratio', 0.0)
        assert 0 < train_ratio < 1, 'train_ratio must be in (0, 1)'

        if self.tp == '/path/to/dataset':
            return

        real_dp = os.path.realpath(self.dp)
        logger.info(f'Analyzing datasets in "{real_dp}" ...')
        analyse_results = self.analyse()
        imgps, im0s, names, maskss, clsss, correlation_matrices, num_targets, static = analyse_results
        # print('analysis done\n')
        self.labeled_imgs = imgps
        """
        增强策略：
        (1)每种类别增强为30张，大中小3种尺寸各10张
        (2)采用随机噪声背景， 640x640
        (3)随机位置贴图
        (4)每张图贴随机1-7随机数目的图，图之间IoU很小。
        """
        ns = self.ns
        tp = self.tp

        if os.path.exists(tp):
            need_ipt = True
            if need_ipt:
                ipt = input(f'Target path [{tp}] exists,\nRemove and Regenerate? [Yes]/no ')
                if ipt in ['N', 'No', 'no', 'n', 'NO']:
                    exit()
                elif ipt in ['', 'Yes', 'yes', 'YES', 'y', 'Y']:
                    shutil.rmtree(tp)
                else:
                    raise KeyError(f'{ipt}')
                print('')
            else:
                shutil.rmtree(tp)
        os.makedirs(f'{tp}/images/train')
        os.makedirs(f'{tp}/images/test')
        os.makedirs(f'{tp}/labels/train')
        os.makedirs(f'{tp}/labels/test')
        if self.save_mask:
            os.makedirs(f'{tp}/masks/train')
            os.makedirs(f'{tp}/masks/test')

        # 保存类别
        with open(f'{tp}/classes.txt', 'w') as f:
            f.writelines([f'{x}\n' for x in names])

        train_num = int(num_augment_images * train_ratio)
        test_num = int(num_augment_images - train_num)

        self.generate_dataset(analyse_results, trte='train', total_num=train_num,
                              samples_ratio_per_cls=samples_ratio_per_cls, **kwargs)
        self.generate_dataset(analyse_results, trte='test', total_num=test_num,
                              samples_ratio_per_cls=samples_ratio_per_cls, **kwargs)
        logger.info(f'Dataset generated in "{tp}"')

    def generate_dataset(self, analyse_results, trte, total_num, samples_ratio_per_cls, **kwargs):
        """
        :param analyse_results: small samples dataset analysis results
        :param trte: 'train' or 'test'
        :param total_num: total image will be generate
        :return:
        """
        imgps, im0s, names, maskss, clsss, correlation_matrices, num_targets, static = analyse_results
        tp = self.tp
        ns = self.ns

        """
        1.生成样本库
        """
        num_samples, sample_idxes = self.gen_samples_library(
            analyse_results, num_imgs=total_num,
            samples_ratio=samples_ratio_per_cls, **kwargs)
        # num_samples = [1, 2, 3, 2, ...]  # 指定每个合成图像有几个样本
        # sample_idxes = [...]  # 指定每个样本的索引
        logger.info(f'{trte} samples: {np.sum(num_samples)} {num_samples}')

        # 生成图像
        stem = 0
        for i, num_s in enumerate(num_samples):  # 6000次
            """
            2.生成背景图
            - 从背景库中随机选取，如果有目标，给随机数目打灰色马赛克
            """
            back_img, back_masks, back_clss = self.gen_background(analyse_results, **kwargs)  # list xyxy list clss

            """
            3.合成图
            """
            idxes = sample_idxes[i:i + num_s]  # 这样图的目标, 长度k，元素[m, n]
            syn_img, syn_masks, syn_clss = self.synthesis_img(analyse_results, idxes, back_img, back_masks, back_clss,
                                                              **kwargs)

            bboxes = [self.mask2bbox(mask) for mask in syn_masks]  # x1y1x2y2 in pixel, 可能会有None
            bboxes = [x for x in bboxes if x is not None]
            if len(bboxes) == 0:
                continue  # 不保存了呀
            # x1y1x2y2 in pixel to x1y1x2y2 in fraction
            try:
                bboxes = np.array(bboxes, dtype=np.float32)
                bboxes[:, 0] /= ns[0]

            except:
                self.imshow(img=syn_img, show_name='none')
                raise NameError(f'{len(syn_clss)} {len(syn_masks)} {bboxes}')
            bboxes[:, 1] /= ns[1]
            bboxes[:, 2] /= ns[0]
            bboxes[:, 3] /= ns[1]  # x1y1x2y2 in fraction
            bboxes = [x for x in bboxes]
            # 画图看看对不对
            plot = False
            if plot:
                print(f'num_targets: {len(syn_clss)} targets: {syn_clss}')
                self.imshow(img=syn_img, bboxes=bboxes, clss=syn_clss, ns=self.ns)

            # 保存图像
            cv2.imwrite(f'{tp}/images/{trte}/{stem:0>6}{self.suffix}', syn_img)
            # 保存bbox标注
            bboxes = np.array(bboxes)
            # print(f'{bboxes} {bboxes.shape} ', end='')
            bboxes = dm.general.xyxy2xywh(bboxes)  # xcycwh
            bboxes_string = '\n'.join(
                [f'{names.index(syn_clss[idx])} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f} {x[3]:.6f}' for idx, x in
                 enumerate(bboxes)])
            with open(f'{tp}/labels/{trte}/{stem:0>6}.txt', 'w') as f:
                f.writelines(bboxes_string)
            # 保存mask
            if self.save_mask:
                masks = np.array(syn_masks)
                with open(f'{tp}/masks/{trte}/{stem:0>6}.npy', 'wb') as f:
                    np.save(f, masks)

            stem += 1

            print(f'\rSave {trte} {stem:0>8}{self.suffix} [{i + 1:>4}/{total_num}] targets: {len(syn_clss)}', end='')
        print('')

    def synthesis_img(self, analyse_results, idxes, back_img, back_masks, back_clss, **kwargs):
        """
        背景图和样本库图合成图像
        :param analyse_results:
        :param idxes:
        :param gen_img:
        :param back_masks:
        :param back_clss:
        :return:
        """
        imgps, im0s, names, maskss, clsss, corr_matrices, num_targets, static = analyse_results

        # 随机添加的目标
        # 背景中存在的目标
        syn_img_masks = back_masks  # list
        syn_img_clss = back_clss  # list
        # print(f'\n新增{len(idxes)}个目标，已存在{len(syn_img_masks)}个目标')
        for i, (m, n) in enumerate(idxes):
            im0 = im0s[m]
            h, w, c = im0.shape
            mask = maskss[m][n, ...]  # [640, 640, 1] 值0和255
            mh, mw, mc = mask.shape
            assert h == mh and w == mw
            # print(f'添加第{i+1}个目标，m: {m} n: {n} len clsss: {len(clsss)} clsss: {clsss} clss: {clsss[m]}')
            cls = clsss[m][n]
            # print(f'类别: {cls}')
            img, mask, matRotateZoom, translation_params = self.random_size_and_angle(
                mask, im0, existing_masks=syn_img_masks, **kwargs)  # 随机尺寸和角度，生成备选图像和mask
            if img is None and mask is None:
                continue
            syn_img_masks.append(mask)
            syn_img_clss.append(cls)
            # 根据关联性扩展mask
            corr_verctor = corr_matrices[m][n]
            for j in range(len(corr_verctor)):
                if j == n:  # 略过与自己的关联关系
                    continue
                corraltion = corr_verctor[j]
                if corraltion < 0.9:
                    continue
                corr_mask = maskss[m][j, ...]  # 关联的mask
                new_corr_mask = cv2.warpAffine(deepcopy(corr_mask), matRotateZoom, (h, w))
                new_corr_mask = new_corr_mask[..., np.newaxis] if new_corr_mask.ndim == 2 else new_corr_mask
                new_corr_mask = self.translate_mask(new_corr_mask, translation_params=translation_params)
                corr_cls = clsss[m][j]  # 关联的类别
                syn_img_masks.append(new_corr_mask)
                syn_img_clss.append(corr_cls)
                # print(f'关联类别: {corr_cls}')
            # print(f'关联向量：{corr_verctor}')
            back_img = self.imgMask(img1=img, img2=back_img, mask=mask)  # 添加多次

            # small_img = np.multiply(im0, mask)
            # small_img = np.array(small_img, dtype=np.uint8)

        syn_img = back_img
        show = False
        if show:
            bboxes = [self.mask2bbox(mask) for mask in syn_img_masks]
            self.imshow(syn_img, bboxes=bboxes, clss=syn_img_clss, show_name='name')
        return syn_img, syn_img_masks, syn_img_clss

    def imgMask(self, img1, img2, mask):
        new_img = np.zeros_like(img1, dtype=np.uint8)
        # mask = np.array([mask]*new_img.shape[-1], dtype=np.uint8)
        # mask = np.transpose(mask, (1, 2, 0))
        mask = np.repeat(mask, 3, axis=2)
        # mask = np.array(mask/(mask.max()-mask.min()+1e-8), dtype=np.uint8)
        # i1 = np.multiply(img1, mask)
        # i2 = np.multiply(img2, 1-mask)
        # print(i1.shape, i2.shape, new_img.shape, mask.shape)
        # tmp = mask[mask != 1]
        # tmp = tmp[tmp != 0]
        # print(f'tmp: {len(tmp)}')
        # exit()
        new_img[mask > 127] = img1[mask > 127]
        new_img[mask < 127] = img2[mask < 127]
        return new_img

    def random_size_and_angle(self, mask, im0, existing_masks=[], **kwargs):
        h, w, c = im0.shape
        num_attempts = 0
        iou_threshold = kwargs.get('iou_threshold', 0.3)
        while True:
            # 随机尺寸
            size = int(np.clip(np.random.randn() / 6 * 2 + 1.3, 0.5, 2))
            angle = int(np.random.randint(0, 180))
            # 获取随机缩放和随机旋转矩阵
            matRotateZoom = self.get_rotation_zoom_matrix(mask, scale_factor=size, angle=angle)
            # 执行随机缩放和旋转
            new_mask = cv2.warpAffine(deepcopy(mask), matRotateZoom, (h, w))
            img = cv2.warpAffine(deepcopy(im0), matRotateZoom, (h, w))
            if new_mask.ndim == 2:
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
            if np.max(occ) <= iou_threshold:
                # print(f'完全无遮挡: {occ}')
                return img, new_mask, matRotateZoom, translation_params
            else:
                num_attempts += 1
                if num_attempts > 10:
                    return None, None, None, None
            num_occ50 = len([x for x in occ if x > iou_threshold])
            # print(f'num_iou>{iou_threshold} {num_occ50} existing mask: {len(existing_masks)}, size_scale: {size:.2f} random_angle: {angle:.0f}')
        print('')

    def random_position(self, mask, img):
        bbox = self.mask2bbox(mask)
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

    def translate_mask(self, mask, translation_params):
        x1, y1, x2, y2, bw, bh, stick_x, stick_y = translation_params
        new_mask = np.zeros_like(mask, dtype=np.uint8)  # [h, w, 1]
        cropped_bbox = mask[y1:y2, x1:x2, :]
        new_mask[stick_y:stick_y + bh, stick_x:stick_x + bw, :] = cropped_bbox
        return new_mask

    def get_rotation_zoom_matrix(self, mask, scale_factor=1, angle=0):
        """根据给定的尺寸因子，缩放目标的形成新的mask
        :param mask: [h, w, c] c=1
        """
        sf = scale_factor
        bbox = self.mask2bbox(mask)
        xc, yc = bbox[1] + (bbox[3] - bbox[1]) / 2, bbox[0] + (bbox[2] - bbox[0]) / 2
        matRotateZoom = cv2.getRotationMatrix2D((xc, yc), angle, sf)
        return matRotateZoom

    def mask2bbox(self, mask):
        if len(mask.shape) >= 3:
            if mask.shape[2] != 1:
                raise NameError(f'mask shape error, {mask.shape}')
            mask = mask[:, :, 0]
        if mask.max() - mask.min() == 0:
            return None
        mask = np.array(mask / (mask.max() - mask.min()), dtype=np.uint8)  # 转成0到1
        # columns = np.argmax(mask, axis=0)
        pts = np.argwhere(mask == 1)  # 返回的是索引
        if pts.size == 0:
            return None
        bbox = np.array([pts.min(axis=0)[1], pts.min(axis=0)[0], pts.max(axis=0)[1], pts.max(axis=0)[0]])
        # print(pts, pts.shape, bbox)
        return bbox

    def gen_samples_library(self, analyse_results, num_imgs, samples_ratio=None, **kwargs):
        """
        生成样本库
        - 要生成图像数目M=6000, 总目标数目=M*平均每张图目标数目
        - 随机每张图目标数数目，正态分布，1~2*tgt_per_imgs，中心是tgt_per_img
        - 样本均衡，设置样本比例
        :return
            num_samples: list, len 6000, 每个元素是第该图有几个目标
            random_idxes: list, len ~15000, 每个元素是随机的maskss的m和n索引
        """
        mean_scale_factor = kwargs.get('mean_scale_factor', 1.0)
        imgps, im0s, names, maskss, clsss, correlation_matrices, num_targets, static = analyse_results
        samples_ratio = samples_ratio if samples_ratio else [1] * len(names)  # 如1:1:1存储为[1, 1, 1]

        tgt_per_img = num_targets / len(imgps)  # targets per img
        cls_per_img = len(names) / len(imgps)  # class per img
        # num_samples = np.clip((np.random.randn(num_imgs, ) + 3) / 6 * (2 * tgt_per_img), 1,
        #                      2 * tgt_per_img)  # -3到3，标准正态分布，样本数量分布, 均值是3个
        # num_samples = np.clip((np.random.randn(num_imgs, )/4*2+1), 1, 3)  # 中心是1，可能1~3
        gaussian_mean = tgt_per_img * mean_scale_factor
        num_samples = np.random.normal(gaussian_mean, gaussian_mean / 2,
                                       num_imgs)  # 正态分布，均值是tgt_per_img，标准差是tgt_per_img/2
        num_samples = np.clip(num_samples, 1, 2 * tgt_per_img)  # 取值范围是1~2*tgt_per_img
        num_samples = np.round(num_samples).astype(np.int32)  # 四舍五入
        # num_samples = np.array(num_samples, dtype=np.int32)  # (6000,) 每个值是1到6，代表每张图的目标数目
        N = np.sum(num_samples)  # 总共的目标数目
        # 样本均衡
        raw_num_each_cls = [len(static[x]) for x in names]  # 原数据每类样本有多少个目标
        new_num_each_cls = [int(np.ceil(N / np.sum(samples_ratio) * x)) for x in samples_ratio]  # 向上取整
        # print(f'num of samples for each cls: {new_num_each_cls} total: {np.sum(new_num_each_cls)}')
        # NOTE: 这里有些不一样的是输电导线、断股和散股是关联出现的，一个输电中可能包含断股和散股目标，因此后面真实的样本数目会大于num_samples

        print(f'Generating samples library...')
        # 1.随机中已有标注中选择样本
        random_idxes = []  # list, 长度为k, 共计k个样本库，随机得从maskss中选择，每个元素是maskss的M和N索引，即第几张图和第几个目标索引
        assert len(new_num_each_cls) == len(
            names), f'len(new_num_each_cls)={len(new_num_each_cls)}, len(names)={len(names)}'
        for i, cls in enumerate(names):
            num_this_cls = new_num_each_cls[i]
            rat_this_cls = static[cls]  # raw all target this class [[m_idx, n_idx], ...]
            # 重复从该类已存在的目标中取样
            random_idx = np.random.randint(0, len(rat_this_cls), size=num_this_cls, dtype=np.int32)
            random_idx = [rat_this_cls[x] for x in random_idx]
            random_idxes.extend(random_idx)
        random.shuffle(random_idxes)  # len 15000

        """
        # 2.处理样本关联性，样本池 还是放到合成那里取吧
        new_imgps = []  # [M]
        new_maskss = []  # [M, n, ns, ns, 1]
        new_clsss = []  # [M, n]
        for i, (m, n) in enumerate(random_idxes):  # 15000次，m和n分别是图像的索引和目标的索引
            imgp = imgps[m]
            mask = maskss[m][n, ...]  # [640, 640, 1]
            cls = clsss[m][n]
        print(len(random_idxes), random_idxes[0], len(random_idxes))
            # new_maskss_idx = [rat_this_cls[idx] for idx in random_idx]  # maskss的双索引， num个，每
        """
        return num_samples, random_idxes

    def gen_background(self, analyse_results, **kwargs):
        """
        从背景库中生成背景：随机选择，随机给已存在的目标打马赛克
        :return:
        """
        imgps, im0s, names, maskss, clsss, correlation_matrices, num_targets, static = analyse_results
        ad_dir = os.path.abspath(self.adp) if self.adp else None
        if self.use_noise_background:  # 噪声背景
            bg_img = np.random.randint(0, 255, size=(self.ns[0], self.ns[1], 3), dtype=np.uint8)
            masks = []
            clss = []
        else:  # 从样本库中选择，样本库包含数据集中标注的图像、未标注的图像和额外的背景图像
            labeled_imgs = imgps  # [] list, 每个元素的ndarray

            ad_bgs = self.ad_bgs
            all_bgs = labeled_imgs + ad_bgs
            # 随机索引
            idx = int(np.random.randint(0, len(all_bgs), size=(1,), dtype=np.int32))
            imgp = all_bgs[idx]
            bg_img = cv2.imread(imgp) if isinstance(imgp, str) else imgp  # 硬盘读取图像
            bg_img = self.img_resize(bg_img, ns=self.ns, back_color=114)  # resized to new size
            if imgp in labeled_imgs:
                clss = clsss[idx]  # list, n
                masks = maskss[idx]  # ndarray, [n, 640, 640, 1]
                masks = [x for x in masks]
                # print('in labeled_imgs, target num:', len(masks), 'erase_ratio:', erase_ratio)
                bg_img, masks, clss = self.target_random_mosaic_mask(bg_img, masks, clss, **kwargs)
                # print(imgp, clss, bg_img.shape, masks, clss)
                # cv2.imshow('xx', bg_img)
                # cv2.waitKeyEx(0)
            else:
                clss = []
                masks = []
        return bg_img, masks, clss

    def target_random_mosaic_mask(self, img, masks, clss, **kwargs):
        """
        根据擦除比例给目标打上灰色马赛克，并调整有效蒙版和类别
        :param img: ndarray, [640, 640, 3]
        :param masks: list, [n, 640, 640, 1]  # n代表有n个目标，值为1的像素点是目标
        :param clss: list, [n]， n代表有n个目标，值每个目标的类型

        """
        # 随机需要打马赛克的数目
        erase_ratio = kwargs.get('erase_ratio', 0.0)  # 确定需要打马赛克的比例
        if isinstance(erase_ratio, float) or isinstance(erase_ratio, int):
            num_erase = int(erase_ratio * len(masks))
        elif isinstance(erase_ratio, list) or isinstance(erase_ratio, tuple):
            low, high = int(erase_ratio[0] * len(masks)), int(erase_ratio[1] * len(masks))
            assert low <= high, 'low must <= high'
            num_erase = int(np.random.uniform(low, high) * len(masks))  # 在[low, high]之间随机选择
        else:
            raise ValueError('erase_ratio must be int, float. list or tuple')

        # 随机索引
        indices = [x for x in range(len(masks))]  # [0, 1, 2, ..., n-1]  n个目标的索引
        random.shuffle(indices)  # 打乱
        indices_need_erase = indices[:num_erase]  # 随机选择num_erase个索引

        # masks = [x for x in masks]  # (n, 640, 640, 1) 转成 list, 长度n，每个元素[640, 640, 1]
        all_masks = deepcopy(masks)
        all_clss = deepcopy(clss)
        # for i in range(nnm):
        for idx in indices_need_erase:
            # 随机给第x个mask打马赛克
            # idx = np.random.randint(0, len(masks))
            mask = all_masks[idx]
            cls = all_clss[idx]
            img = self.mosaic_one_mask(img, mask)  # 灰色马赛克
            # masks.pop(idx)
            # clss.pop(idx)
        new_masks = [x for i, x in enumerate(masks) if i not in indices_need_erase]
        new_clss = [x for i, x in enumerate(clss) if i not in indices_need_erase]
        # return img, masks, clss
        return img, new_masks, new_clss

    def target_random_mosaic(self, img, bboxes, clss):
        """随机给已经存在的目标随机打马赛克， 目标类型是bbox"""
        nnm = np.random.randint(0, len(bboxes) + 1)  # num need mosaic
        for i in range(nnm):  # 循环要删除这么多次
            # 每次随机生成一个索引
            idx = np.random.randint(0, len(bboxes))  # 每次删除的索引是随机的
            bbox = bboxes[idx]
            img = self.mosaic_one_box(bbox, img)
            bboxes.pop(idx)
            clss.pop(idx)
        return img, bboxes, clss

    def mosaic_one_mask(self, img, mask):
        """根据蒙版打灰色马赛克"""
        # color = (0, 0, 0) if np.random.randn() < 0 else (255, 255, 255)
        color = (114, 114, 114)
        mask = mask[..., 0]  # (640, 640, 1) -> (640, 640)
        # white_idx = mask == 255
        img[mask == 255, :] = color
        return img

    def mosaic_one_box(self, bbox, img):
        h, w, c = bbox[3] - bbox[1], bbox[2] - bbox[0], 3
        color = (0, 0, 0) if np.random.randn() < 0 else (255, 255, 255)
        mosaic = np.zeros((h, w, c))
        mosaic[...] = color
        mosaic = np.array(mosaic, dtype=np.uint8)
        img = dm.general.imgAdd(mosaic, img, x=bbox[1], y=bbox[0], alpha=1)
        return img

    def read_json(self, json_file, bbox_resize=False, rimg=None):
        with open(json_file) as f:
            data = json.load(f)
        labels = data['shapes']  # 是个list，每个元素是dict
        bboxes = []
        clss = []
        for lb in labels:
            shape_type = lb['shape_type']
            assert shape_type == 'rectangle'
            cls = lb['label']
            pts = lb['points']  # [[x1, y1], [x2, y2]]
            bbox = [pts[0][0], pts[0][1], pts[1][0], pts[1][1]]
            bbox = np.array(bbox, dtype=np.int32)  # x1 y1 x2 y2
            if bbox_resize:
                assert rimg is not None
                bbox = self.bbox_resize(img=rimg, bbox=bbox, ns=self.ns)
            bboxes.append(bbox)
            clss.append(cls)
        return bboxes, clss

    def analyse(self):
        # 分析数据集，获取种类，数目等信息

        dp = os.path.abspath(self.dp)
        jsons = sorted([f'{dp}/{x}' for x in os.listdir(dp) if x.endswith('.json')])
        labeled_imgs = [f'{dp}/{Path(x).stem}{self.suffix}' for x in jsons if
                        os.path.exists(f'{dp}/{Path(x).stem}{self.suffix}')]

        imgps = []  # list, 长度M，M张图的绝对路径
        im0s = []  # list, 长度M，M张图读取后的路径
        names = []  # 数据集总的目标类别
        classess = []  # 每张图的每个目标的类别，list, 长度M，每个元素又是list,长度是N，N的每个元素是类别索引
        maskss = []  # list，长度M，共M张图，每个元素是masks[N, ns, ns, 1] N是每张图的目标数目

        correlation_matrices = []  # list，长度M，元素是关联矩阵[N, N], N个目标之间的关联性，包含关系
        num_targets = 0  # M张图所有目标N的求和
        static = {}  # 类别统计，键是类别名，值是list，每个元素是二元组[m, n]对应maskss中M和N的索引
        for i, js in enumerate(jsons):
            stem = Path(js).stem
            img_file = f'{dp}/{stem}{self.suffix}'
            assert os.path.exists(img_file), f'{img_file} not exists'
            img = cv2.imread(img_file)
            new_img = self.img_resize(deepcopy(img), ns=self.ns)  # resized to new size
            with open(js) as f:
                data = json.load(f)
            h, w = data['imageHeight'], data['imageWidth']
            labels = data['shapes']  # 是个list，每个元素是dict

            masks = np.zeros((len(labels), self.ns[0], self.ns[1], 1), dtype=np.uint8)  # mask [n, ns, ns, 1], n是n个目标
            classes = []
            for j, lb in enumerate(labels):
                shape_type = lb['shape_type']
                cls = lb['label']
                pts = lb['points']  # [[x1, y1], [x2, y2]]
                if shape_type == 'rectangle':
                    bbox = [pts[0][0], pts[0][1], pts[1][0], pts[1][1]]
                    bbox = np.array(bbox, dtype=np.int32)  # x1 y1 x2 y2
                    bbox = self.bbox_resize(img=img, bbox=bbox, ns=self.ns)
                    # cropped_bbox = deepcopy(new_img[bbox[1]:bbox[3], bbox[0]:bbox[2]])
                    masks[j, bbox[1]:bbox[3], bbox[0]:bbox[2], :] = 255  # 填充
                elif shape_type == 'polygon':
                    pts = np.array(np.ceil(pts), dtype=np.int32)  # 四舍五入像素取整, [n, 2] n个点，2是x和y
                    # bbox = np.array([pts.min(axis=0)[0], pts.min(axis=0)[1], pts.max(axis=0)[0], pts.max(axis=0)[1]])
                    pts = np.reshape(pts, (-1, 1, 2))  # opencv需要的pts是[n, 1, 2]的格式
                    # bbox = self.bbox_resize(img=img, bbox=bbox, ns=self.ns)
                    # poly = Polygon(np.ceil(pts))
                    # 处理该目标的蒙版mask
                    mask = np.zeros_like(img, dtype=np.uint8)
                    cv2.polylines(mask, pts, True, (255, 255, 255))
                    cv2.fillPoly(mask, [pts], (255, 255, 255))
                    mask = self.img_resize(mask, ns=self.ns, back_color=0)
                    masks[j, ...] = mask[:, :, 0:1]
                else:
                    raise NotImplementedError(f'label shape type {shape_type} not implemented.')

                # 保存类别
                names.append(cls) if cls not in names else None
                # 保存每个目标对应的类别
                classes.append(cls)
                # 保存类别统计
                if cls not in list(static.keys()):
                    static[cls] = [[i, j]]
                else:
                    static[cls].append([i, j])
                # 保存目标数目统计
                num_targets += 1
            # 处理关联性，物理意义是一个目标包含另一个目标的程度（不是被包含）
            corr_matrix = np.zeros((masks.shape[0], masks.shape[0]), dtype=np.float32)  # [n, n] 一张图的n个目标间的关联矩阵
            # print(masks.shape)  # [n, ns, ns, 1]  # n个目标
            for k in range(masks.shape[0]):
                mask1 = deepcopy(masks[k, ...])  # [ns, ns, 1]
                for l in range(masks.shape[0]):
                    mask2 = deepcopy(masks[l, ...])
                    correlation = self.cal_contains(mask1, mask2)
                    corr_matrix[k, l] = correlation

                    # print(k, l, correlation)
                    # cv2.imshow('mk1', mask1)
                    # cv2.imshow('mk2', mask2)
                    # if cv2.waitKeyEx(0) == ord('q'):
                    #     exit()

            # 保存图像路径
            imgps.append(img_file)
            im0s.append(new_img)
            # 保存蒙版
            maskss.append(masks)
            # 保存类别
            classess.append(classes)
            # 保存关联矩阵
            correlation_matrices.append(corr_matrix)

            # print(f'imgp: {img_file} classes: {classes}')
            print(f'[{i + 1:>3}/{len(jsons):>3}] stem: {stem} num_classes: {len(classes)}')
        lb_per_img = num_targets / len(jsons)
        cls_per_img = len(names) / len(jsons)

        analyse_result = dict()
        assert len(imgps) == len(im0s), f'imgps: {len(imgps)} im0s: {len(im0s)}'
        analyse_result['images'] = len(imgps)
        # analyse_result['classes'] = names
        analyse_result['num_classes'] = len(names)
        analyse_result['total_targets'] = num_targets

        class2num_dict = dict()
        for k, v in static.items():
            class2num_dict[k] = len(v)
        analyse_result['num_targets_of_each_class'] = class2num_dict

        analyse_result['label_per_image'] = f'{lb_per_img:.2f}'
        analyse_result['label_per_class'] = f'{cls_per_img:.2f}'

        analyse_result_string = dm.misc.dict2string(analyse_result)
        # logger.info('Dataset analyse results:')
        print('Dataset analyse results:')
        print(analyse_result_string)

        return [imgps, im0s, names, maskss, classess, correlation_matrices, num_targets, static]

    def bbox_resize(self, img, bbox, ns=(640, 640)):
        # bbox变换
        h, w = img.shape[:2]
        max_r = np.max([w / ns[0], h / ns[1]])
        rw, rh = int(w / max_r), int(h / max_r)  # resized w h
        padx, pady = int((ns[0] - rw) / 2), int((ns[1] - rh) / 2)

        bbox = [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]  # x1y1x2y2 in fraction
        bbox = np.array([bbox[0] * rw, bbox[1] * rh, bbox[2] * rw, bbox[3] * rh],
                        dtype=np.int32)  # x1y1x2y2 in pixel for new size
        padd = np.array([padx, pady, padx, pady])
        bbox = bbox + padd

        # print(f'raw wh: {w} {h}, r: {max_r} resized wh {rw} {rh} pad xy: {padx} {pady}')
        return bbox

    def cal_contains(self, mask1, mask2):
        """计算mask1和mask2中的目标的包含关系，即mask1与mask2的交集 除以 mask2的面积"""
        intersection = np.sum(mask2[mask1 == 255] == 255)  # 交集
        area2 = np.sum(mask2 == 255)
        correlation = intersection / (area2 + 1e-8)
        return correlation

    def img_resize(self, img, ns, back_color=114):
        # ns: new size: w h
        h, w = img.shape[:2]
        max_r = np.max([w / ns[0], h / ns[1]])
        rw, rh = int(w / max_r), int(h / max_r)

        resized_img = cv2.resize(img, (rw, rh))
        new_img = np.zeros((ns[1], ns[1], 3), dtype=np.uint8)
        new_img[...] = back_color

        padx, pady = int((ns[0] - rw) / 2), int((ns[1] - rh) / 2)
        new_img = dm.general.imgAdd(resized_img, new_img, x=pady, y=padx, alpha=1)

        # cv2.imshow('xx', new_img)
        # cv2.waitKeyEx(0)

        return new_img

    def imshow(self, img, bboxes=None, clss=None, ns=None, show_name='x'):
        if bboxes:
            for j, bbox in enumerate(bboxes):
                if ns is not None:
                    bbox[0] *= ns[0]
                    bbox[1] *= ns[1]
                    bbox[2] *= ns[0]
                    bbox[3] *= ns[1]
                cls = clss[j]
                img = dm.general.plot_one_box_trace_pose_status(
                    x=bbox, img=img, label=cls
                )
        cv2.imshow(show_name, img)
        if cv2.waitKey(0) == ord('q'):
            cv2.destroyAllWindows()
            exit()


def get_opt():
    home = os.environ['HOME']
    dset_rp = f'{home}/datasets/transmission_line'  # dset_rp: dataset root path

    paser = argparse.ArgumentParser()
    paser.add_argument('-dp', '--data_path', type=str,
                       default=f'{dset_rp}/raw0_lr',
                       help='small samples directory which contains imgs and .json labels')
    paser.add_argument('-tp', '--target_path', type=str,
                       default=f'{dset_rp}/raw0_lr_v5_format',
                       help='target directory for store augmented datasets')
    paser.add_argument('-adp', '--additional_data_path', type=str,
                       default=f'{dset_rp}/background',
                       help='additional demo_for_dm.data path which contains additional back ground images')
    paser.add_argument('--suffix', type=str, default='.jpg', help='images suffix')
    paser.add_argument('-noise', '--noise-background', action='store_true',
                       help='use noise backgrounds if True, while use raw image and additional backgrounds if False')
    # paser.add_argument('--force', action='store_true', help='force to overwrite existing tp')
    paser.add_argument('-p', '--policy', type=str, default=None, help='policy for augmentation')
    opt = paser.parse_args()
    return opt


def policy(opt):
    home = os.environ['HOME']
    dset_rp = f'{home}/datasets/transmission_line'  # dset_rp: dataset root path
    pls = dict(
        p1=dict(
            dp=f'{dset_rp}/raw0_lr',
            tp=f'{dset_rp}/raw0_lr_v5_format',
        ),
        p2=dict(
            dp=f'{dset_rp}/raw1_sr',
            tp=f'{dset_rp}/raw1_sr_v5_format',
        ),
        p3=dict(
            dp=f'{dset_rp}/raw1_sr',
            tp=f'{dset_rp}/raw1_sr_v5_format_noise',
            noise_background=True,
        ),
        p4=dict(
            dp=f'{dset_rp}/raw2_daisheng',
            tp=f'{dset_rp}/raw2_daisheng_v5_format',
        ),
    )
    p = pls[opt.policy]

    opt.data_path = p['dp']
    opt.target_path = p['tp']
    opt.noise_background = p.get('noise_background', False)

    return opt


if __name__ == '__main__':
    opt = get_opt()

    if opt.policy:
        opt = policy(opt)

    print(f'opt: {opt}')
    ad = AugmentData(opt)
    ad()
