from .augment_dataset import AugmentData
from .multiscale_slice import MultiScaleSlice, AugmentMSS


def augment(
        source_path: str,
        target_path: str = None,
        backgrounds_path: str = None,
        anno_fmt: str = 'labelme',
        out_fmt: str = 'YOLOfmt',
        use_noise_background: bool = False,
        out_size: int or tuple = 640,
        num_augment_images: int = 8000,
        train_ratio: float = 0.8,
        erase_ratio: float or tuple = (0, 1),
        mean_scale_factor: float = 1,
        save_mask: bool = False,
        suffix: str = '.jpg',
        *args,
        **kwargs, ):
    """
    A script to augment dataset from small annotated dataset. The target is random position, random size and random angle, and the background is dynamic and random.

    :param source_path: Annotated dataset path.
    :type source_path: str
    :param target_path: Path to save augmented dataset. Default is the same directory of 'source_path', e.g. xxx_augmented_{out_fmt}.
    :type target_path: str
    :param backgrounds_path: Path to background images. Default is None, which means use background source images.
    :type backgrounds_path: str
    :param anno_fmt: The labeling software used in annotation. Default is 'labelme'.
    :type anno_fmt: str
    :param out_fmt: The output format of augmented dataset. Default is 'YOLOfmt'.
    :type out_fmt: str
    :param use_noise_background: Whether to generate an image using a noise background. Default is False.
    :type use_noise_background: bool
    :param out_size: The size of the output images. Default is 640.
    :type out_size: int or tuple
    :param num_augment_images: The number of augmented images. Default: 8000.
    :type num_augment_images: int
    :param train_ratio: The ratio of training dataset. Default: 0.8.
    :type train_ratio: float
    :param samples_ratio_per_class: The ratio of per class for samples equilibrium. Default: [1] * len(classes).
    :type samples_ratio_per_class: list
    :param erase_ratio: The ratio of target need to be erased (replaced by gray color) when using annotated image as background. Default: 0.0, which means no erase.
    :type erase_ratio: float or tuple
    :param mean_scale_factor: The mean scale factor of gaussian distribustion, decide the num of targets in synthetic image. Default: 1.
    :type mean_scale_factor: float
    :param save_mask: Whether to save the mask of each target. Default is False. If True, If true, each mask will be saved as a binary mask in .npy format (w, h, 1), where 0 represents the background and 1 represents the target.
    :type save_mask: bool
    :param suffix: The raw image suffix. Default is .jpg
    :type suffix: str
    :return: None

    Example:
        >>> import damei as dm
        >>> dm.data.augment("/path/to/dataset")
    """
    ad = AugmentData(
        dp=source_path,
        tp=target_path,
        adp=backgrounds_path,
        anno_fmt=anno_fmt,
        out_fmt=out_fmt,
        use_noise_background=use_noise_background,
        out_size=out_size,
        save_mask=save_mask,
        suffix=suffix,
        *args,
        **kwargs
    )
    ad(num_augment_images=num_augment_images,
       train_ratio=train_ratio,
       erase_ratio=erase_ratio,
       mean_scale_factor=mean_scale_factor,
       *args, **kwargs)


def augment_mss(
        source_path: any,
        target_path: str = None,
        min_wh: int = 640,
        max_wh: int = None,
        stride_ratio: float = 0.5,
        pss_factor: float = 1.25,
        need_annotation: bool = True,
        anno_fmt: str = 'json',

):
    """
    基于滑动窗口的多级尺寸切片算法，multi scale slice algorithm based on slide window, mss.
    输入一张图、文件夹或多张图，对每一张图，利用滑动窗口动态匹配方法，输出从较小尺寸（例如：640）到原始尺寸的多级尺寸切片

    :param source_path: 图片路径或图片文件夹路径
    :param target_path: 输出文件夹路径
    :param min_wh: 最小尺寸，默认640
    :param max_wh: 最大尺寸，默认None时，从最小尺寸自动递增至原图尺寸
    :param stride_ratio: 滑动窗口每次移动的比例，默认0.5
    :param pss_factor: 等比数列比例因子(proportional series scale factor)，默认1.25，控制窗口尺寸的增长
    :param need_annotation: 是否需要标注，默认True，标注需与图像在同一路径下
    :param anno_fmt: 标注文件格式，默认json

    :return: None, 在输出文件夹路径下生成增强的图像

    Example:
        >>> dm.data.augment_mss("/path/to/img(fold) or imgs list", other_params)
    """
    mss = AugmentMSS(
        sp=source_path,
        tp=target_path,

    )
    mss(min_wh=min_wh,
        max_wh=max_wh,
        stride_ratio=stride_ratio,
        pss_factor=pss_factor,
        need_annotation=need_annotation,
        anno_fmt=anno_fmt,
        )
