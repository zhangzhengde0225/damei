from .augment_dataset import AugmentData


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
    :return: None

    Example:
        >>> import damei as dm
        >>> dm.demo_for_dm.data.augment("/path/to/dataset")
    """
    ad = AugmentData(
        dp=source_path,
        tp=target_path,
        adp=backgrounds_path,
        anno_fmt=anno_fmt,
        out_fmt=out_fmt,
        use_noise_background=use_noise_background,
        out_size=out_size,
        *args,
        **kwargs
    )
    ad(num_augment_images=num_augment_images,
       train_ratio=train_ratio,
       erase_ratio=erase_ratio,
       mean_scale_factor=mean_scale_factor,
       *args, **kwargs)
