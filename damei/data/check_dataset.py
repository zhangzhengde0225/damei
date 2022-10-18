from damei.tools.check_coco import CheckCOCO
from damei.tools.check_yolo import CheckYOLO


def check_COCO(json_path, img_dir=None, line_thicknes=3, only_show_id=None, **kwargs):
    """
    Check the COCO dataset

    :param json_path, path to coco json file, for example: /xxx/annotations/instances_train2017.json
    :type json_path: str

    :param img_dir, image dir
    :type img_dir: str

    :param line_thicknes: thickness for rectangle of bbox
    :type line_thicknes: int

    :param only_show_id: only show the img with this id if is None
    :type only_show_id: int

    :return: None
    """
    cc = CheckCOCO(json_path, img_dir=img_dir, line_thickness=line_thicknes)
    cc(only_show_id=only_show_id)


def check_YOLO(sp, trte=None, save_dir=None, **kwargs):
    """
    Check the YOLO dataset

    :param sp: The source path to YOLO format dataset, e.g: /xxx/xxx_yolo_format.
    :type sp: str
    :param trte: Which type to check, e.g: 'train, 'test' or 'val'. Default is train
    :type trte: str
    :param save_dir: The path to save the plotte image. Default is None, which means not save.
    :type save_dir: str
    :param classes: The classes of the dataset. Default is None, which means the classes is in the file 'classes.txt' in the source path.
    :type classes: list


    :return: None
    """
    cy = CheckYOLO(dp=sp)
    cy(trte=trte,
       save_dir=save_dir,
       **kwargs)
