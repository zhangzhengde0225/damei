import os
import shutil
import warnings
from pathlib import Path

try:
    import cv2
except:
    pass
# import cv2

from .check_coco import CheckCOCO
from .check_yolo import CheckYOLO


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

    def check_COCO(self, json_path, img_dir=None, line_thicknes=3, only_show_id=None):
        """check coco dataset
        :param json_path, path to coco json file, for example: /xxx/annotations/instances_train2017.json
        :param img_dir, image dir
        :param line_thicknes: thickness for rectangle of bbox
        :param only_show_id: only show the img with this id if is None
        """
        cc = CheckCOCO(json_path, img_dir=img_dir, line_thickness=line_thicknes)
        cc(only_show_id=only_show_id)

    def check_YOLO(self, dp, trte=None, **kwargs):
        """
        check YOLO dataset
        :param dp: yolo dataset dir, for example: /xxx/xxx_yolo_format
        :return: None
        """
        cy = CheckYOLO(dp=dp)
        cy(trte=trte, **kwargs)


def video2frames(video_path, output_path=None, decoder='comm', *args, **kwargs):
    """"""
    output_path = output_path if output_path else f'{Path(video_path).parent}/{Path(video_path).stem}'
    interval = kwargs.pop('interval', 5)
    digits = kwargs.pop('digits', 6)

    # print(output_path)

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    if decoder == 'comm':
        pass
        code = \
            fr'comm -i {video_path} -vf "select=not(mod(n\, {interval}))" ' \
            fr'-y -acodec copy -vsync 0 {output_path}/%0{digits}d.jpg'
        print(f'exec code: {code}')
        os.system(code)
    elif decoder == 'cv2' or decoder == 'opencv':
        # print "正在读取视频：", each_video
        print("正在读取视频：", video_path)  # 我的是Python3.6

        cap = cv2.VideoCapture(video_path)
        frame_index = 0
        frame_count = 0
        if cap.isOpened():
            success = True
        else:
            success = False
            warnings.warn(f'read video: {video_path} failed')

        while (success):
            success, frame = cap.read()
            # print "---> 正在读取第%d帧:" % frame_index, success
            print(f"\r读取第{frame_index}帧, 保存第{frame_count}帧: {success}", end='')

            if frame_index % interval == 0 and success:  # 如路径下有多个视频文件时视频最后一帧报错因此条件语句中加and success
                # resize_frame = cv2.resize(frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
                # cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % frame_index, resize_frame)
                img_save_path = f"{output_path}/{frame_count:0>{digits}}.jpg"
                # cv2.imwrite(each_video_save_full_path + "%d.jpg" % frame_count, resize_frame)
                cv2.imwrite(img_save_path, frame)
                frame_count += 1
            frame_index += 1

        cap.release()  # 这行要缩一下、原博客会报错
    else:
        raise NotImplementedError(f'decoder: {decoder}')
