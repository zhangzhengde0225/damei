"""
调用dm.ffmpeg把单张图或者单个视频文件推成流
"""
import os, sys
from pathlib import Path

pydir = Path(os.path.abspath(__file__)).parent
try:
    import damei as dm
except:
    sys.path.append(str(pydir.parent.parent))
    import damei as dm
import numpy as np
import cv2
import os


def push_video2stream():
    video_path = f'{dm.DATA_ROOT}/dm.ffmpeg/demo.mp4'
    dm.ffmpeg.push_stream(
        source=video_path
    )


def push_img2stream():
    """
    单张图推流
    """
    imgs_dir = f'{dm.DATA_ROOT}/dm.ffmpeg/imgs'
    imgs = [f'{imgs_dir}/{x}' for x in os.listdir(imgs_dir)]
    # print(len(imgs), imgs)
    for i, img_file in enumerate(imgs):
        assert os.path.exists(img_file), f'Could Not Found Image: {img_file}'
        img = cv2.imread(img_file)
        print(f'\rPush image: [{i:>3}/{len(imgs)}] {img.shape} {type(img)} {img.dtype}', end='')
        dm.ffmpeg.push_stream(
            source=img,
        )
    print('')
    sys.exit(0)


if __name__ == '__main__':
    # push_video2stream()
    push_img2stream()
