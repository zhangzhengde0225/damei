import ctypes
import cv2
import numpy as np
import torch

from trt import trt_wrapper

# ctypes.CDLL('/home/zy/trt_engine/384640_bs1/libmyplugins.so')
plugins_path = '/home/zy/trt_engine/384640_bs1/libmyplugins.so'
# plugins_path = '/home/zy/trt_engine/384640_bs8/libmyplugins.so'
engine_path = '/home/zy/trt_engine/384640_bs1/yolov5x.engine' # max_batch_1
# engine_path = '/home/zy/trt_engine/384640_bs8/yolov5x.engine'
np_source3 = np.random.randint(low=0, high=255, size=(3, 384, 640))
np_source4 = np.random.randint(low=0, high=255, size=(8, 3, 384, 640))

trt_wrapper = trt_wrapper(engine_file_path=engine_path, plugins_path=plugins_path, ori_imgsize=(1080, 1920))
output = trt_wrapper.infer(np_source3, post_proc=True)
print('end')