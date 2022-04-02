"""
测试dm.ffmpeg模块
"""
import os, sys

sys.path.append(os.path.abspath('..'))
import damei as dm
import cv2
import numpy as np


def video2stream():
	video_path = f'{os.environ["HOME"]}/datasets/jiangwu/show_demo/video_for_demo.mp4'

	dm.ffmpeg.push_stream(
		source=video_path,
		ip='192.168.3.111',
		port=1935,
		stream_type='rtmp',
		key=None,
		vcodec='h264',
	)


def frame2stream():
	video_path = f'{os.environ["HOME"]}/datasets/jiangwu/show_demo/video_for_demo.mp4'
	cap = cv2.VideoCapture(video_path)
	dmpeg = dm.DmFFMPEG(
		ip='192.168.3.111', port=1935, mute=True,
	)

	i = 0
	while cap.isOpened():
		ret, frame = cap.read()
		# print(f'ret: {ret} frame: {frame.shape} {type(frame)}')
		if not ret:
			print(f'{i} done!!!')
			break
		dmpeg.push_stream(frame)
		i += 1


# exit()

# video2stream()
frame2stream()
