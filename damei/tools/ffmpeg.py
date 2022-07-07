"""
利用ffmpeg推流
"""
import os
import subprocess

import numpy as np

from damei.misc.logger import getLogger

logger = getLogger('DmFFMPEG')


class DmFFMPEG(object):
    def __init__(
            self,
            ip='127.0.0.1',
            port='1935',
            stream_type='rtmp',
            key=None,
            vcodec='h264',
            acodec=None,
            **kwargs
    ):
        self.ip = ip
        self.port = port
        self.stream_type = stream_type
        self.key = key
        self.vcodec = vcodec
        self.acodec = acodec
        self._pipe = None
        self.mute = kwargs.pop('mute', False)  #

    def init_pipe(self, img, **kwargs):
        """初始化管道，第一次调用push_stream时且传入为图像时，调用"""
        ip = kwargs.pop('ip', self.ip)
        port = kwargs.pop('port', self.port)
        stream_type = kwargs.pop('stream_type', self.stream_type)
        key = kwargs.pop('key', self.key)
        vcodec = kwargs.pop('vcodec', self.vcodec)
        acodec = kwargs.pop('acodec', self.acodec)
        fps = kwargs.pop('fps', 15)

        key = f'/{key}' if key else ''
        vcodec = f'{vcodec}' if vcodec else ''
        acodec = f'{acodec}' if acodec else ''

        url = f'{stream_type}://{ip}:{port}/live{key}'
        h, w, c = img.shape
        command = [
            'comm',
            # 're',#
            # '-y', # 无需询问即可覆盖输出文件
            '-f', 'rawvideo',  # 强制输入或输出文件格式
            # '-vcodec','rawvideo', # 设置视频编解码器。这是-codec:v的别名
            '-pix_fmt', 'bgr24',  # 设置像素格式
            '-s', f'{w}x{h}',  # 设置图像大小
            '-r', f'{fps}',  # 设置帧率
            '-i', '-',  # 输入
            '-vcodec', vcodec,
            '-acodec', acodec,
            # '-pix_fmt', 'yuv420p',
            # '-preset', 'ultrafast',
            '-f', 'flv',  # 强制输入或输出文件格式
            url]
        # print(command)
        # exit()
        logger.info(f'The stream will be pushed to: {url}.')

        if self.mute:
            pipe = subprocess.Popen(
                command, stdin=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdout=subprocess.PIPE)
        else:
            pipe = subprocess.Popen(
                command, stdin=subprocess.PIPE,
            )
        # exit()
        self._pipe = pipe
        return pipe

    def get_pipe(self, img=None, **kwargs):
        if self._pipe is not None:
            return self._pipe
        else:
            return self.init_pipe(img=img, **kwargs)

    # return self._pipe

    def push_stream(self, source, **kwargs):
        """
        推流
        :param source: 可以是rtsp rtmp流或.mp4 .avi等视频文件
                        也可以是单张图，必须是np.ndarray格式
        :param kwargs:
        :return:
        """
        ip = kwargs.pop('ip', self.ip)
        port = kwargs.pop('port', self.port)
        stream_type = kwargs.pop('stream_type', self.stream_type)
        key = kwargs.pop('key', self.stream_type)
        vcodec = kwargs.pop('vcodec', self.vcodec)
        acodec = kwargs.pop('acodec', self.acodec)
        key = f'/{key}' if key else ''
        vcodec = f'-vcodec {vcodec}' if vcodec else ''
        acodec = f'-acodec {acodec}' if acodec else ''

        # 推视频流
        if isinstance(source, str):
            code = f'comm -i {source} {vcodec} {acodec} -f flv {stream_type}://{ip}:{port}/live{key}'
            if self.mute:
                subprocess.Popen(code, stdout=subprocess.STDOUT, stderr=subprocess.STDOUT)
            else:
                os.system(code)
        elif isinstance(source, np.ndarray):
            # 第一次调用时检查是否有输出pipe
            pipe = self.get_pipe(source, **kwargs)
            pipe.stdin.write(source.tobytes())
        else:
            raise NotImplementedError(f'Source type error: {source} {type(source)}.')

    def show_stream(self, source=None, stream_type='rtmp', ip='127.0.0.1', port='1935', key=None):
        key = f'/{key}' if key else ''
        source = source if source else f'{stream_type}://{ip}:{port}/live{key}'
        code = f'ffplay {source}'
        os.system(code)
