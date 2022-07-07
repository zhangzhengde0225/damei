import os
import cv2
from .ffmpeg import DmFFMPEG

dmffmpeg = DmFFMPEG()


def push_stream(
        source,
        ip='127.0.0.1',
        port=1935,
        stream_type='rtmp',
        key=None,
        vcodec='h264',
        acodec=None,
        suffix='.jpg',
        **kwargs, ):
    """
    Push stream via DmFFMPEG based on FFMPEG.
    Note: You need configure Nginx to serve the stream.

    :param source: The source needs to be pushed. It could be single image (numpy.array), folder contains images, stream (rtsp or rtmp) or video (.mp4, .avi etc.), .
    :type source: str
    :param ip: The ip of Nginx. Default is "127.0.0.1".
    :type ip: str
    :param port: The port of Nginx. Default is 1935.
    :type port: int
    :param stream_type: The stream type, "rtmp" or "rtsp". Default is "rtmp".
    :type stream_type: str
    :param key: The key of the stream, which is the additional chars in url. Default is None.
    :type key: str
    :param vcodec: The video codec for comm. Default is "h264".
    :type vcodec: str
    :param acodec: The audio codec for comm. Default is None.
    :type acodec: str
    :param suffix: The suffix of the image, valid only when source is a folder. Default is ".jpg".
    :type suffix: str
    :param kwargs: Other parameters for comm.
    :return: None, The stream will be pushed to Nginx server in url: stream_type://ip:port/live/key, i.e. rtmp://127.0.0.1:1935/live.

    Example1: Push stream from video:
        >>> import damei as dm
        >>> video_path = "your video path"
        >>> dm.comm.push_stream(source=video_path)

    Eaxmple2: Push stream from images (np.array):
        >>> import damei as dm
        >>> import cv2
        >>> img_files = ["your image path1", "your image path2"]
        >>> for i, img_file in enumerate(img_files):
        >>>     img = cv2.imread(img_file)  # read image from file, and convert to numpy.array
        >>>     dm.comm.push_stream(img)

    Display:
        >>> $ ffplay "rtmp://127.0.0.1:1935/live"  # to show the stream
    """

    if isinstance(source, str) and os.path.isdir(source):
        img_files = [f'{source}/{x}' for x in os.listdir(source) if x.endswith(suffix)]
        for i, img_file in enumerate(img_files):
            assert os.path.exists(img_file), f'Could Not Found Image: {img_file}'
            img = cv2.imread(img_file)
            dmffmpeg.push_stream(
                source=img,
                ip=ip,
                port=port,
                stream_type=stream_type,
                key=key,
                vcodec=vcodec,
                acodec=acodec,
                **kwargs,
            )

    else:
        dmffmpeg.push_stream(
            source=source,
            ip=ip,
            port=port,
            stream_type=stream_type,
            key=key,
            vcodec=vcodec,
            acodec=acodec,
            **kwargs,
        )
