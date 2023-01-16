damei.comm
=============
通信相关函数

Communication (comm)


damei.comm.push_stream
-------------------------

使用damei库推流，支持在for循环中不断推流、单张图像、图像文件夹和视频推流。

基于ffmpeg和Nginx的推流。配置Nginx，请参阅；`搭建流媒体服务器 <https://www.cnblogs.com/zhumengke/articles/11206794.html>`_

A push stream based on ffmpeg and Nginx.

+ push stream from numpy.array
+ push stream from video
+ push stream from imgs folder


.. autofunction:: damei.comm.push_stream
