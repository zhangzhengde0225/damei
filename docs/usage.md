damei库各模块使用方法

安装```pip install damei```

导入damei库

```python
import damei as dm
```

## 关于时间

```python
ct = dm.current_time()  # 当前时间，str, 格式是：2020-01-01 00:00:00
et = dm.plus_time(ct=None, seconds=1)  # 未来某个到期时间
bool = dm.within_time(et, ct=None)  # 输入到期时间，判断当前时间是否在内 
```

## misc

```python
import damei as dm

data = dm.misc.rcfile2dict(rcfile)  # 从.rc文件中读取配置，返回字典
# 颜色

# 
dmtools = dm.Tools()
dmtools.check_YOLO(dp='/path/to/yolo_dataset')
```

## dmscp

便捷地实现本都与服务器间的传输。

相比与scp，简化了命令行参数，配置简单。

支持免密传输，支持文件夹传输，支持远程自动创建文件夹。

```
# 配置支持的服务器
mkdir ~/.damei
vi ~/.damei/.dmscprc  # 编辑配置文件, 支持多个服务器，配置如下
"""
siyuan:  # server_name，任意命名
	host: sylogin.hpc.sjtu.edu.cn  # ip地址
	port: 22  # 端口号
	user: xxx  # 用户名
	passwd: xxx  # 密码

default: siyuan  # 多个服务器时，默认使用的服务器
"""

dmscp servers  # 查看已配置的服务器列表
dmscp server_names  # 查看服务器名字
dmscp xx.zip  # 最简单形式，从本地上传文件/文件夹到默认服务器
    # 参数：
    dmscp name # 第一个位置参数指定服务器
    -u xxx # 指定上传的文件/文件夹
    -d xxx # 指定下载的文件/文件夹
    -f # 强制模式，如果目标目录不存在，则在接收端创建
    -r # 递归模式，上传/下载文件夹时使用，递归地上传

```

## tools

```python
from pathlib import Path
import damei as dm

dmtools = dm.Tools()  # 实例化工具

"""1. 检查YOLO格式的数据集"""
dp = f'{Path.home()}/datasets/AID_datasets/real_rsi'  # 指定数据集路径
dmtools.check_YOLO(dp, trte='test', save_dir=None)
# 参数：trte: train or test, 指定检查训练集或测试集，默认为train
#      save_dir: 指定保存的路径，默认为None，绘制后只显示，不保存

```

uaii = dm.nn.api.UAII()

```

API：

```python
import damei as dm
from damei.nn.api.base import AbstractModule, AbstractInput, AbstractOutput, AbstractQue
from damei.nn.api.registry import MODULES, SCRIPTS, IOS
from damei.nn.api.utils import Config
```

```python
uaii = dm.nn.api.UAII()
stream = uaii.get_stream(name='test')
stream_cfg = stream.get_config('/module_name')

ret = stream.run()  # 在主线程中运行，结果直接返回
stream.start()  # 在新线程中运行，结果从队列中获取
ret = stream.train()  # 在主进程运行训练，只有mono_stream才有效
ret = stream.evalute()  # 在主进程运行评估
ret = stream.infer()  # 在主进程运行推理
ret = stream.start_train()  # 在新进程中运行训练
ret = stream.start_evaluate()  # 在新进程中运行评估
ret = stream.start_infer()  # 在新进程中运行推理
```

# data

关于数据处理的程序

```python
import damei as dm

# 动态背景增强
dm.data.dynamic_background_augment()
```
