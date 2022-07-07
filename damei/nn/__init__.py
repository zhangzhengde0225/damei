"""
Damei neural network module
"""
# import os, sys
from damei.utils.exception import exception_handler as EXCEPTION

try:
    import torch
except ImportError as e:
    EXCEPTION(ImportError, e, info="You may need to install 'torch' for full functionality")

from . import api
from .modules import *
# from .uaii.utils.config_loader import PyConfigLoader as Config

from .api import AbstractModule, AbstractInput, AbstractOutput, AbstractQue
from .api import MODULES, SCRIPTS, IOS
from .api import Config
from .api import UAII
from .api import test_module, test_script, test_io

"""隐式地导入当前目录下的dmapi和repos下的模块"""
dm_path = ['dmapi', 'repos']
dirs = [x for x in os.listdir('.') if os.path.isdir(x)]  # 当前路径文件夹
for dmp in dm_path:
    if dmp in dirs:
        # if dmp == 'repos':
        # code = f'from repos.xxx.dmapi import *'
        code = f'from {dmp} import *'
        exec(code)
