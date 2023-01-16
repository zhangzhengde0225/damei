"""
Damei neural network module
"""
import os, sys
from damei.utils.exception import exception_handler as EXCEPTION
from pathlib import Path

pydir = Path(os.path.abspath(__file__)).parent

try:
    import torch
except ImportError as e:
    pass
    # EXCEPTION(ImportError, e, info="You may need to install 'torch' for full functionality")

from . import api
# from .uaii.utils.config_loader import PyConfigLoader as Config

from .api import AbstractModule, AbstractInput, AbstractOutput, AbstractQue
from .api import MODULES, SCRIPTS, IOS, init_register
from .api import Config
from .api import UAII
from .api import test_module, test_script, test_io

# 导入内部模块和外部模块
internal_modules = [
#     'modules/loader/vis_loader',
#     'modules/exporter/vis_exporter',
]
# external_folders = ['dmapi', 'repos']
external_folders = []

init_register(
    internal_modules=internal_modules,
    external_folders=external_folders
)
