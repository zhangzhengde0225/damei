"""
Damei neural network module
"""
from damei.utils.exception import exception_handler as EXCEPTION

try:
    import torch
except ImportError as e:
    EXCEPTION(ImportError, e, info="You may need to install 'torch' for full functionality")

from . import api
from .modules import *
# from .uaii.utils.config_loader import PyConfigLoader as Config
