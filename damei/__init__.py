__name__ = "damei"

import subprocess

from damei import version

__version__ = version.__version__
__author__ = version.__author__

# dmsystem
from damei.dmsystem.system import current_system, system_lib_suffix

# controls
from damei.controls.color import ColorControl


def colors(num):
	return ColorControl(num).color


# functions
from damei.functions import general, torch_utils
from damei.functions.sub_process import popen

# post
from damei.post import post

# tools
from damei.tools.tools import Tools

# from damei.tools.tools import video2frames
tools = Tools()
from damei.tools.ffmpeg import DmFFMPEG

ffmpeg = DmFFMPEG()

from damei.tools.check_yolo import CheckYOLO

# misc
from damei.misc import misc
from damei.misc.logger import getLogger, get_logger
from damei.misc.time import current_time, plus_time, within_time
from damei.misc.dm_rsa import DmRsa
from damei.misc import fake_argparse as argparse

rsa = DmRsa()
# from damei.misc.config_loader import PyConfig
from damei.misc.scrcpy.scrcpy import Scrcpy

# nn
import damei.nn as nn
from damei.nn.api.utils import Config
