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

tools = Tools()
from damei.tools.tools import video2frames

from damei.tools.check_yolo import CheckYOLO

# misc
from damei.misc import misc
