__name__ = "damei"

from importlib import import_module

from damei import version
from damei.controls.color import ColorControl
from damei.dmsystem.system import current_system, system_lib_suffix
from damei.functions.sub_process import popen
from damei.misc import fake_argparse as argparse
from damei.misc.dm_rsa import DmRsa
from damei.misc.logger import getLogger, get_logger
from damei.misc.time import current_time, plus_time, within_time
from damei.tools.tools import Tools
from damei.utils.exception import exception_handler as EXCEPTION

__version__ = version.__version__
__author__ = version.__author__
DATA_ROOT = version.DATA_ROOT


def colors(num):
    return ColorControl(num).color


class _LazyObject(object):
    def __init__(self, factory):
        self._factory = factory
        self._instance = None

    def _load(self):
        if self._instance is None:
            self._instance = self._factory()
        return self._instance

    def __getattr__(self, item):
        return getattr(self._load(), item)

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)

    def __repr__(self):
        if self._instance is None:
            return "<lazy damei object>"
        return repr(self._instance)


class _LazyModule(object):
    def __init__(self, module_name):
        self._module_name = module_name
        self._module = None

    def _load(self):
        if self._module is None:
            self._module = import_module(self._module_name)
        return self._module

    def __getattr__(self, item):
        return getattr(self._load(), item)

    def __repr__(self):
        if self._module is None:
            return f"<lazy module {self._module_name}>"
        return repr(self._module)


class _LazyAttr(object):
    def __init__(self, module_name, attr_name):
        self._module_name = module_name
        self._attr_name = attr_name
        self._value = None

    def _load(self):
        if self._value is None:
            self._value = getattr(import_module(self._module_name), self._attr_name)
        return self._value

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._load(), item)

    def __repr__(self):
        if self._value is None:
            return f"<lazy attribute {self._module_name}.{self._attr_name}>"
        return repr(self._value)

tools = Tools()
rsa = DmRsa()

general = _LazyModule("damei.functions.general")
torch_utils = _LazyModule("damei.functions.torch_utils")
nn = _LazyModule("damei.nn")
data = _LazyModule("damei.data")
comm = _LazyModule("damei.comm")
misc = _LazyModule("damei.misc")

post = _LazyAttr("damei.post.post", "post")
CheckYOLO = _LazyAttr("damei.tools.check_yolo", "CheckYOLO")
Scrcpy = _LazyAttr("damei.misc.scrcpy.scrcpy", "Scrcpy")
Config = _LazyAttr("damei.nn.api.utils", "Config")


__all__ = [
    "__version__",
    "__author__",
    "DATA_ROOT",
    "EXCEPTION",
    "current_system",
    "system_lib_suffix",
    "ColorControl",
    "colors",
    "popen",
    "getLogger",
    "get_logger",
    "current_time",
    "plus_time",
    "within_time",
    "DmRsa",
    "rsa",
    "argparse",
    "tools",
    "general",
    "torch_utils",
    "post",
    "Tools",
    "CheckYOLO",
    "Scrcpy",
    "nn",
    "Config",
    "data",
    "comm",
    "misc",
]
