__name__ = "damei"

from damei import version

__version__ = version.__version__

from damei.functions import general, torch_utils
from damei.controls.color import ColorControl
from damei.post import post
# from damei.network import a
