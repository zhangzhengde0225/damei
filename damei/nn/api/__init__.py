"""
damei nn api for ai algorithm management (unified ai interface)
"""

from damei.nn.uaii.uaii_main import UAII
from .base import AbstractModule, AbstractInput, AbstractOutput, AbstractQue
from .registry import MODULES, SCRIPTS, IOS
from .utils import Config
