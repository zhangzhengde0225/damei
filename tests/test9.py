"""
关于dm.nn
"""

import damei as dm

uaii = dm.nn.api.UAII()
print(dm.nn.__doc__)
print(dm.nn.api.__doc__)

print(uaii.__doc__)

print(uaii.ps())

from damei.nn.api.registry import MODULES
