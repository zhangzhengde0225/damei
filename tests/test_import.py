import os, sys
from pathlib import Path

pydir = Path(os.path.abspath(__file__)).parent
sys.path.append(f'{pydir.parent}')

import damei as dm
