from importlib.metadata import PackageNotFoundError, version

import torch
import pydisort
import pyharp

from .kintera import *
from .atm2d import *

try:
    __version__ = version("kintera")
except PackageNotFoundError:
    __version__ = "0.0.0"
