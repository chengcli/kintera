from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import pydisort
import pyharp

from .kintera import *
from .atm2d import *

try:
    __version__ = version("kintera")
except PackageNotFoundError:
    __version__ = "0.0.0"

add_resource_directory(str(Path(__file__).parent / "data"), prepend=False)
