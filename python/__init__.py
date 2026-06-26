from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import ctypes

import torch
import pydisort
import pyharp
import kintera

from .kintera import *

def _add_packaged_resource_directory() -> None:
    data_dir = Path(kintera.__file__).with_name("data")
    if data_dir.is_dir():
        kintera.add_resource_directory(str(data_dir))


_add_packaged_resource_directory()

from .atm2d import *
from .kinetics_base_titan import *

try:
    __version__ = version("kintera")
except PackageNotFoundError:
    __version__ = "0.0.0"
