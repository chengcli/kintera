from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import ctypes

import torch
import pydisort
import pyharp

_native_lib = Path(__file__).with_name("lib") / "libkintera_release.dylib"
if _native_lib.exists():
    ctypes.CDLL(str(_native_lib), mode=ctypes.RTLD_GLOBAL)

from .kintera import *
from .atm2d import *
from .kinetics_base_titan import *

try:
    __version__ = version("kintera")
except PackageNotFoundError:
    __version__ = "0.0.0"
