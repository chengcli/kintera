import torch
import pydisort
import pyharp
import sysconfig
import ctypes
import os
import platform
from pathlib import Path

from .kintera import *

NODELETE = getattr(os, "RTLD_NODELETE", 0x1000)
MODE = os.RTLD_NOW | os.RTLD_GLOBAL | NODELETE

def load_once(name):
    lib = Path(__file__).parent / "lib" / name
    if platform.system()=="Linux" and lib.exists():
        ctypes.CDLL(str(lib), mode=MODE)

load_once("libkintera_release.so")
load_once("libkintera_cuda_release.so")

__version__ = "0.7.3"
