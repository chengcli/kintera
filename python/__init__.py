import torch
import pydisort
import pyharp
import sysconfig
import ctypes
import os
import platform
from pathlib import Path

from .kintera import *

# link cuda library if on linux system
p = Path(__file__).parent / "lib" / "libkintera_cuda_release.so"
if platform.system() == "Linux" and p.exists():
    site_dir = sysconfig.get_paths()["purelib"]

    lib_path = f"{site_dir}/kintera/lib/libkintera_release.so"
    ctypes.CDLL(lib_path, mode=os.RTLD_NOW | os.RTLD_GLOBAL)

    lib_path = f"{site_dir}/kintera/lib/libkintera_cuda_release.so"
    ctypes.CDLL(lib_path, mode=os.RTLD_NOW | os.RTLD_GLOBAL)

__version__ = "0.7.3"
