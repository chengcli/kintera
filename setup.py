#!/usr/bin/env python
import os
import sys
import glob
import platform
from pathlib import Path
from setuptools import setup
from torch.utils import cpp_extension
import sysconfig


def parse_library_names(libdir):
    library_names = []
    for root, _, files in os.walk(libdir):
        for file in files:
            if file.endswith((".a", ".so", ".dylib")):
                file_name = os.path.basename(file)
                library_names.append(file_name[3:].rsplit(".", 1)[0])

    # add system netcdf library
    library_names.extend(['netcdf'])

    # 1) non-cuda libs first (consumers)
    kintera_non_cuda = [l for l in library_names if l.startswith("kintera") and "cuda" not in l]
    # 2) cuda libs last (providers)
    kintera_cuda = [l for l in library_names if l.startswith("kintera") and "cuda" in l]
    # 3) everything else
    other = [l for l in library_names if not l.startswith("kintera")]
    return kintera_non_cuda + other + kintera_cuda


def has_kintera_cuda_library(libdir):
    for root, _, files in os.walk(libdir):
        for file in files:
            if file.startswith("libkintera_cuda") and file.endswith((".a", ".so", ".dylib")):
                return True
    return False


def cuda_root_candidates():
    roots = []
    for name in ("CUDA_HOME", "CUDA_PATH", "CUDAToolkit_ROOT"):
        value = os.environ.get(name)
        if value:
            roots.append(value)
    if platform.system() == "Linux":
        roots.extend(["/usr/local/cuda", "/usr/local/cuda-13.1", "/usr/local/cuda-13"])
    return [root for root in roots if os.path.isdir(root)]

site_dir = sysconfig.get_paths()["purelib"]

current_dir = os.getenv("WORKSPACE", Path().absolute())
build_lib_dir = f"{current_dir}/build/lib"
enable_cuda = has_kintera_cuda_library(build_lib_dir)
include_dirs = [
    f"{current_dir}",
    f"{current_dir}/build",
    f"{current_dir}/build/_deps/fmt-src/include",
    f'{current_dir}/build/_deps/yaml-cpp-src/include',
    f"{site_dir}/pyharp",
]
if enable_cuda:
    for cuda_root in cuda_root_candidates():
        cuda_include = os.path.join(cuda_root, "include")
        if os.path.isdir(cuda_include):
            include_dirs.append(cuda_include)
            break

# add homebrew directories if on MacOS
lib_dirs = [build_lib_dir]
if platform.system() == 'Darwin':
    lib_dirs.extend(['/opt/homebrew/lib'])
else:
    lib_dirs.extend(['/lib64/', '/usr/lib/x86_64-linux-gnu/'])
if enable_cuda:
    for cuda_root in cuda_root_candidates():
        for suffix in ("lib64", "lib"):
            cuda_lib = os.path.join(cuda_root, suffix)
            if os.path.isdir(cuda_lib):
                lib_dirs.append(cuda_lib)
                break
nc_home = os.environ.get("NC_HOME")
if nc_home:
    lib_dirs.append(f"{nc_home}/lib")

libraries = parse_library_names(build_lib_dir)
if enable_cuda:
    for cuda_system_lib in ["cusolver", "cusparse", "cudart"]:
        if cuda_system_lib not in libraries:
            libraries.append(cuda_system_lib)

if sys.platform == "darwin":
    extra_link_args = [
        "-Wl,-rpath,@loader_path/lib",
        "-Wl,-rpath,@loader_path/../torch/lib",
        "-Wl,-rpath,@loader_path/../pydisort/lib",
        "-Wl,-rpath,@loader_path/../pyharp/lib",
    ]
else:
    # ubuntu system has an aggressive linker that removes unused shared libs
    # add cuda library explicitly if built with cuda
    cuda_linker = []
    cuda_libraries = [lib for lib in libraries if "cuda" in lib]
    if cuda_libraries:
        for lib in cuda_libraries:
            libraries.remove(lib)
        cuda_linker = (
            ["-Wl,--no-as-needed"]
            + [f"-l{lib}" for lib in cuda_libraries]
            + ["-Wl,--as-needed"]
            )

    extra_link_args = [
        "-Wl,-rpath,$ORIGIN/lib",
        "-Wl,-rpath,$ORIGIN/../torch/lib",
        "-Wl,-rpath,$ORIGIN/../pydisort/lib",
        "-Wl,-rpath,$ORIGIN/../pyharp/lib"
        ]
    extra_link_args += cuda_linker

ext_module = cpp_extension.CppExtension(
    name='kintera.kintera',
    sources=sorted(glob.glob('python/csrc/*.cpp')),
    include_dirs=include_dirs,
    library_dirs=lib_dirs,
    libraries=libraries,
    extra_compile_args=['-Wno-attributes'],
    extra_link_args=extra_link_args,
    )

setup(
    package_dir={"kintera": "python"},
    ext_modules=[ext_module],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
