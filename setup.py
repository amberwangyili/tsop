import os
import platform
import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# Detect Windows platform
is_windows = sys.platform.startswith("win")

# MSVC flags work for x86, x64 and ARM on Windows
msvc_compile_args = ["/std:c++17", "/EHsc"]

# GCC/Clang flags for Unix-like systems (with OpenMP)
if sys.platform == "darwin":
    # macOS: find libomp from Homebrew
    if platform.machine() == "arm64":
        omp_prefix = "/opt/homebrew/opt/libomp"
    else:
        omp_prefix = "/usr/local/opt/libomp"
    unix_compile_args = [
        "-std=c++17", "-Wall", "-Wextra", "-O3",
        "-Xpreprocessor", "-fopenmp",
        f"-I{omp_prefix}/include",
    ]
    unix_link_args = ["-g", f"-L{omp_prefix}/lib", "-lomp"]
else:
    unix_compile_args = ["-std=c++17", "-Wall", "-Wextra", "-O3", "-fopenmp"]
    unix_link_args = ["-g", "-lgomp"]

compile_args = msvc_compile_args if is_windows else unix_compile_args
link_args = ["-g"] if is_windows else unix_link_args

include_dirs = [np.get_include()]

extensions = [
    Extension(
        name="tsop.basic",
        sources=["tsop/basic.pyx"],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )
]

setup(
    name="tsop",
    version="0.1.0",
    description="Cython/C++ operator library for 2D financial time series",
    author="Yili",
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "cython",
    ],
    packages=["tsop"],
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
    zip_safe=False,
)
