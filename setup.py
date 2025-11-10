import sys

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

# Detect Windows platform
is_windows = sys.platform.startswith("win")

# MSVC flags work for x86, x64 and ARM on Windows
msvc_compile_args = ["/std:c++17", "/EHsc", "/openmp"]
msvc_link_args = []

# GCC/Clang flags for Unix-like systems
unix_compile_args = ["-std=c++17", "-Wall", "-Wextra", "-O3"]
unix_link_args = ["-g"]

if is_windows:
    compile_args = msvc_compile_args
    link_args = msvc_link_args
else:
    compile_args = unix_compile_args.copy()
    link_args = unix_link_args.copy()

    if sys.platform == "darwin":
        compile_args.extend(["-Xpreprocessor", "-fopenmp"])
        link_args.extend(["-lomp", "-fopenmp"])
    else:
        compile_args.append("-fopenmp")
        link_args.append("-fopenmp")

extensions = [
    Extension(
        name="tsop.basic",
        sources=["tsop/basic.pyx"],
        include_dirs=[np.get_include()],
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
