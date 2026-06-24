"""
Build the cerebellum C++ extension in-place:

    python setup.py build_ext --inplace

Produces cerebellum.cpython-310-...so (or .pyd on Windows)
in the same directory, importable as:

    from cerebellum import Cerebellum
"""

from setuptools import setup, Extension
import pybind11

ext = Extension(
    name="cerebellum",
    sources=["cerebellum.cpp"],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=["-std=c++14", "-O2"],
)

setup(
    name="cerebellum",
    ext_modules=[ext],
)
