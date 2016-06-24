from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
    name = "Knight app",
    ext_modules = cythonize('knight2.pyx', include_path=[numpy.get_include()]),
)
