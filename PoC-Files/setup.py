import sys

from setuptools import Extension,setup
from Cython.Build import cythonize

ext = Extension("QSv3_049",["QSv3_049.pyx"],include_dirs=sys.path,libraries=['gmp','mpfr','mpc'])

ext.cython_directives={'language_level':"3"}

setup(name="QSv3_049",ext_modules=cythonize([ext],include_path=sys.path))
