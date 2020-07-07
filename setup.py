r"""
Packaging setup file for Pypi and installation via pip.

Notes
-----
#.  Written by David C. Stauffer in August 2017.
#.  Currently this package is not available on pip, but it is designed such that it could be if it
    is ever so desired.
"""

#%% Import
import os

from setuptools import setup

#%% Support functions
def readme():
    r"""Opens the README.rst file for additional descriptions."""
    filename = os.path.join(os.path.dirname(__file__), 'README.rst')
    with open(filename) as file:
        return file.read()

#%% Setup
setup(
    name='dstauffman',
    version='2.0',
    description='Generic python utilities for aerospace and health policy applications',
    long_description=readme(),
    keywords='dstauffman numerical analysis plotting quaternions statistics batch estimation',
    url='https://github.com/dstauffman/dstauffman',
    author='David C. Stauffer',
    author_email='dstauffman@yahoo.com',
    license='LGPLv3',
    packages=['dstauffman'],
    install_requires=[
        'h5py',
        'matplotlib',
        'numpy',
        'PyQt5',
        'scipy',
    ],
    python_requires='>=3.7',
    include_package_data=True,
    zip_safe=False)
