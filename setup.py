r"""
Packaging setup file for Pypi and installation via pip.

Notes
-----
#.  Written by David C. Stauffer in August 2017.
#.  Currently this package is not available on pip, but it is designed such that it could be if it
    is ever so desired.
"""

# %% Import
from pathlib import Path

from setuptools import setup  # type: ignore[import]


# %% Support functions - readme
def readme():
    r"""Opens the README.rst file for additional descriptions."""
    filename = Path(__file__).resolve().parent / "README.rst"
    with open(filename) as file:
        return file.read()


# %% Support functions - get_version
def get_version():
    r"""Reads the version information from the library."""
    filename = Path(__file__).resolve().parent.joinpath("dstauffman", "version.py")
    with open(filename) as file:
        text = file.read()
    for line in text.splitlines():
        if line.startswith("version_info = "):
            return line.split("(")[1].split(")")[0].replace(", ", ".")
    raise RuntimeError("Unable to load version information.")


# %% Setup
setup(
    name="dstauffman",
    version=get_version(),
    description="Generic python utilities for aerospace and health policy applications",
    long_description=readme(),
    keywords="dstauffman numerical analysis plotting quaternions statistics batch estimation",
    url="https://github.com/dstauffman/dstauffman",
    author="David C. Stauffer",
    author_email="dstauffman@yahoo.com",
    license="LGPLv3",
    packages=["dstauffman"],
    package_data={"dstauffman": ["py.typed"]},
    install_requires=[
        "h5py>=3.7",
        "matplotlib>=3.7",
        "numba>=0.59",
        "numpy>=2.0",
        "pandas>=2.0",
        "pytest>=7.4",
        "qtpy>=2.2",  # wraps any of PyQt5, PyQt6, PySide2, or PySide6
        "scipy>=1.10",
        "tblib>=1.7",
    ],
    extra_require={"shading": ["datashader>=0.15"]},
    python_requires=">=3.10",
    include_package_data=True,
    zip_safe=False,
)
