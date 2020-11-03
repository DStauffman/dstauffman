r"""
Define constants for use in the rest of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
import logging
import os
import unittest
import warnings

HAVE_NUMPY: bool
HAVE_H5PY: bool

# optional dependencies
try:
    import coverage
    assert coverage  # not really used, but it silences the warninngs
    HAVE_COVERAGE = True
except ModuleNotFoundError:
    HAVE_COVERAGE = False
try:
    import h5py
    assert h5py  # not really used, but it silences the warninngs
    HAVE_H5PY = True
except ModuleNotFoundError:
    HAVE_H5PY = False
try:
    import matplotlib
    assert matplotlib  # not really used, but it silences the warninngs
    HAVE_MPL = True
except ModuleNotFoundError:
    HAVE_MPL = False
try:
    import numpy as np
    HAVE_NUMPY = True
except ModuleNotFoundError:
    HAVE_NUMPY = False
try:
    import pandas as pd
    assert pd
    HAVE_PANDAS = True
except ModuleNotFoundError:
    HAVE_PANDAS = False
try:
    import pytest
    assert pytest
    HAVE_PYTEST = True
except ModuleNotFoundError:
    HAVE_PYTEST = False
try:
    import scipy
    assert scipy
    HAVE_SCIPY = True
except ModuleNotFoundError:
    HAVE_SCIPY = False

from dstauffman.enums import LogLevel

#%% Optional settings
if HAVE_NUMPY:
    # Set NumPy error state for module
    np.seterr(invalid='raise', divide='raise')
    # Set NumPy printing options
    np.set_printoptions(threshold=1000)
else:
    warnings.warn('numpy was not imported, so a lot of capabilities will be limited.')
if not HAVE_H5PY:
    warnings.warn('h5py was not imported, so some file save and load capabilities will be limited.')

#%% Register custom logging levels
logging.addLevelName(LogLevel.L0, 'L0')
logging.addLevelName(LogLevel.L1, 'L1')
logging.addLevelName(LogLevel.L2, 'L2')
logging.addLevelName(LogLevel.L3, 'L3')
logging.addLevelName(LogLevel.L4, 'L4')
logging.addLevelName(LogLevel.L5, 'L5')
logging.addLevelName(LogLevel.L6, 'L6')
logging.addLevelName(LogLevel.L7, 'L7')
logging.addLevelName(LogLevel.L8, 'L8')
logging.addLevelName(LogLevel.L9, 'L9')
logging.addLevelName(LogLevel.L10, 'L10')
logging.addLevelName(LogLevel.L11, 'L11')
logging.addLevelName(LogLevel.L12, 'L12')
logging.addLevelName(LogLevel.L20, 'L20')

#%% Configure default logging if not already set
logging.basicConfig(level=logging.WARNING)

#%% Constants
# A specified integer token value to use when you need one
INT_TOKEN: int = -1

# Whether we are currently on Windows or not
IS_WINDOWS: bool = os.name == 'nt'

# Interval units for numpy dates and times
NP_DATETIME_UNITS: str = 'ns'

# Interval form to use for storing data as np.datetime64's
NP_DATETIME_FORM: str = 'datetime64[ns]'

# Interval form to use for storing data as np.timedelta64's
NP_TIMEDELTA_FORM: str = 'timedelta64[ns]'

# Scale factor for converting numpy time forms to seconds (nanoseconds per second)
NP_INT64_PER_SEC: int = 10**9

#%% Functions
# None

#%% Script
if __name__ == '__main__':
    # print all the constants (anything in ALL_CAPS)
    fields = [k for k in sorted(dir()) if k.isupper()]
    max_len = max(map(len, fields))
    for field in fields:
        pad = ' ' * (max_len - len(field))
        value = locals()[field]
        print('{}{} = {}'.format(field, pad, value))

    # run unittests
    unittest.main(module='dstauffman.tests.test_constants', exit=False)
