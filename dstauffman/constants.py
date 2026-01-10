r"""
Define constants for use in the rest of the "dstauffman" library.

Notes
-----
#.  Written by David C. Stauffer in March 2015.

"""

# %% Imports
from __future__ import annotations

import os
from typing import Final
import unittest
import warnings

HAVE_H5PY: bool
HAVE_NUMPY: bool

# %% Set flags for optional dependencies
try:
    import coverage

    assert coverage  # not really used, but it silences the warnings
    HAVE_COVERAGE = True
except ModuleNotFoundError:
    HAVE_COVERAGE = False
try:
    import h5py

    assert h5py  # not really used, but it silences the warnings
    HAVE_H5PY = True
except ModuleNotFoundError:
    HAVE_H5PY = False
try:
    import matplotlib  # noqa: ICN001

    assert matplotlib  # not really used, but it silences the warnings
    HAVE_MPL = True
except ModuleNotFoundError:
    HAVE_MPL = False
if HAVE_MPL:
    try:
        import datashader

        assert datashader
        HAVE_DS = True
    except ModuleNotFoundError:
        HAVE_DS = False
    except OSError:
        HAVE_DS = False
else:
    HAVE_DS = False
try:
    import numpy as np

    HAVE_NUMPY = True
    try:
        import keras

        keras.backend.set_floatx("float64")  # TODO: this doesn't seem to work!!!
        HAVE_KERAS = True
    except ModuleNotFoundError:
        HAVE_KERAS = False
except ModuleNotFoundError:
    HAVE_NUMPY = False
    HAVE_KERAS = False
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

# %% Optional settings
if HAVE_NUMPY:
    # Set NumPy error state for module
    np.seterr(invalid="raise", divide="raise")
    # Set NumPy printing options
    np.set_printoptions(threshold=20, precision=17)
else:
    warnings.warn("numpy was not imported, so a lot of capabilities will be limited.")
if not HAVE_H5PY:
    warnings.warn("h5py was not imported, so some file save and load capabilities will be limited.")

# %% Constants
# A specified integer token value to use when you need one
INT_TOKEN: Final = -1

# Whether we are currently on Windows or not
IS_WINDOWS: Final = os.name == "nt"

# Interval units for numpy dates and times
NP_DATETIME_UNITS: Final = "ns"

# Interval form to use for storing data as np.datetime64's
NP_DATETIME_FORM: Final = "datetime64[ns]"

# Interval form to use for storing data as np.timedelta64's
NP_TIMEDELTA_FORM: Final = "timedelta64[ns]"

# Scale factor for converting numpy time forms to seconds (nanoseconds per second)
NP_INT64_PER_SEC: Final = 10**9

# % Numpy constants
NP_ONE_SECOND: np.timedelta64
NP_ONE_MINUTE: np.timedelta64
NP_ONE_HOUR: np.timedelta64
NP_ONE_DAY: np.timedelta64
NP_NAT: np.datetime64
NP_DATETIME_MIN: np.datetime64
NP_DATETIME_MAX: np.datetime64
if HAVE_NUMPY:
    NP_ONE_SECOND = np.timedelta64(1, "s").astype(NP_TIMEDELTA_FORM)
    NP_ONE_MINUTE = np.timedelta64(1, "m").astype(NP_TIMEDELTA_FORM)
    NP_ONE_HOUR = np.timedelta64(1, "h").astype(NP_TIMEDELTA_FORM)
    NP_ONE_DAY = np.timedelta64(1, "D").astype(NP_TIMEDELTA_FORM)
    ZERO_NP_SECONDS = np.timedelta64(0, "s").astype(NP_TIMEDELTA_FORM)

    # -2**63   -2**63 + 1 ...... 0 ...... 2**63-1
    #  NP_NAT NP_DATETIME_MIN         NP_DATETIME_MAX
    NP_NAT = np.datetime64("NaT", NP_DATETIME_UNITS)
    NP_DATETIME_MIN = np.datetime64(np.iinfo(np.int64).min + 1, NP_DATETIME_UNITS)
    NP_DATETIME_MAX = np.datetime64(np.iinfo(np.int64).max, NP_DATETIME_UNITS)
else:
    NP_ONE_SECOND = NP_ONE_MINUTE = NP_ONE_HOUR = NP_ONE_DAY = ZERO_NP_SECONDS = None  # type: ignore[assignment]
    NP_NAT = NP_DATETIME_MIN = NP_DATETIME_MAX = None  # type: ignore[assignment]

# %% Functions
# None

# %% Unit test
if __name__ == "__main__":
    # print all the constants (anything in ALL_CAPS)
    fields = [k for k in sorted(dir()) if k.isupper()]
    max_len = max(map(len, fields))
    for field in fields:
        pad = " " * (max_len - len(field))
        value = locals()[field]
        print(f"{field}{pad} = {value}")

    # run unittests
    unittest.main(module="dstauffman.tests.test_constants", exit=False)
