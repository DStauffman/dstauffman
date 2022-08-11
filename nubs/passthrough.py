r"""
Methods designed to be compiled with numba in nopython=True mode.

Notes
-----
#.  Written by David C. Stauffer in January 2021.
"""

#%% Normal Imports
from __future__ import annotations

import doctest
import functools
import platform
import sys
from typing import Any, Callable, TYPE_CHECKING
import unittest

if platform.python_implementation() == "CPython":
    try:
        # Try importing numba to determine if it is there.
        from numba import njit

        HAVE_NUMBA = True
    except ModuleNotFoundError:
        HAVE_NUMBA = False
else:
    HAVE_NUMBA = False  # pragma: no cover

try:
    import numpy

    assert numpy
    HAVE_NUMPY = True
except ModuleNotFoundError:
    HAVE_NUMPY = False

if TYPE_CHECKING:
    _C = Callable[..., Any]

#%% Support Functions
def _fake_decorator(func: _C) -> _C:
    r"""Fake decorator for when numba isn't installed."""

    @functools.wraps(func)
    def wrapped_decorator(*args, **kwargs):  # pragma: no cover
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # must treat this differently if no arguments were passed
            return func(args[0])

        def real_decorator(func2: _C) -> _C:
            return func(func2, *args, **kwargs)  # type: ignore[no-any-return]

        return real_decorator

    return wrapped_decorator


@_fake_decorator
def fake_jit(func: _C, *args, **kwargs) -> _C:  # pylint: disable=unused-argument
    r"""Fake jit decorator for when numba isn't installed."""
    return func


#%% Conditional imports
if HAVE_NUMBA:
    # always cached version of njit, which is also jit(cache=True, nopython=True)
    def ncjit(func: _C, *args, **kwargs) -> _C:
        r"""Fake decorator for when numba isn't installed."""
        return njit(func, cache=True, *args, **kwargs)  # type: ignore[no-any-return]

    # target for vectorized functions
    assert sys.version_info.major == 3, "Must be Python 3"
    assert sys.version_info.minor >= 8, "Must be Python v3.8 or higher"
    # Note: no longer using "parallel" in Python v3.9+ as it breaks the vectorize error catching
    TARGET = "cpu" if sys.version_info.minor > 8 else "cpu"
else:
    # Support for when you don't have numba.  Note, some functions won't work as expected
    TARGET = ""
    ncjit = fake_jit

#%% Unit test
if __name__ == "__main__":
    unittest.main(module="nubs.tests.test_passthrough", exit=False)
    doctest.testmod(verbose=False)
