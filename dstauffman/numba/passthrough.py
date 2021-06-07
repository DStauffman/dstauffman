r"""
Methods designed to be compiled with numba in nopython=True mode.

Notes
-----
#.  Written by David C. Stauffer in January 2021.
"""

#%% Normal Imports
import doctest
import functools
import platform
import sys
import unittest

if platform.python_implementation() == 'CPython':
    try:
        # Try importing numba to determine if it is there.  Use this instead of HAVE_NUMBA from dstauffman
        # to avoid any circular dependencies
        from numba import njit  # type: ignore[attr-defined]
        HAVE_NUMBA = True
    except ModuleNotFoundError:
        HAVE_NUMBA = False
else:
    HAVE_NUMBA = False

#%% Support Functions
def _fake_decorator(func):
    r"""Fake decorator for when numba isn't installed."""
    @functools.wraps(func)
    def wrapped_decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # must treat this differently if no arguments were passed
            return func(args[0])
        def real_decorator(func2):
            return func(func2, *args, **kwargs)
        return real_decorator
    return wrapped_decorator

@_fake_decorator
def fake_jit(func, *args, **kwargs):
    r"""Fake jit decorator for when numba isn't installed."""
    return func

#%% Conditional imports
if HAVE_NUMBA:
    # always cached version of njit, which is also jit(cache=True, nopython=True)
    def ncjit(func, *args, **kwargs):
        r"""Fake decorator for when numba isn't installed."""
        return njit(func, cache=True, *args, **kwargs)

    # target for vectorized functions
    assert sys.version_info.major == 3, 'Must be Python 3'
    assert sys.version_info.minor >= 8, 'Must be Python v3.8 or higher'
    if sys.version_info.minor > 8:
        TARGET = 'parallel'  # Python v3.9+
    else:
        TARGET = 'cpu'
else:
    # Support for when you don't have numba.  Note, some functions won't work as expected
    TARGET = ''
    ncjit = fake_jit

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_numba_passthrough', exit=False)
    doctest.testmod(verbose=False)
