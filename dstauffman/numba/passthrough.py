r"""
Methods designed to be compiled with numba in nopython=True mode.

Notes
-----
#.  Written by David C. Stauffer in January 2021.
"""

#%% Imports
import doctest
import functools
import sys
import unittest

from dstauffman import HAVE_NUMBA

if HAVE_NUMBA:
    from numba import float64, int32, jit, njit, vectorize
    from numba.typed import List
    from numba.experimental import jitclass

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
    # Go through a bunch of worthless closures to get the necessary stubs
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

    # fake constants
    TARGET = ''

    # fake decorators
    @_fake_decorator
    def jit(func, *args, **kwargs):
        r"""Fake jit decorator for when numba isn't installed."""
        return func

    # fake types
    List = list
    int32 = jit  # int as a callable with multiple args?
    float64 = jit  # float as a callable with multiple args?

    njit = jit
    ncjit = jit
    jitclass = jit
    vectorize = jit

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_numba_passthrough', exit=False)
    doctest.testmod(verbose=False)
