r"""
Replacement utilities that are optimized for speed.

Notes
-----
#.  Written by David C. Stauffer in July 2020.
"""

#%% Imports
import doctest
import functools
import unittest

try:
    from numba import njit
except ModuleNotFoundError:
    # Support for when you don't have numba.  Presumably you either aren't using these functions,
    # as they will be slow, or you are using pypy instead and it will run the JIT
    # Go through a bunch of worthless closures to get the necessary stubs
    def fake_decorator(func):
        @functools.wraps(func)
        def wrapped_decorator(*args, **kwargs):
            def real_decorator(func2):
                return func(func2, *args, **kwargs)
            return real_decorator
        return wrapped_decorator

    @fake_decorator
    def njit(func, *args, **kwargs):
        return(func)

#%% np_any
@njit(cache=True)
def np_any(x):
    r"""
    Returns true if anything in the vector is true.

    Parameters
    ----------
    x : array_like
        Input array

    Notes
    -----
    #.  Replacement for np.any with short-circuiting.
        It is faster if something is likely True, but slower if it has to check the entire array.
    #.  Written by David C. Stauffer in July 2020.

    Examples
    --------
    >>> from dstauffman import np_any
    >>> import numpy as np
    >>> x = np.zeros(1000, dtype=bool)
    >>> print(np_any(x))
    False

    >>> x[333] = True
    >>> print(np_any(x))
    True

    """
    for i in range(len(x)):
        if x[i]:
            return True
    return False

#%% np_all
@njit(cache=True)
def np_all(x):
    r"""
    Returns true if everything in the vector is true.

    Parameters
    ----------
    x : array_like
        Input array

    Notes
    -----
    #.  Replacement for np.all with short-circuiting.
        It is faster if something is likely False, but slower if it has to check the entire array.
    #.  Written by David C. Stauffer in July 2020.

    Examples
    --------
    >>> from dstauffman import np_all
    >>> import numpy as np
    >>> x = np.ones(1000, dtype=bool)
    >>> print(np_all(x))
    True

    >>> x[333] = False
    >>> print(np_all(x))
    False

    """
    for i in range(len(x)):
        if not x[i]:
            return False
    return True

#%% issorted_opt
@njit(cache=True)
def issorted_opt(x, descend=False):
    r"""
    Tells whether the given array is sorted or not.

    Parameters
    ----------
    x : array_like
        Input array
    descend : bool, optional, default is False
        Whether to check that the array is sorted in descending order

    Notes
    -----
    #.  Written by David C. Stauffer in July 2020.

    Examples
    --------
    >>> from dstauffman import issorted_opt
    >>> import numpy as np
    >>> x = np.array([1, 3, 3, 5, 7])
    >>> print(issorted_opt(x))
    True

    >>> y = np.array([3, 5, 1, 7])
    >>> print(issorted_opt(y))
    False

    """
    if descend:
        for i in range(len(x)-1):
            if x[i+1] > x[i]:
                return False
    else:
        for i in range(len(x)-1):
            if x[i+1] < x[i] :
                    return False
    return True

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_optimized', exit=False)
    doctest.testmod(verbose=False)
