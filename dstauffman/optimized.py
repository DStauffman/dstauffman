r"""
Replacement utilities that are optimized for speed.

Notes
-----
#.  Written by David C. Stauffer in July 2020.
"""

#%% Imports
import doctest
import unittest

import numba

#%% np_any
@numba.njit(cache=True)
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
@numba.njit(cache=True)
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

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_optimized', exit=False)
    doctest.testmod(verbose=False)
