r"""
Replacement utilities that are optimized for speed.

Notes
-----
#.  Written by David C. Stauffer in July 2020.
"""

#%% Imports
from __future__ import annotations
import doctest
import functools
import math
import unittest

try:
    from numba import float64, njit, vectorize
except ModuleNotFoundError:
    # Support for when you don't have numba.  Presumably you either aren't using these functions,
    # as they will be slow, or you are using pypy instead and it will run the JIT
    # Go through a bunch of worthless closures to get the necessary stubs
    def fake_decorator(func):
        r"""Fake decorator for when numba isn't installed."""
        @functools.wraps(func)
        def wrapped_decorator(*args, **kwargs):
            def real_decorator(func2):
                return func(func2, *args, **kwargs)
            return real_decorator
        return wrapped_decorator

    @fake_decorator
    def njit(func, *args, **kwargs):
        r"""Fake njit decorator for when numba isn't installed."""
        return func

    @fake_decorator
    def vectorize(func, *args, **kwargs):
        r"""Fake vectorize decorator for when numba isn't installed."""
        return func

    float64 = lambda x, y: None

#%% np_any
@njit(cache=True)
def np_any(x, /):
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
def np_all(x, /):
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
def issorted_opt(x, /, descend=False):
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

#%% Functions - prob_to_rate_opt
@vectorize([float64(float64, float64)], nopython=True, target='parallel', cache=True)  # TODO: can't use optional argument?
def prob_to_rate_opt(prob, time):
    r"""
    Convert a given probability and time to a rate.

    Parameters
    ----------
    prob : numpy.ndarray
        Probability of event happening over the given time
    time : float
        Time for the given probability in years

    Returns
    -------
    rate : numpy.ndarray
        Equivalent annual rate for the given probability and time

    Notes
    -----
    #.  Written by David C. Stauffer in January 2016.

    Examples
    --------
    >>> from dstauffman import prob_to_rate_opt
    >>> import numpy as np
    >>> prob = np.array([0, 0.1, 1])
    >>> time = 3
    >>> rate = prob_to_rate_opt(prob, time)
    >>> with np.printoptions(precision=8):
    ...     print(rate) # doctest: +NORMALIZE_WHITESPACE
    [0. 0.03512017 inf]

    """
    # check ranges
    if prob < 0:
        raise ValueError('Probability must be >= 0')
    if prob > 1:
        raise ValueError('Probability must be <= 1')
    # calculate rate
    if prob == 1:
        return math.inf
    if prob == 0:
        return prob
    return -math.log(1 - prob) / time

#%% Functions - rate_to_prob_opt
@vectorize([float64(float64, float64)], nopython=True, target='parallel', cache=True)  # TODO: can't use optional argument?
def rate_to_prob_opt(rate, time):
    r"""
    Convert a given rate and time to a probability.

    Parameters
    ----------
    rate : float
        Annual rate for the given time
    time : float
        Time period for the desired probability to be calculated from, in years

    Returns
    -------
    float
        Equivalent probability of event happening over the given time

    Notes
    -----
    #.  Written by David C. Stauffer in January 2016.
    #.  Converted to numba version by David C. Stauffer in November 2020.

    Examples
    --------
    >>> from dstauffman import rate_to_prob_opt
    >>> import numpy as np
    >>> rate = np.array([0, 0.1, 1, 100, np.inf])
    >>> time = 1./12
    >>> prob = rate_to_prob_opt(rate, time)
    >>> with np.printoptions(precision=8):
    ...     print(prob) # doctest: +NORMALIZE_WHITESPACE
    [0. 0.00829871 0.07995559 0.99975963 1. ]

    """
    # check ranges
    if rate < 0:
        raise ValueError('Rate must be >= 0')
    # calculate probability
    return 1 - math.exp(-rate * time)

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_optimized', exit=False)
    doctest.testmod(verbose=False)
