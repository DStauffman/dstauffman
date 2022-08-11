r"""
Replacement utilities that are optimized for speed using numba but not numpy.

Notes
-----
#.  Written by David C. Stauffer in July 2020.
#.  Moved into a submodule by David C. Stauffer in February 2021.
"""

#%% Imports
from __future__ import annotations

import doctest
import math
from typing import Sequence, TYPE_CHECKING, Union
import unittest

from nubs.passthrough import fake_jit, HAVE_NUMBA, HAVE_NUMPY, ncjit, TARGET

if HAVE_NUMBA:
    from numba import float32, float64, int32, int64, vectorize
else:
    if HAVE_NUMPY:
        from numpy import vectorize
    float32 = float64 = int32 = int64 = fake_jit

if TYPE_CHECKING:
    from numpy import ndarray

    _N = Union[float, ndarray]

#%% np_any
@ncjit
def np_any(x: Sequence, /) -> bool:
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
    >>> from nubs import np_any
    >>> import numpy as np
    >>> x = np.zeros(1000, dtype=bool)
    >>> print(np_any(x))
    False

    >>> x[333] = True
    >>> print(np_any(x))
    True

    """
    for i in range(len(x)):  # pylint: disable=consider-using-enumerate
        if x[i]:
            return True
    return False


#%% np_all
@ncjit
def np_all(x: Sequence, /) -> bool:
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
    >>> from nubs import np_all
    >>> import numpy as np
    >>> x = np.ones(1000, dtype=bool)
    >>> print(np_all(x))
    True

    >>> x[333] = False
    >>> print(np_all(x))
    False

    """
    for i in range(len(x)):  # pylint: disable=consider-using-enumerate
        if not x[i]:
            return False
    return True


#%% issorted_opt
@ncjit
def issorted_opt(x: Sequence, /, descend: bool = False) -> bool:
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
    >>> from nubs import issorted_opt
    >>> import numpy as np
    >>> x = np.array([1, 3, 3, 5, 7])
    >>> print(issorted_opt(x))
    True

    >>> y = np.array([3, 5, 1, 7])
    >>> print(issorted_opt(y))
    False

    """
    if descend:
        for i in range(len(x) - 1):
            if x[i + 1] > x[i]:
                return False
    else:
        for i in range(len(x) - 1):
            if x[i + 1] < x[i]:
                return False
    return True


#%% Functions - prob_to_rate_opt
# @vectorize (done below)
def prob_to_rate_opt(prob: _N, time: _N) -> _N:
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
    >>> from nubs import HAVE_NUMBA, prob_to_rate_opt
    >>> import numpy as np
    >>> prob = np.array([0, 0.1, 1])
    >>> time = 3
    >>> rate = prob_to_rate_opt(prob, time) if HAVE_NUMBA else [prob_to_rate_opt(p, time) for p in prob]
    >>> print(np.array_str(np.asanyarray(rate), precision=8))  # doctest: +NORMALIZE_WHITESPACE
    [0. 0.03512017 inf]

    """
    # check ranges
    if prob < 0:
        raise ValueError("Probability must be >= 0")
    if prob > 1:
        raise ValueError("Probability must be <= 1")
    # calculate rate
    if prob == 1:
        return math.inf
    if prob == 0:
        return prob
    return -math.log(1 - prob) / time


if HAVE_NUMBA:
    prob_to_rate_opt = vectorize([float64(float64, float64)], nopython=True, target=TARGET, cache=True)(prob_to_rate_opt)
elif HAVE_NUMPY:
    prob_to_rate_opt = vectorize(prob_to_rate_opt, cache=True)

#%% Functions - rate_to_prob_opt
# @vectorize (done below)
def rate_to_prob_opt(rate: _N, time: _N) -> _N:
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
    >>> from nubs import HAVE_NUMBA, rate_to_prob_opt
    >>> import numpy as np
    >>> rate = np.array([0, 0.1, 1, 100, np.inf])
    >>> time = 1./12
    >>> prob = rate_to_prob_opt(rate, time) if HAVE_NUMBA else [rate_to_prob_opt(r, time) for r in rate]
    >>> print(np.array_str(np.asanyarray(prob), precision=8))  # doctest: +NORMALIZE_WHITESPACE
    [0. 0.00829871 0.07995559 0.99975963 1. ]

    """
    # check ranges
    if rate < 0:
        raise ValueError("Rate must be >= 0")
    # calculate probability
    return 1 - math.exp(-rate * time)


if HAVE_NUMBA:
    rate_to_prob_opt = vectorize([float64(float64, float64)], nopython=True, target=TARGET, cache=True)(rate_to_prob_opt)
elif HAVE_NUMPY:
    rate_to_prob_opt = vectorize(rate_to_prob_opt, cache=True)

#%% Functions - zero_divide
# @vectorize (done below)
def zero_divide(num: _N, den: _N) -> _N:
    r"""
    Numba compatible version of np.divide(num, den, out=np.zeros_like(num), where=den!=0).

    Parameters
    ----------
    num : float
        Numerator
    den : float
        Denominator

    Returns
    -------
    float
        result of divison, except return zero for anything divided by zero, including 0/0

    Notes
    -----
    #.  Written by David C. Stauffer in February 2021.

    Examples
    --------
    >>> from nubs import zero_divide
    >>> print(zero_divide(1., .2))
    5.0

    >>> print(zero_divide(3.14, 0.))
    0.0

    >>> print(zero_divide(0., 0.))
    0.0

    """
    if den == 0.0:
        return 0.0
    return num / den


if HAVE_NUMBA:
    zero_divide = vectorize(
        [float32(int32, int32), float64(int64, int64), float32(float32, float32), float64(float64, float64)],
        nopython=True,
        target=TARGET,
        cache=True,
    )(zero_divide)
elif HAVE_NUMPY:
    zero_divide = vectorize(zero_divide, cache=True)

#%% Unit test
if __name__ == "__main__":
    unittest.main(module="nubs.tests.test_optimized", exit=False)
    doctest.testmod(verbose=False)
