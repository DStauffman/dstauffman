r"""
Methods designed to be compiled with numba in nopython=True mode.

Notes
-----
#.  Written by David C. Stauffer in January 2021.
"""

#%% Imports
from __future__ import annotations
import doctest
from typing import Tuple, Union
import unittest

from dstauffman.constants import HAVE_NUMBA, HAVE_NUMPY

if HAVE_NUMPY:
    import numpy as np

if HAVE_NUMBA:
    from numba import njit
    from numba.typed import List
else:
    from dstauffman.constants import fake_decorator

    @fake_decorator
    def njit(func, *args, **kwargs):
        r"""Fake njit decorator for when numba isn't installed."""
        return func

#%% _reduce_shape
@njit(cache=True)
def _reduce_shape(shape: Tuple, axis: int) -> List[int]:
    r"""Gives what will be the reduced array shape after applying an operation to the given axis."""
    num = len(shape)
    assert num > axis, 'The specified axis must be less than the number of dimensions.'
    out = List()
    for (i, s) in enumerate(shape):
        if i != axis:
            out.append(s)
    return out

#%% issorted_ascend
@njit(cache=True)
def issorted_ascend(x: Union[np.ndarray[int, 1], np.ndarray[float, 1]]) -> bool:
    r"""
    Tells whether the given array is sorted in ascending order or not.

    Parameters
    ----------
    x : array_like
        Input array

    Notes
    -----
    #.  Written by David C. Stauffer in January 2021.

    Examples
    --------
    >>> from dstauffman import issorted_ascend
    >>> import numpy as np
    >>> x = np.array([1, 3, 3, 5, 7])
    >>> print(issorted_ascend(x))
    True

    >>> y = np.array([3, 5, 1, 7])
    >>> print(issorted_ascend(y))
    False

    """
    return np.all(x[:-1] <= x[1:])  # type: ignore[no-any-return]

#%% issorted_descend
@njit(cache=False)
def issorted_descend(x):
    r"""
    Tells whether the given array is sorted in descending order or not.

    Parameters
    ----------
    x : array_like
        Input array

    Notes
    -----
    #.  Written by David C. Stauffer in January 2021.

    Examples
    --------
    >>> from dstauffman import issorted_descend
    >>> import numpy as np
    >>> x = np.array([1, 3, 3, 5, 7])
    >>> print(issorted_descend(x))
    False

    >>> y = np.array([7, 5, 3, 3, 1])
    >>> print(issorted_descend(y))
    True

    """
    return np.all(x[1:] <= x[:-1])

#%% Functions - np_all_axis0
@njit(cache=True)
def np_all_axis0(x):
    r"""
    Numba compatible version of np.all(x, axis=0).

    Parameters
    ----------
    x : array_like
        Input array

    Notes
    -----
    #.  Written by David C. Stauffer in January 2021.

    Examples
    --------
    >>> from dstauffman import np_all_axis0
    >>> import numpy as np
    >>> x = np.array([[True, True, False, False], [True, False, True, False]], dtype=np.bool_)
    >>> print(np_all_axis0(x))
    [ True False False False]

    """
    if x.ndim > 1:
        out = np.ones(x.shape[1:], dtype=np.bool_)
        for i in range(x.shape[0]):
            out = np.logical_and(out, x[i, :, ...])
    else:
        out = np.all(x)
    return out

#%% Functions - np_all_axis1
@njit(cache=True)
def np_all_axis1(x):
    """Numba compatible version of np.all(x, axis=1).

    Parameters
    ----------
    x : array_like
        Input array

    Notes
    -----
    #.  Written by David C. Stauffer in January 2021.

    Examples
    --------
    >>> from dstauffman import np_all_axis1
    >>> import numpy as np
    >>> x = np.array([[True, True, False, False], [True, False, True, False]], dtype=np.bool_)
    >>> print(np_all_axis1(x))
    [False False]

    """
    out = x[:, 0, ...]
    for i in range(1, x.shape[1]):
        out = np.logical_and(out, x[:, i, ...])
    return out

#%% Functions - np_any_axis0
@njit(cache=True)
def np_any_axis0(x):
    """Numba compatible version of np.any(x, axis=0).

    Parameters
    ----------
    x : array_like
        Input array

    Notes
    -----
    #.  Written by David C. Stauffer in January 2021.

    Examples
    --------
    >>> from dstauffman import np_any_axis0
    >>> import numpy as np
    >>> x = np.array([[True, True, False, False], [True, False, True, False]], dtype=np.bool_)
    >>> print(np_any_axis0(x))
    [ True  True  True False]

    """
    if x.ndim > 1:
        out = np.zeros(x.shape[1:], dtype=np.bool_)
        for i in range(x.shape[0]):
            out = np.logical_or(out, x[i, :, ...])
    else:
        out = np.any(x)
    return out

#%% Functions - np_any_axis1
@njit(cache=True)
def np_any_axis1(x):
    """Numba compatible version of np.any(x, axis=1).

    Parameters
    ----------
    x : array_like
        Input array

    Notes
    -----
    #.  Written by David C. Stauffer in January 2021.

    Examples
    --------
    >>> from dstauffman import np_any_axis1
    >>> import numpy as np
    >>> x = np.array([[True, True, False, False], [True, False, True, False]], dtype=np.bool_)
    >>> print(np_any_axis1(x))
    [ True  True]

    """
    out = x[:, 0, ...]
    for i in range(1, x.shape[1]):
        out = np.logical_or(out, x[:, i, ...])
    return out

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_numba', exit=False)
    doctest.testmod(verbose=False)
