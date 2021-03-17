r"""
Methods designed to be compiled with numba in nopython=True mode.

Notes
-----
#.  Written by David C. Stauffer in January 2021.
"""

#%% Imports
from __future__ import annotations
import doctest
from typing import Tuple
import unittest

from dstauffman.numba.passthrough import boolean, ncjit, List

try:
    # Note: avoid using HAVE_NUMPY from dstauffman to avoid circular imports
    import numpy as np
except ModuleNotFoundError:
    pass

#%% _reduce_shape
@ncjit
def _reduce_shape(shape: Tuple, axis: int) -> List[int]:
    r"""Gives what will be the reduced array shape after applying an operation to the given axis."""
    num = len(shape)
    if num <= axis:
        raise ValueError('The specified axis must be less than the number of dimensions.')
    out = List()
    for (i, s) in enumerate(shape):
        if i != axis:
            out.append(s)
    return out

#%% issorted_ascend
@ncjit
def issorted_ascend(x: np.ndarray) -> boolean:
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
    >>> from dstauffman.numba import issorted_ascend
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
@ncjit
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
    >>> from dstauffman.numba import issorted_descend
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
@ncjit
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
    >>> from dstauffman.numba import np_all_axis0
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
@ncjit
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
    >>> from dstauffman.numba import np_all_axis1
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
@ncjit
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
    >>> from dstauffman.numba import np_any_axis0
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
@ncjit
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
    >>> from dstauffman.numba import np_any_axis1
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
    unittest.main(module='dstauffman.tests.test_numba_numpy_mods', exit=False)
    doctest.testmod(verbose=False)
