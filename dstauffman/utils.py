r"""
Generic utilities that can be independently defined and used by other modules.

Notes
-----
#.  By design, this module does not reference any other piece of the dstauffman code base except
        constants to avoid circular references.
#.  Written by David C. Stauffer in March 2015.

"""  # pylint: disable=too-many-lines

# %% Imports
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import datetime
import doctest
from functools import reduce
import inspect
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Callable, Generator, Iterable, Literal, NotRequired, overload, TYPE_CHECKING, TypedDict, Unpack
import unittest
import warnings

from slog import ReturnCodes, write_text_file

from dstauffman.constants import HAVE_NUMPY, HAVE_SCIPY, IS_WINDOWS
from dstauffman.units import MONTHS_PER_YEAR

if HAVE_NUMPY:
    import numpy as np
    from numpy import inf, isnan, logical_not, nan
else:
    from math import inf, isnan, nan  # type: ignore[assignment]

    logical_not = lambda x: not x  # type: ignore[assignment]  # pylint: disable=unnecessary-lambda-assignment  # noqa: E731

if HAVE_SCIPY:
    from scipy.interpolate import interp1d
    from scipy.signal import butter, sosfiltfilt

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    _B = NDArray[np.bool_]
    _D = NDArray[np.datetime64]
    _I = NDArray[np.int_]
    _N = NDArray[np.floating]
    _SingleNum = int | float | np.datetime64
    _Lists = _N | list[_N] | tuple[_N, ...]
    _FN = float | np.floating | _N
    _Array = _B | _D | _I | _N

    class _PrintOptsKwArgs(TypedDict):
        precision: NotRequired[int | None]
        threshold: NotRequired[int | None]
        edgeitems: NotRequired[int | None]
        linewidth: NotRequired[int | None]
        suppress: NotRequired[bool | None]
        nanstr: NotRequired[str | None]
        infstr: NotRequired[str | None]
        sign: NotRequired[Literal["-" | "+" | " "] | None]
        formatter: NotRequired[dict[str, Callable]]
        floatmode: NotRequired[Literal["fixed" | "unique" | "maxprec" | "maxprec_equal"] | None]

    class _ButterKwArgs(TypedDict):
        btype: NotRequired[Literal["lowpass", "highpass", "bandpass", "bandstop"]]
        analog: NotRequired[bool]


# %% Globals
_ALLOWED_ENVS: dict[str, str] | None = None  # allows any environment variables to be invoked


# %% Functions - _nan_equal
def _nan_equal(  # pylint: disable=too-many-return-statements  # noqa: C901
    a: Any, b: Any, /, tolerance: float | None = None
) -> bool:
    r"""
    Test ndarrays for equality, but ignore NaNs.

    Parameters
    ----------
    a : ndarray
        Array one
    b : ndarray
        Array two
    tolerance : float, optional
        Numerical tolerance used to compare two numbers that are close together to consider them equal

    Returns
    -------
    bool
        Flag for whether the inputs are the same or not

    Examples
    --------
    >>> from dstauffman.utils import _nan_equal
    >>> import numpy as np
    >>> a = np.array([1, 2, np.nan])
    >>> b = np.array([1, 2, np.nan])
    >>> print(_nan_equal(a, b))
    True

    >>> a = np.array([1, 2, np.nan])
    >>> b = np.array([3, 2, np.nan])
    >>> print(_nan_equal(a, b))
    False

    """

    def _is_nan(x: Any) -> bool:
        try:
            out = isnan(x)
        except Exception:  # pylint: disable=broad-exception-caught
            return False
        return out  # type: ignore[no-any-return]

    try:
        if HAVE_NUMPY:
            # use numpy testing module to assert that they are equal (ignores NaNs)
            do_simple = tolerance is None or tolerance == 0 or a is None or b is None
            if not do_simple:
                # see if these can be cast to numeric values that can be compared
                try:
                    _ = np.isfinite(a) & np.isfinite(b)
                except TypeError:
                    do_simple = True
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
            if do_simple:
                np.testing.assert_array_equal(a, b)
            else:
                assert tolerance is not None
                np.testing.assert_allclose(a, b, atol=tolerance, equal_nan=True)
        else:
            if tolerance is not None and tolerance != 0:
                raise ValueError("You must have numpy installed to use a non-zero tolerance.")
            if a != b:
                return False
            if hasattr(a, "__len__"):
                if hasattr(b, "__len__"):
                    if len(a) != len(b):
                        return False
                    return all(x == y or _is_nan(x) or _is_nan(y) for (x, y) in zip(a, b))
                return False
            if hasattr(b, "__len__"):
                return False
            return a == b or _is_nan(a) or _is_nan(b)
    except AssertionError:
        # if assertion fails, then they are not equal
        return False
    return True


# %% Functions - find_in_range
def find_in_range(
    value: ArrayLike,
    min_: _SingleNum = -inf,
    max_: _SingleNum = inf,
    *,
    inclusive: bool = False,
    mask: bool | _B | None = None,
    precision: _SingleNum = 0,
    left: bool = False,
    right: bool = False,
) -> _D | _I | _N:
    r"""
    Finds values in the given range.

    Parameters
    ----------
    value : (N,) ndarray of float
        Value to compare against, which could be NaN
    min_ : int or float, optional
        Minimum value to include in range
    max_ : int or float, optional
        Maximum value to include in range
    inclusive : bool, optional, default is False
        Whether to inclusively count both endpoints (overrules left and right)
    mask : (N,) ndarray of bool, optional
        A mask to preapply to the results
    precision : int or float, optional, default is zero
        A precision to apply to the comparisons
    left : bool, optional, default is False
        Whether to include the left endpoint in the range
    right : bool, optional, default is False
        Whether to include the right endpoint in the range

    Returns
    -------
    valid : (N,) ndarray of bool
        True or False flag for whether the person in the given range

    Notes
    -----
    #.  Written by David C. Stauffer in August 2020.

    Examples
    --------
    >>> from dstauffman import find_in_range
    >>> import numpy as np
    >>> valid = find_in_range(np.array([-1, 15, 30.3, 40, 0, 0, 10, np.nan, 8000]), min_=12, max_=35)
    >>> print(valid)
    [False  True  True False False False False False False]

    """
    # ensure this is an ndarray
    value = np.asanyarray(value)
    # find the people with valid values to compare against
    not_nan = ~np.isnan(value)
    if mask is not None:
        not_nan &= mask
    # find those greater than the minimum bound
    if np.isfinite(min_):
        func = np.greater_equal if inclusive or left else np.greater
        valid = func(value, min_ - precision, out=np.zeros(value.shape, dtype=bool), where=not_nan)  # type: ignore[operator]
    else:
        if np.isnan(min_) or np.sign(min_) > 0:
            raise AssertionError("The minimum should be -np.inf if not finite.")
        valid = not_nan.copy()
    # combine with those less than the maximum bound
    if np.isfinite(max_):
        func = np.less_equal if inclusive or right else np.less  # type: ignore[assignment]
        valid &= func(value, max_ + precision, out=np.zeros(value.shape, dtype=bool), where=not_nan)  # type: ignore[operator]
    else:
        if np.isnan(max_) or np.sign(max_) < 0:
            raise AssertionError("The maximum should be np.inf if not finite.")
    return valid  # type: ignore[no-any-return]


# %% Functions - rms
@overload
def rms(data: ArrayLike, axis: Literal[None] = ..., keepdims: bool = ..., ignore_nans: bool = ...) -> np.floating: ...
@overload
def rms(data: ArrayLike, axis: int, keepdims: Literal[False] = ..., ignore_nans: bool = ...) -> np.floating | _N: ...
@overload
def rms(data: ArrayLike, axis: int, keepdims: Literal[True], ignore_nans: bool = ...) -> _N: ...
def rms(data: ArrayLike, axis: int | None = None, keepdims: bool = False, ignore_nans: bool = False) -> np.floating | _N:
    r"""
    Calculate the root mean square of a number series.

    Parameters
    ----------
    data : array_like
        input data
    axis : int, optional
        Axis along which RMS is computed. The default is to compute the RMS of the flattened array.
    keepdims : bool, optional
        If true, the axes which are reduced are left in the result as dimensions with size one.
        With this option, the result will broadcast correctly against the original `data`.

    Returns
    -------
    out : ndarray
        RMS results

    See Also
    --------
    numpy.mean, numpy.nanmean, numpy.conj, numpy.sqrt

    Notes
    -----
    #.  Written by David C. Stauffer in Mar 2015.

    Examples
    --------
    >>> from dstauffman import rms
    >>> out = rms([0, 1, 0., -1])
    >>> print(f"{out:.12f}")
    0.707106781187

    """
    # check for empty data
    if not np.isscalar(data) and len(data) == 0:  # type: ignore[arg-type]
        return np.array(np.nan)
    # do the root-mean-square, but use x * conj(x) instead of square(x) to handle complex numbers correctly
    if not ignore_nans:
        out = np.sqrt(np.mean(data * np.conj(data), axis=axis, keepdims=keepdims))  # type: ignore[arg-type]
    else:
        # check for all NaNs case
        if np.all(np.isnan(data)):
            if axis is None:
                out = np.nan
            else:
                assert isinstance(data, np.ndarray)
                if keepdims:
                    shape = (*data.shape[:axis], 1, *data.shape[axis + 1 :])
                else:
                    shape = (*data.shape[:axis], *data.shape[axis + 1 :])
                out = np.full(shape, np.nan)
        else:
            out = np.sqrt(np.nanmean(data * np.conj(data), axis=axis, keepdims=keepdims))  # type: ignore[arg-type]
    # return the result
    return out  # type: ignore[no-any-return]


# %% Functions - rss
@overload
def rss(data: ArrayLike, axis: Literal[None] = ..., keepdims: bool = ..., ignore_nans: bool = ...) -> np.floating: ...
@overload
def rss(data: ArrayLike, axis: int, keepdims: Literal[False] = ..., ignore_nans: bool = ...) -> np.floating | _N: ...
@overload
def rss(data: ArrayLike, axis: int, keepdims: Literal[True], ignore_nans: bool = ...) -> _N: ...
def rss(data: ArrayLike, axis: int | None = None, keepdims: bool = False, ignore_nans: bool = False) -> np.floating | _N:
    r"""
    Calculate the root sum square of a number series.

    Parameters
    ----------
    data : array_like
        input data
    axis : int, optional
        Axis along which RMS is computed. The default is to compute the RMS of the flattened array.
    keepdims : bool, optional
        If true, the axes which are reduced are left in the result as dimensions with size one.
        With this option, the result will broadcast correctly against the original `data`.

    Returns
    -------
    out : ndarray
        RSS results

    See Also
    --------
    numpy.sum, numpy.nansum, numpy.conj

    Notes
    -----
    #.  Written by David C. Stauffer in April 2016.
    #.  Bug fixed by David C. Stauffer in February 2024 to actually take the square root.

    Examples
    --------
    >>> from dstauffman import rss
    >>> out = rss([0, 1, 0., -1])
    >>> print(f"{out:.8f}")
    1.41421356

    """
    # check for empty data
    if not np.isscalar(data) and len(data) == 0:  # type: ignore[arg-type]
        return np.array(np.nan)
    # do the root-mean-square, but use x * conj(x) instead of square(x) to handle complex numbers correctly
    if not ignore_nans:
        out = np.sqrt(np.sum(data * np.conj(data), axis=axis, keepdims=keepdims))
    else:
        # check for all NaNs case
        if np.all(np.isnan(data)):
            if axis is None:
                out = np.array(np.nan)
            else:
                assert isinstance(data, np.ndarray)
                if keepdims:
                    shape = (*data.shape[:axis], 1, *data.shape[axis + 1 :])
                else:
                    shape = (*data.shape[:axis], *data.shape[axis + 1 :])
                out = np.full(shape, np.nan)
        else:
            out = np.sqrt(np.nansum(data * np.conj(data), axis=axis, keepdims=keepdims))
    # return the result
    return out


# %% Functions - compare_two_classes
def compare_two_classes(  # noqa: C901
    c1: Any,
    c2: Any,
    /,
    *,
    suppress_output: bool = False,
    names: tuple[str, str] | list[str] | None = None,
    ignore_callables: bool = True,
    compare_recursively: bool = True,
    is_subset: bool = False,
    tolerance: float | None = None,
    exclude: set[str] | None = None,
) -> bool:
    r"""
    Compare two classes by going through all their public attributes and showing that they are equal.

    Parameters
    ----------
    c1 : class object
        Any class object
    c2 : class object
        Any other class object
    suppress_output : bool, optional
        If True, suppress the information printed to the screen, defaults to False.
    names : list of str, optional
        List of the names to be printed to the screen for the two input classes.
    ignore_callables : bool, optional
        If True, ignore differences in callable attributes (i.e. methods), defaults to True.
    is_subset : bool, optional
        If True, only compares in c1 is a strict subset of c2, but c2 can have extra fields, defaults to False
    tolerance : float, optional
        Numerical tolerance used to compare two numbers that are close together to consider them equal
    exclude: set of str, optional
        Field attributes to exclude in the comparison, typically because they are non-standard and can't be compared

    Returns
    -------
    is_same : bool
        True/False flag for whether the two class are the same.

    Examples
    --------
    >>> from dstauffman import compare_two_classes
    >>> c1 = type("Class1", (object, ), {"a": 0, "b" : "[1, 2, 3]", "c": "text"})
    >>> c2 = type("Class2", (object, ), {"a": 0, "b" : "[1, 2, 4]", "d": "text"})
    >>> is_same = compare_two_classes(c1, c2)
    b is different from c1 to c2.
    c is only in c1.
    d is only in c2.
    "c1" and "c2" are not the same.

    """

    def _not_true_print() -> bool:
        r"""Set is_same to False and optionally prints information to the screen."""
        is_same = False
        if not suppress_output:
            print(f"{this_attr} is different from {name1} to {name2}.")
        return is_same

    def _is_function(obj: Any) -> bool:
        r"""Determine whether the object is a function or not."""
        # need second part for Python compatibility for v2.7, which distinguishes unbound methods from functions.
        return inspect.isfunction(obj) or inspect.ismethod(obj) or inspect.isbuiltin(obj)

    def _is_class_instance(obj: Any) -> bool:
        r"""Determine whether the object is an instance of a class or not."""
        return hasattr(obj, "__dict__") and not _is_function(obj)  # and hasattr(obj, "__call__")

    def _is_public(name: str) -> bool:
        r"""Return True if the name is public, ie doesn't start with an underscore."""
        return not name.startswith("_")

    # preallocate answer to True until proven otherwise
    is_same = True
    # get names if specified
    if names is not None:
        name1 = names[0]
        name2 = names[1]
    else:
        name1 = "c1"
        name2 = "c2"
    # simple test
    if c1 is not c2:  # pylint: disable=too-many-nested-blocks
        # get the list of public attributes
        attrs1 = frozenset(filter(_is_public, dir(c1)))
        attrs2 = frozenset(filter(_is_public, dir(c2)))
        # compare the attributes that are in both
        same = attrs1 & attrs2
        for this_attr in sorted(same):
            if exclude is not None and this_attr in exclude:
                continue
            # alias the attributes
            attr1 = inspect.getattr_static(c1, this_attr)
            attr2 = inspect.getattr_static(c2, this_attr)
            # determine if this is a subclass
            if _is_class_instance(attr1):
                if _is_class_instance(attr2):
                    if compare_recursively:
                        names = [name1 + "." + this_attr, name2 + "." + this_attr]
                        # Note: don't want the "and" to short-circuit, so do the "and is_same" last
                        if isinstance(attr1, dict) and isinstance(attr2, dict):
                            is_same = (
                                compare_two_dicts(
                                    attr1,
                                    attr2,
                                    suppress_output=suppress_output,
                                    names=names,
                                    is_subset=is_subset,
                                    tolerance=tolerance,
                                )
                                and is_same
                            )
                        else:
                            is_same = (
                                compare_two_classes(
                                    attr1,
                                    attr2,
                                    suppress_output=suppress_output,
                                    names=names,
                                    ignore_callables=ignore_callables,
                                    compare_recursively=compare_recursively,
                                    is_subset=is_subset,
                                    tolerance=tolerance,
                                )
                                and is_same
                            )
                        continue
                    continue
                is_same = _not_true_print()
                continue
            if _is_class_instance(attr2):
                is_same = _not_true_print()
            if _is_function(attr1) or _is_function(attr2):
                if ignore_callables:
                    continue  # pragma: no cover (actually covered, optimization issue)
                is_same = _not_true_print()
                continue
            # if any differences, then this test fails
            if isinstance(attr1, Mapping) and isinstance(attr2, Mapping):
                is_same = (
                    compare_two_dicts(attr1, attr2, suppress_output=True, is_subset=is_subset, tolerance=tolerance) and is_same
                )
            elif logical_not(_nan_equal(attr1, attr2, tolerance=tolerance)):
                is_same = _not_true_print()
        # find the attributes in one but not the other, if any, then this test fails
        diff = attrs1 ^ attrs2
        for this_attr in sorted(diff):
            if exclude is not None and this_attr in exclude:
                # TODO: Should this case be allowed, or is it still a failure?
                continue
            if is_subset and this_attr in attrs2:
                # if only checking that c1 is a subset of c2, then skip this condition
                continue
            is_same = False
            if not suppress_output:
                if this_attr in attrs1:
                    print(f"{this_attr} is only in {name1}.")
                else:
                    print(f"{this_attr} is only in {name2}.")
    # display results
    if not suppress_output:
        if is_same:
            subset_text = " (subset)" if is_subset else ""
            print(f'"{name1}" and "{name2}" are the same{subset_text}.')
        else:
            print(f'"{name1}" and "{name2}" are not the same.')
    return is_same


# %% Functions - compare_two_dicts
def compare_two_dicts(  # noqa: C901
    d1: Mapping[Any, Any],
    d2: Mapping[Any, Any],
    /,
    *,
    suppress_output: bool = False,
    names: tuple[str, str] | list[str] | None = None,
    is_subset: bool = False,
    tolerance: float | None = None,
    exclude: set[str] | None = None,
) -> bool:
    r"""
    Compare two dictionaries for the same keys, and the same value of those keys.

    Parameters
    ----------
    d1 : class object
        Any class object
    d2 : class object
        Any other class object
    suppress_output : bool, optional
        If True, suppress the information printed to the screen, defaults to False.
    names : list of str, optional
        List of the names to be printed to the screen for the two input classes.
    is_subset : bool, optional
        If True, only compares in c1 is a strict subset of c2, but c2 can have extra fields, defaults to False
    tolerance : float, optional
        Numerical tolerance used to compare two numbers that are close together to consider them equal

    Returns
    -------
    is_same : bool
        True/False flag for whether the two class are the same.

    Examples
    --------
    >>> from dstauffman import compare_two_dicts
    >>> d1 = {"a": 1, "b": 2, "c": 3}
    >>> d2 = {"a": 1, "b": 5, "d": 6}
    >>> is_same = compare_two_dicts(d1, d2)
    b is different.
    c is only in d1.
    d is only in d2.
    "d1" and "d2" are not the same.

    """
    # preallocate answer to True until proven otherwise
    is_same = True
    # get names if specified
    if names is not None:
        name1 = names[0]
        name2 = names[1]
    else:
        name1 = "d1"
        name2 = "d2"
    # simple test
    if d1 is not d2:
        # compare the keys that are in both
        same = set(d1) & set(d2)
        for key in sorted(same):
            if exclude is not None and key in exclude:
                continue
            s1 = d1[key]
            s2 = d2[key]
            if isinstance(s1, dict) and isinstance(s2, dict):
                is_same = compare_two_dicts(
                    s1,
                    s2,
                    suppress_output=suppress_output,
                    names=[f"{name1}['{key}']", f"{name2}['{key}']"],
                    is_subset=is_subset,
                    tolerance=tolerance,
                )
            # if any differences, then this test fails
            elif logical_not(_nan_equal(s1, s2, tolerance=tolerance)):
                is_same = False
                if not suppress_output:
                    print(f"{key} is different.")
        # find keys in one but not the other, if any, then this test fails
        diff = set(d1) ^ set(d2)
        for key in sorted(diff):
            if exclude is not None and key in exclude:
                # TODO: Should this case be allowed, or is it still a failure?
                continue
            if is_subset and key in d2:
                # if only checking that d1 is a subset of d2, then skip this condition
                continue
            is_same = False
            if not suppress_output:
                if key in d1:
                    print(f"{key} is only in {name1}.")
                else:
                    print(f"{key} is only in {name2}.")
    # display results
    if not suppress_output:
        if is_same:
            subset_text = " (subset)" if is_subset else ""
            print(f'"{name1}" and "{name2}" are the same{subset_text}.')
        else:
            print(f'"{name1}" and "{name2}" are not the same.')
    return is_same


# %% Functions - magnitude
def magnitude(data: _Lists, axis: int = 0) -> np.floating | _N:
    r"""
    Return a vector of magnitudes for each subvector along a specified dimension.

    Parameters
    ----------
    data : ndarray
        Data
    axis : int, optional
        Axis upon which to normalize, defaults to first axis (i.e. column normalization for 2D matrices)

    Returns
    -------
    norm_data : ndarray
        Normalized data

    See Also
    --------
    sklearn.preprocessing.normalize

    Notes
    -----
    #.  Written by David C. Stauffer in July 2021.

    Examples
    --------
    >>> from dstauffman import magnitude
    >>> import numpy as np
    >>> data = np.array([[1, 0, -1], [0, 0, 0], [0, 0, 1]])
    >>> mag = magnitude(data, axis=0)
    >>> with np.printoptions(precision=8):
    ...     print(mag)  # doctest: +NORMALIZE_WHITESPACE
    [1. 0. 1.41421356]

    """
    if isinstance(data, (list, tuple)):
        data = np.vstack(data).T
    assert isinstance(data, np.ndarray)
    if axis >= data.ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {data.ndim}")
    return np.sqrt(np.sum(data * np.conj(data), axis=axis))  # type: ignore[no-any-return]


# %% Functions - unit
def unit(data: _Lists, axis: int = 0) -> _N:
    r"""
    Normalize a matrix into unit vectors along a specified dimension.

    Parameters
    ----------
    data : numpy.ndarray
        Data
    axis : int, optional
        Axis upon which to normalize, defaults to first axis (i.e. column normalization for 2D matrices)

    Returns
    -------
    norm_data : numpy.ndarray
        Normalized data

    See Also
    --------
    sklearn.preprocessing.normalize

    Notes
    -----
    #.  Written by David C. Stauffer in May 2015.

    Examples
    --------
    >>> from dstauffman import unit
    >>> import numpy as np
    >>> data = np.array([[1, 0, -1], [0, 0, 0], [0, 0, 1]])
    >>> norm_data = unit(data, axis=0)
    >>> with np.printoptions(precision=8):
    ...     print(norm_data)  # doctest: +NORMALIZE_WHITESPACE
    [[ 1. 0. -0.70710678]
     [ 0. 0.  0.        ]
     [ 0. 0.  0.70710678]]

    """
    if isinstance(data, (list, tuple)):
        data = np.vstack(data).T
    assert isinstance(data, np.ndarray)
    if axis >= data.ndim:
        raise ValueError(f"axis {axis} is out of bounds for array of dimension {data.ndim}")
    # calculate the magnitude of each vector
    mag = np.asanyarray(magnitude(data, axis=axis))
    # check for zero vectors, and replace magnitude with 1 to make them unchanged
    mag[mag == 0] = 1
    # calculate the new normalized data
    norm_data: _N = data / mag
    return norm_data


# %% modd
@overload
def modd(x1: ArrayLike, x2: ArrayLike, /) -> None: ...
@overload
def modd(x1: ArrayLike, x2: ArrayLike, /, out: Literal[None]) -> None: ...
@overload
def modd(x1: ArrayLike, x2: ArrayLike, /, out: _I) -> _I: ...
@overload
def modd(x1: ArrayLike, x2: ArrayLike, /, out: _N) -> _N: ...
def modd(x1: ArrayLike, x2: ArrayLike, /, out: _I | _N | None = None) -> _I | _N | None:
    r"""
    Return element-wise remainder of division, except that instead of zero it gives the divisor instead.

    Parameters
    ----------
    x1 : array_like
        Dividend array.
    x2 : array_like
        Divisor array.
    out : ndarray, optional
        Array into which the output is placed. Its type is preserved and it must be of the right
        shape to hold the output. See doc.ufuncs.

    Returns
    -------
    y : ndarray
        The remainder of the quotient x1/x2, element-wise. Returns a scalar if both x1 and x2 are
        scalars.  Replaces what would be zeros in the normal modulo command with the divisor instead.

    Notes
    -----
    #.  Written by David C. Stauffer in October 2015.

    Examples
    --------
    >>> from dstauffman import modd
    >>> import numpy as np
    >>> x1 = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    >>> x2 = 4
    >>> y = modd(x1, x2)
    >>> print(y)
    [4 1 2 3 4 1 2 3 4]

    """
    x1 = np.asanyarray(x1)
    if out is None:
        y = np.mod(x1 - 1, x2) + 1
        return y  # type: ignore[no-any-return]
    np.mod(x1 - 1, x2, out)
    np.add(out, 1, out)  # needed to force add to be inplace operation
    return out


# %% is_np_int
def is_np_int(x: Any, /) -> bool:
    r"""
    Returns True if the input is an int or any form of an np.integer type.

    Parameters
    ----------
    x : int, float or ndarray
        Input value

    Returns
    -------
    bool
        Whether input is an integer type

    Examples
    --------
    >>> from dstauffman import is_np_int
    >>> import numpy as np
    >>> print(is_np_int(1))
    True

    >>> print(is_np_int(1.))
    False

    >>> print(is_np_int(np.array([1, 2])))
    True

    >>> print(is_np_int(np.array([1., 2.])))
    False

    >>> print(is_np_int(np.array(2**62)))
    True

    """
    if isinstance(x, int) or (hasattr(x, "dtype") and np.issubdtype(x.dtype, np.integer)):
        return True
    return False


# %% np_digitize
def np_digitize(x: ArrayLike, /, bins: ArrayLike, right: bool = False) -> _I:
    r"""
    Act as a wrapper to the numpy.digitize function with customizations.

    The customizations include additional error checks, and bins starting from 0 instead of 1.

    Parameters
    ----------
    x : array_like
        Input array to be binned.
    bins : array_like
        Array of bins. It has to be 1-dimensional and monotonic.
    right : bool, optional
        Indicating whether the intervals include the right or the left bin
        edge. Default behavior is (right==False) indicating that the interval
        does not include the right edge. The left bin end is closed in this
        case, i.e., bins[i-1] <= x < bins[i] is the default behavior for
        monotonically increasing bins.

    Returns
    -------
    out : ndarray of ints
        Output array of indices, of same shape as `x`.

    Raises
    ------
    ValueError
        If `bins` is not monotonic.
    TypeError
        If the type of the input is complex.

    See Also
    --------
    numpy.digitize

    Notes
    -----
    #.  This function is equilavent to the MATLAB `discretize` function.

    Examples
    --------
    >>> from dstauffman import np_digitize
    >>> import numpy as np
    >>> x    = np.array([0.2, 6.4, 3.0, 1.6])
    >>> bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
    >>> out  = np_digitize(x, bins)
    >>> print(out)
    [0 3 2 1]

    """
    # allow an empty x to pass through just fine
    if np.size(x) == 0:
        return np.array([], dtype=int)

    # check for NaNs
    if np.any(np.isnan(x)):
        raise ValueError("Some values were NaN.")

    # check the bounds
    tolerance: int | float | None = None  # TODO: do I need a tolerance here?
    bmin = bins[0] if tolerance is None else bins[0] - tolerance  # type: ignore[index, operator]
    bmax = bins[-1] if tolerance is None else bins[-1] + tolerance  # type: ignore[index, operator]
    bad_bounds = False
    if right:
        if np.any(x <= bmin) or np.any(x > bmax):  # type: ignore[operator]
            bad_bounds = True
            bad_left = np.flatnonzero(x <= bmin)  # type: ignore[operator]
            bad_right = np.flatnonzero(x > bmax)  # type: ignore[operator]
    else:
        if np.any(x < bmin) or np.any(x >= bmax):  # type: ignore[operator]
            bad_bounds = True
            bad_left = np.flatnonzero(x < bmin)  # type: ignore[operator]
            bad_right = np.flatnonzero(x >= bmax)  # type: ignore[operator]
    if bad_bounds:
        message = f"Some values ({len(bad_left)} left, {len(bad_right)} right) of x are outside the given bins ([{bmin}, {bmax}])."  # type: ignore[str-bytes-safe]
        if bad_left.size > 0:
            message += f" Such as {np.atleast_1d(x)[bad_left[0]]}"
            if bad_right.size > 0:
                message += f" and {np.atleast_1d(x)[bad_right[0]]}"
        elif bad_right.size > 0:
            message += f" Such as {np.atleast_1d(x)[bad_right[0]]}"
        raise ValueError(message)

    # do the calculations by calling the numpy command and shift results by one
    out = np.digitize(x, bins, right) - 1  # type: ignore[arg-type]
    return out


# %% histcounts
def histcounts(x: ArrayLike, /, bins: ArrayLike, right: bool = False) -> _I:
    r"""
    Count the number of points in each of the given bins.

    Parameters
    ----------
    x : array_like
        Input array to be binned.
    bins : array_like
        Array of bins. It has to be 1-dimensional and monotonic.
    right : bool, optional
        Indicating whether the intervals include the right or the left bin
        edge. Default behavior is (right==False) indicating that the interval
        does not include the right edge. The left bin end is open in this
        case, i.e., bins[i-1] <= x < bins[i] is the default behavior for
        monotonically increasing bins.

    Returns
    -------
    hist : ndarray of ints
        Output array the number of points in each bin

    See Also
    --------
    numpy.digitize, np_digitize

    Notes
    -----
    #.  This function is equilavent to the MATLAB `histcounts` function.

    Examples
    --------
    >>> from dstauffman import histcounts
    >>> import numpy as np
    >>> x    = np.array([0.2, 6.4, 3.0, 1.6, 0.5])
    >>> bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])
    >>> hist = histcounts(x, bins)
    >>> print(hist)
    [2 1 1 1]

    """
    # get the bin number that each point is in
    ix_bin = np_digitize(x, bins, right=right)
    # count the number in each bin
    hist = np.bincount(ix_bin, minlength=len(bins) - 1)  # type: ignore[arg-type]
    return hist


# %% full_print
@contextmanager
def full_print(**kwargs: Unpack[_PrintOptsKwArgs]) -> Generator[None, None, None]:
    r"""
    Context manager for printing full numpy arrays.

    Parameters
    ----------
    kwargs : dict, optional
        Arguments that will be passed through to the np.set_printoptions function

    Notes
    -----
    #.  Adapted by David C. Stauffer in January 2017 from a stackover flow answer by Paul Price,
        given here: http://stackoverflow.com/questions/1987694/print-the-full-numpy-array
    #.  Updated by David C. Stauffer in August 2020 to allow arbitrary arguments to pass through.

    Examples
    --------
    >>> from dstauffman import full_print
    >>> import numpy as np
    >>> temp_options = np.get_printoptions()
    >>> np.set_printoptions(threshold=10)
    >>> a = np.zeros((10, 5))
    >>> a[3, :] = 1.23
    >>> print(a)  # doctest: +NORMALIZE_WHITESPACE
    [[0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]
     ...
     [0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0.]]

    >>> with full_print():
    ...     print(a)
    [[0.   0.   0.   0.   0.  ]
     [0.   0.   0.   0.   0.  ]
     [0.   0.   0.   0.   0.  ]
     [1.23 1.23 1.23 1.23 1.23]
     [0.   0.   0.   0.   0.  ]
     [0.   0.   0.   0.   0.  ]
     [0.   0.   0.   0.   0.  ]
     [0.   0.   0.   0.   0.  ]
     [0.   0.   0.   0.   0.  ]
     [0.   0.   0.   0.   0.  ]]

    >>> np.set_printoptions(**temp_options)

    """
    # get the desired threshold, default is all elements
    threshold = kwargs.pop("threshold", sys.maxsize)
    # get current options
    opt = np.get_printoptions()
    # update to print all elements and any other criteria specified
    np.set_printoptions(threshold=threshold, **kwargs)  # type: ignore[arg-type, misc]
    # yield options for the context manager to do it's thing
    yield
    # reset the options back to what they were originally
    np.set_printoptions(**opt)


# %% combine_per_year
@overload
def combine_per_year(data: None, func: Callable[..., Any]) -> None: ...
@overload
def combine_per_year(data: _Array, func: Callable[..., Any]) -> _Array: ...
def combine_per_year(data: _Array | None, func: Callable[..., Any] | None = None) -> _Array | None:
    r"""
    Combine the time varying values over one year increments using a supplied function.

    Parameters
    ----------
    data : ndarray, 1D or 2D
        Data array

    Returns
    -------
    data2 : ndarray, 1D or 2D
        Data array combined as mean over 12 month periods

    Notes
    -----
    #.  Written by David C. Stauffer in October 2015.
    #.  Made more generic by David C. Stauffer in August 2017.
    #.  This function was designed with np.nanmean and np.nansum in mind.

    Examples
    --------
    >>> from dstauffman import combine_per_year
    >>> import numpy as np
    >>> time = np.arange(120)
    >>> data = np.sin(time)
    >>> data2 = combine_per_year(data, func=np.mean)

    """
    # check that a function was provided
    if func is None or not callable(func):
        raise AssertionError("A callable function must be provided.")
    # check for null case and exit
    if data is None:
        return None
    # check dimensionality
    is_1d = data.ndim == 1
    # get original sizes
    if is_1d:
        data = data[:, np.newaxis]
    assert isinstance(data, np.ndarray)
    (num_time, num_chan) = data.shape
    num_year = num_time // MONTHS_PER_YEAR
    # check for case with all NaNs
    if np.all(np.isnan(data)):
        data2 = np.full((num_year, num_chan), np.nan)
    else:
        # disables warnings for time points that are all NaNs for nansum or nanmean
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            warnings.filterwarnings("ignore", message="Sum of empty slice")
            # calculate sum or mean (or whatever)
            data2 = func(np.reshape(data[: num_year * MONTHS_PER_YEAR, :], (num_year, MONTHS_PER_YEAR, num_chan)), axis=1)
    # optionally squeeze the vector case back to 1D
    if is_1d:
        data2 = data2.squeeze(axis=1)
    return data2


# %% Functions - execute
def execute(
    command: str | list[str],
    folder: Path,
    *,
    ignored_codes: Iterable[int] | None = None,
    env: dict[str, str] | None = None,
) -> Generator[str, None, int]:
    r"""
    Wrapper to subprocess that allows the screen to be updated for long running commands.

    Parameters
    ----------
    command : str or list of str
        Command to execute
    folder : class pathlib.Path
        Path to execute the command in
    ignored_codes : iterable of int, optional
        If given, a list of non-zero error codes to ignore
    env : dict
        Dictionary of environment variables to update for the call

    Returns
    -------
    rc : ReturnCodes enum
        return code from running the command

    Notes
    -----
    #.  Written by David C. Stauffer in October 2019.
    #.  Updated by David C. Stauffer in April 2021 to use pathlib.

    Examples
    --------
    >>> from dstauffman import execute
    >>> import pathlib
    >>> command = "ls"
    >>> folder  = pathlib.Path.cwd()
    >>> # Note that this command may not work right within the IPython console, it's intended for command windows.
    >>> execute(command, folder)  # doctest: +SKIP

    """
    # overlay environment variables
    if env is not None:
        env = os.environ.copy().update(env)

    # create a process to spawn the thread
    popen = subprocess.Popen(  # pylint: disable=consider-using-with
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=None,
        cwd=folder,
        shell=False,
        universal_newlines=True,
        env=env,
    )
    # intermittenly read output lines
    yield from iter(popen.stdout.readline, "")  # type: ignore[union-attr]
    # once done, close and get return codes
    popen.stdout.close()  # type: ignore[union-attr]
    return_code = popen.wait()

    # method 2
    # while True:
    #     output = process.stdout.readline()
    #     if output == "" and process.poll() is not None:
    #         break
    #     if output:
    #         yield output
    # process.stdout.close()
    # return_code = process.poll()

    # determine if command exited cleanly or not and return appropriate code
    if return_code:
        if ignored_codes is None or return_code not in ignored_codes:
            # raise subprocess.CalledProcessError(return_code, command)
            return ReturnCodes.bad_command
    return ReturnCodes.clean


# %% Functions - execute_wrapper
def execute_wrapper(
    command: str | list[str],
    folder: Path,
    *,
    dry_run: bool = False,
    ignored_codes: Iterable[int] | None = None,
    filename: Path | None = None,
    append: bool = False,
    env: dict[str, str] | None = None,
    print_status: bool = True,
) -> ReturnCodes | list[str]:
    r"""
    Wrapper to the wrapper to subprocess with options to print the command do dry-runs.

    Parameters
    ----------
    command : str or list of str
        Command to execute
    folder : class pathlib.Path
        Path to execute the command in
    dry_run : bool, optional, default is False
        Whether the command should be displayed or actually run
    ignored_codes : int or iterable of int, optional, default is None
        If given, a list of non-zero error codes to ignore
    filename : class pathlib.Path, optional, default is to not write
        Name of the file to write the output to, ignore if empty string
    append : bool, optional, default is False
        Whether to append to the given filename if it already exists
    env : dict, optional
        Dictionary of environment variables to update for the call
    print_status : bool, optional, default is True
        Whether to print the status of the command to standard output

    Notes
    -----
    #.  Written by David C. Stauffer in November 2019.
    #.  Updated by David C. Stauffer in April 2021 to use pathlib.

    Examples
    --------
    >>> from dstauffman import execute_wrapper
    >>> import pathlib
    >>> command = "ls"
    >>> folder  = pathlib.Path.cwd()
    >>> dry_run = True
    >>> rc = execute_wrapper(command, folder, dry_run=dry_run)  # doctest: +ELLIPSIS
    Would execute "ls" in "..."

    """
    # simple dry run case, just display what would happen
    if dry_run:
        if isinstance(command, list):
            command = shlex.join(command)
        print(f'Would execute "{command}" in "{folder}"')
        return ReturnCodes.clean
    # clean up command
    if isinstance(command, str):
        command_list = shlex.split(command)
    elif isinstance(command, list):
        command_list = command
    else:
        raise TypeError("Unexpected type for the command list.")
    # check that the folder exists
    if not folder.is_dir():
        print(f'Warning: folder "{folder}" doesn\'t exist, so command "{command}" was not executed.')
        return ReturnCodes.bad_folder
    # execute command and print status
    assert print_status or filename is not None, "You must either print the status or save results to a filename."
    if print_status:
        lines = []
        for line in execute(command_list, folder, ignored_codes=ignored_codes, env=env):
            # print each line as it comes so you can review long running commands as they execute
            print(line, end="")
            lines.append(line)
    else:
        lines = list(execute(command_list, folder, ignored_codes=ignored_codes, env=env))
    # optionally write to text file if a filename is given
    if filename is not None:
        write_text_file(filename, "".join(lines), append=append)
    return lines


# %% Functions - get_env_var
def get_env_var(env_key: str, default: str | None = None) -> str:
    r"""
    Return an environment variable assuming is has been set.

    Parameters
    ----------
    env_key : str
        Environment variable to try and retrieve.
    default : str, optional
        Default value to use if the variable doesn't exist, if None, an error is raised

    Returns
    -------
    value : str
        Value of the given environment variable

    Notes
    -----
    #.  Written by Alex Kershetsky in November 2019.
    #.  Incorporated into dstauffman tools by David C. Stauffer in January 2020.

    Examples
    --------
    >>> from dstauffman import get_env_var
    >>> value = get_env_var("HOME")

    """
    if _ALLOWED_ENVS is not None:
        if env_key not in _ALLOWED_ENVS:
            raise KeyError(f'The environment variable of "{env_key}" is not on the allowed list.')
    try:
        value = os.environ[env_key]
    except KeyError:
        if default is None:
            raise KeyError(f'The appropriate environment variable "{env_key}" has not been set.') from None
        value = default
    return value


# %% Functions - get_username
def get_username() -> str:
    r"""
    Gets the current username based on environment variables.

    Returns
    -------
    username : str
        Name of the username

    Notes
    -----
    #.  Written by David C. Stauffer in August 2020.

    Examples
    --------
    >>> from dstauffman import get_username
    >>> username = get_username()

    """
    if IS_WINDOWS:
        return os.environ["USERNAME"]
    try:
        return os.environ["USER"]
    except KeyError:  # pragma: no cover
        return os.environ["GITLAB_USER_LOGIN"]


# %% Functions - is_datetime
def is_datetime(time: datetime.datetime | ArrayLike | None) -> bool:
    r"""
    Determines if the given time is either a datetime.datetime or np.datetime64 or just a regular number.

    Parameters
    ----------
    time : float
        Time

    Returns
    -------
    out : bool
        Whether this is a datetime

    Notes
    -----
    #.  Written by David C. Stauffer in May 2020.

    Examples
    --------
    >>> from dstauffman import is_datetime
    >>> import datetime
    >>> import numpy as np
    >>> time1 = 0.5
    >>> time2 = np.datetime64("now")
    >>> time3 = datetime.datetime.now()
    >>> print(is_datetime(time1))
    False

    >>> print(is_datetime(time2))
    True

    >>> print(is_datetime(time3))
    True

    """
    out = False
    if isinstance(time, datetime.datetime) or (hasattr(time, "dtype") and np.issubdtype(time.dtype, np.datetime64)):  # type: ignore[union-attr]
        out = True
    return out


# %% Functions - intersect
@overload
def intersect(a: ArrayLike, b: ArrayLike, /, *, return_index: Literal[False] = ...) -> _I | _N | _D: ...
@overload
def intersect(a: ArrayLike, b: ArrayLike, /, *, return_index: Literal[True]) -> tuple[_I | _N | _D, _I, _I]: ...
@overload
def intersect(
    a: ArrayLike,
    b: ArrayLike,
    /,
    *,
    tolerance: int | float | np.timedelta64,
    assume_unique: bool,
    return_index: Literal[False] = ...,
) -> _I | _N | _D: ...
@overload
def intersect(
    a: ArrayLike,
    b: ArrayLike,
    /,
    *,
    tolerance: int | float | np.timedelta64,
    assume_unique: bool,
    return_index: Literal[True],
) -> tuple[_I | _N | _D, _I, _I]: ...
def intersect(  # type: ignore[misc]
    a: ArrayLike,
    b: ArrayLike,
    /,
    *,
    tolerance: int | float | np.timedelta64 = 0,
    assume_unique: bool = False,
    return_indices: bool = False,
) -> _I | _N | _D | tuple[_I | _N | _D, _I, _I]:
    r"""
    Finds the intersect of two arrays given a numerical tolerance.

    Return the sorted, unique values that are in both of the input arrays.

    Parameters
    ----------
    a, b : array_like
        Input arrays. Will be flattened if not already 1D.
    tolerance : float or int
        Tolerance for which something is considered unique
    assume_unique : bool
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.
    return_indices : bool
        If True, the indices which correspond to the intersection of the two
        arrays are returned. The first instance of a value is used if there are
        multiple. Default is False.

    Returns
    -------
    c : ndarray
        Sorted 1D array of common and unique elements.
    ia : ndarray
        The indices of the first occurrences of the common values in `ar1`.
        Only provided if `return_indices` is True.
    ib : ndarray
        The indices of the first occurrences of the common values in `ar2`.
        Only provided if `return_indices` is True.

    See Also
    --------
    numpy.intersect1d : Function used to do comparsion with sets of quantized inputs.

    Notes
    -----
    #.  Written by David C. Stauffer in March 2019.
    #.  Updated by David C. Stauffer in June 2020 to allow for a numeric tolerance.

    Examples
    --------
    >>> from dstauffman import intersect
    >>> import numpy as np
    >>> a = np.array([1, 2, 4, 4, 6], dtype=int)
    >>> b = np.array([0, 8, 2, 2, 5, 8, 6, 8, 8], dtype=int)
    >>> (c, ia, ib) = intersect(a, b, return_indices=True)
    >>> print(c)
    [2 6]

    >>> print(ia)
    [1 4]

    >>> print(ib)
    [2 6]

    """
    # allow a zero tolerance to be passed in and behave like the normal intersect command
    if hasattr(tolerance, "dtype") and np.issubdtype(tolerance.dtype, np.timedelta64):
        tol_is_zero = (
            tolerance.astype(np.int64) == 0  # type: ignore[union-attr]
        )  # Note that this avoids a numpy bug, see issue 6784  # type: ignore[union-attr]
    else:
        tol_is_zero = tolerance == 0
    if tol_is_zero:
        return np.intersect1d(a, b, assume_unique=assume_unique, return_indices=return_indices)  # type: ignore[call-overload, no-any-return]

    # allow list and other array_like inputs (or just scalar floats)
    a = np.atleast_1d(np.asanyarray(a))
    b = np.atleast_1d(np.asanyarray(b))
    tolerance = np.asanyarray(tolerance)  # type: ignore[assignment]

    # check for datetimes and convert to integers
    is_dates = np.array([is_datetime(a), is_datetime(b)], dtype=bool)
    assert np.count_nonzero(is_dates) != 1, "Both arrays must be datetimes if either is."
    if np.any(is_dates):
        orig_datetime = a.dtype
        a = a.astype(np.int64)
        b = b.astype(np.int64)
        tolerance = tolerance.astype(np.int64)  # type: ignore[assignment, union-attr]

    # check if largest component of a and b is too close to the tolerance floor (for floats)
    all_int = is_np_int(a) and is_np_int(b) and is_np_int(tolerance)
    max_a_or_b = np.max((np.max(np.abs(a), initial=0), np.max(np.abs(b), initial=0)))
    if not all_int and ((max_a_or_b / tolerance) > (0.01 / np.finfo(float).eps)):
        warnings.warn("This function may have problems if tolerance gets too small.")

    # due to the splitting of the quanta, two very close numbers could still fail the quantized intersect
    # fix this by repeating the comparison when shifted by half a quanta in either direction
    half_tolerance = tolerance / 2
    if all_int:
        # allow for integer versions of half a quanta in either direction
        lo_tol = np.floor(half_tolerance).astype(tolerance.dtype)  # type: ignore[union-attr]
        hi_tol = np.ceil(half_tolerance).astype(tolerance.dtype)  # type: ignore[union-attr]
    else:
        lo_tol = half_tolerance
        hi_tol = half_tolerance

    # create quantized version of a & b, plus each one shifted by half a quanta
    a1 = np.floor_divide(a, tolerance)
    b1 = np.floor_divide(b, tolerance)
    a2 = np.floor_divide(a - lo_tol, tolerance)
    b2 = np.floor_divide(b - lo_tol, tolerance)
    a3 = np.floor_divide(a + hi_tol, tolerance)
    b3 = np.floor_divide(b + hi_tol, tolerance)

    # do a normal intersect on the quantized data for different comparisons
    (_, ia1, ib1) = np.intersect1d(a1, b1, assume_unique=assume_unique, return_indices=True)
    (_, ia2, ib2) = np.intersect1d(a1, b2, assume_unique=assume_unique, return_indices=True)
    (_, ia3, ib3) = np.intersect1d(a1, b3, assume_unique=assume_unique, return_indices=True)
    (_, ia4, ib4) = np.intersect1d(a2, b1, assume_unique=assume_unique, return_indices=True)
    (_, ia5, ib5) = np.intersect1d(a3, b1, assume_unique=assume_unique, return_indices=True)

    # combine the results
    ia = reduce(np.union1d, [ia1, ia2, ia3, ia4, ia5])
    ib = reduce(np.union1d, [ib1, ib2, ib3, ib4, ib5])

    # calculate output
    # Note that a[ia] and b[ib] should be the same with a tolerance of 0, but not necessarily otherwise
    # This function returns the values from the first vector a
    c = np.sort(a[ia])
    if np.any(is_dates):
        c = c.astype(orig_datetime)
    if return_indices:
        return (c, ia, ib)
    return c


# %% issorted
def issorted(x: ArrayLike, /, descend: bool = False) -> bool:
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
    >>> from dstauffman import issorted
    >>> x = np.array([1, 3, 3, 5, 7])
    >>> print(issorted(x))
    True

    >>> y = np.array([3, 5, 1, 7])
    >>> print(issorted(y))
    False

    """
    x = np.asanyarray(x)
    if descend:
        return np.all(x[1:] <= x[:-1])  # type: ignore[return-value]
    return np.all(x[:-1] <= x[1:])  # type: ignore[return-value]


# %% zero_order_hold
def zero_order_hold(
    x: ArrayLike,
    xp: ArrayLike,
    yp: ArrayLike,
    *,
    left: int | float | str = nan,
    assume_sorted: bool = False,
    return_indices: bool = False,
) -> _N:
    r"""
    Interpolates a function by holding at the most recent value.

    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    xp: 1-D sequence of floats
        The x-coordinates of the data points, must be increasing if argument period is not specified. Otherwise, xp is internally sorted after normalizing the periodic boundaries with xp = xp % period.
    yp: 1-D sequence of float or complex
        The y-coordinates of the data points, same length as xp.
    left: int or float, optional, default is np.nan
        Value to use for any value less that all points in xp
    assume_sorted : bool, optional, default is False
        Whether you can assume the data is sorted and do simpler (i.e. faster) calculations

    Returns
    -------
    y : float or complex (corresponding to fp) or ndarray
        The interpolated values, same shape as x.

    Notes
    -----
    #.  Written by David C. Stauffer in July 2020.

    Examples
    --------
    >>> from dstauffman import zero_order_hold
    >>> import numpy as np
    >>> xp = np.array([0., 111., 2000., 5000.])
    >>> yp = np.array([0, 1, -2, 3])
    >>> x = np.arange(0.0, 6001.0)
    >>> y = zero_order_hold(x, xp, yp)

    """
    # force arrays
    x = np.asanyarray(x)
    xp = np.asanyarray(xp)
    yp = np.asanyarray(yp)
    # find the minimum value, as anything left of this is considered extrapolated
    xmin = xp[0] if assume_sorted else np.min(xp)
    # check that xp data is sorted, if not, use slower scipy version
    if assume_sorted or issorted(xp):
        ix = np.searchsorted(xp, x, side="right") - 1
        is_left = np.asanyarray(x) < xmin
        out = np.where(is_left, left, yp[ix])
        if return_indices:
            return (out, np.where(is_left, None, ix))  # type: ignore[call-overload, return-value]
        return out
    if not HAVE_SCIPY:
        raise RuntimeError("You must have scipy available to run this.")
    if return_indices:
        raise RuntimeError("Data must be sorted in order to ask for indices.")
    func = interp1d(xp, yp, kind="zero", fill_value="extrapolate", assume_sorted=False)
    return np.where(np.asanyarray(x) < xmin, left, func(x).astype(yp.dtype))


# %% linear_interp
def linear_interp(
    x: ArrayLike,
    xp: ArrayLike,
    yp: ArrayLike,
    *,
    left: int | float | None = None,
    right: int | float | None = None,
    assume_sorted: bool = False,
    extrapolate: bool = False,
) -> _N:
    r"""
    Interpolates a function using linear interpolation.

    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    xp: 1-D sequence of floats
        The x-coordinates of the data points, must be increasing if argument period is not specified. Otherwise, xp is internally sorted after normalizing the periodic boundaries with xp = xp % period.
    yp: 1-D sequence of float or complex
        The y-coordinates of the data points, same length as xp
    left: int or float, optional, default is yp[0]
        Value to use for any value less than all points in xp
    right: int or float, optional, default is yp[-1]
        Value to use for any value greater than all points in xp
    assume_sorted : bool, optional, default is False
        Whether you can assume the data is sorted and do simpler (i.e. faster) calculations
    extrapolate : bool, optional, default is False
        Whether to allow function to extrapolate data on either end

    Returns
    -------
    y : float or complex (corresponding to yp) or ndarray
        The interpolated values, same shape as x.

    Notes
    -----
    #.  Written by Steven Hiramoto in June 2022.

    Examples
    --------
    >>> from dstauffman import linear_interp
    >>> import numpy as np
    >>> xp = np.array([0.0, 111.0, 2000.0, 5000.0])
    >>> yp = np.array([0.0, 1.0, -2.0, 3.0])
    >>> x = np.arange(0.0, 6001.0)
    >>> y = linear_interp(x, xp, yp, extrapolate=True)

    """
    # force arrays
    x = np.asanyarray(x)
    xp = np.asanyarray(xp)
    yp = np.asanyarray(yp)
    # use simpler numpy version if data is sorted
    if assume_sorted or issorted(xp):
        if not extrapolate:
            # checks the bounds for any bad data
            if np.any(x < xp[0]) or np.any(x > xp[-1]):
                raise ValueError("Desired points outside given xp array and extrapolation is False")
        out = np.interp(x, xp, yp, left=left, right=right)
        return out  # type: ignore[no-any-return]
    # use slower scipy version
    if not HAVE_SCIPY:
        raise RuntimeError("You must have scipy available to run this.")
    fill_value: str | tuple[int, int] | tuple[float, float] | None
    if extrapolate:
        bounds_error = False
        if left is None or right is None:
            fill_value = "extrapolate"
        else:
            fill_value = (left, right)
    else:
        bounds_error = True
        fill_value = None
    func = interp1d(xp, yp, kind="linear", fill_value=fill_value, bounds_error=bounds_error, assume_sorted=False)
    return func(x).astype(yp.dtype)  # type: ignore[no-any-return]


# %% linear_lowpass_interp
def linear_lowpass_interp(
    x: ArrayLike,
    xp: ArrayLike,
    yp: ArrayLike,
    *,
    assume_sorted: bool = False,
    extrapolate: bool = False,
    filt_order: int = 2,
    filt_freq: float = 0.01,
    filt_samp: float = 1.0,
    **kwargs: Unpack[_ButterKwArgs],
) -> _N:
    r"""
    Interpolates a function using linear interpolation along with a configurable low pass filter.

    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    xp: 1-D sequence of floats
        The x-coordinates of the data points, must be increasing if argument period is not specified. Otherwise, xp is internally sorted after normalizing the periodic boundaries with xp = xp % period.
    yp: 1-D sequence of float or complex
        The y-coordinates of the data points, same length as xp.
    assume_sorted : bool, optional, default is False
        Whether you can assume the data is sorted and do simpler (i.e. faster) calculations
    extrapolate : bool, optional, default is False
        Whether to allow function to extrapolate data on either end
    filt_order : int, optional, default is 2
        Low pass filter order
    filt_freq : float, optional, default is 0.01
        Default filter frequency
    filt_samp : float, optional, default is 1.0
        Default filter sample rate
    kwargs : Any
        Additional key-word arguments to pass through to scipy.signal.butter

    Returns
    -------
    y : float or complex (corresponding to yp) or ndarray
        The interpolated values, same shape as x.

    Notes
    -----
    #.  Written by Steven Hiramoto in June 2022.

    Examples
    --------
    >>> from dstauffman import linear_lowpass_interp
    >>> import numpy as np
    >>> xp = np.array([0.0, 111.0, 2000.0, 5000.0])
    >>> yp = np.array([0.0, 1.0, -2.0, 3.0])
    >>> x = np.arange(0.0, 6001.0)
    >>> y = linear_lowpass_interp(x, xp, yp, extrapolate=True)

    """
    # force arrays
    x = np.asanyarray(x)
    xp = np.asanyarray(xp)
    yp = np.asanyarray(yp)
    # must have scipy to execute this function
    if not HAVE_SCIPY:
        raise RuntimeError("You must have scipy available to run this.")
    if extrapolate:
        bounds_error = False
        fill_value = "extrapolate"
    else:
        bounds_error = True
        fill_value = None
    func = interp1d(xp, yp, kind="linear", bounds_error=bounds_error, fill_value=fill_value, assume_sorted=assume_sorted)
    temp = func(x).astype(yp.dtype)
    sos = butter(filt_order, filt_freq, fs=filt_samp, output="sos", **kwargs)
    return sosfiltfilt(sos, temp)  # type: ignore[no-any-return]


# %% drop_following_time
@overload
def drop_following_time(times: _D, drop_starts: _D, dt_drop: np.datetime64) -> _B: ...
@overload
def drop_following_time(times: _I, drop_starts: _I, dt_drop: int) -> _B: ...
@overload
def drop_following_time(times: _N, drop_starts: _N, dt_drop: float) -> _B: ...
@overload
def drop_following_time(times: _D, drop_starts: _D, dt_drop: np.datetime64, *, reverse: bool) -> _B: ...
@overload
def drop_following_time(times: _I, drop_starts: _I, dt_drop: int, *, reverse: bool) -> _B: ...
@overload
def drop_following_time(times: _N, drop_starts: _N, dt_drop: float, *, reverse: bool) -> _B: ...
def drop_following_time(
    times: _D | _I | _N,
    drop_starts: _D | _I | _N,
    dt_drop: int | float | np.datetime64,
    *,
    reverse: bool = False,
) -> _B:
    r"""
    Drops the times within the dt_drop after drop_starts.

    Parameters
    ----------
    times : (N, ) array_like
        Times at which you want to know the drop status
    drop_starts : (M, ) array_like
        Times at which the drops start
    dt_drop : float or int
        Delta time for each drop window

    Returns
    -------
    drop_make : (N, ) ndarray of bool
        Mask where the data points should be dropped

    Notes
    -----
    #.  Written by David C. Stauffer in August 2020.

    Examples
    --------
    >>> from dstauffman import drop_following_time
    >>> import numpy as np
    >>> times = np.arange(50)
    >>> drop_starts = np.array([5, 15, 17, 25])
    >>> dt_drop = 3
    >>> drop_mask = drop_following_time(times, drop_starts, dt_drop)

    """
    # Version with for loop # TODO: would like to do this without the loop
    drop_mask = np.zeros(times.size, dtype=bool)
    for drop_time in drop_starts:
        # drop the times within the specified window
        if reverse:
            drop_mask |= (times > drop_time - dt_drop) & (times <= drop_time)
        else:
            drop_mask |= (times >= drop_time) & (times < drop_time + dt_drop)
    return drop_mask


# %% Unit test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_utils", exit=False)
    doctest.testmod(verbose=False)
