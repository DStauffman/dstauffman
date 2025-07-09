r"""
Functions that make it easier to deal with Matlab data.

Notes
-----
#.  Written by David C. Stauffer in May 2016.  Mostly based on Matlab routines.
#.  Expanded by David C. Stauffer in December 2018 to include load_matlab capabilities.
#.  Combined by David C. Stauffer in August 2022 for what used to be estimation.linalg.

"""

# %% Imports
from __future__ import annotations

import doctest
from pathlib import Path
from typing import Any, Final, overload, TYPE_CHECKING
import unittest

from nubs import ncjit

from dstauffman.constants import HAVE_H5PY, HAVE_NUMPY

if HAVE_H5PY:
    import h5py
if HAVE_NUMPY:
    import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    _B = NDArray[np.bool_]
    _I = NDArray[np.int_]
    _N = NDArray[np.floating]
    _M = NDArray[np.floating]  # 2D

# %% Constants
_EPS: Final = float(np.finfo(float).eps) if HAVE_NUMPY else 2.220446049250313e-16


# %% load_matlab
def load_matlab(  # noqa: C901
    filename: str | Path,
    varlist: list[str] | set[str] | tuple[str, ...] | None = None,
    *,
    squeeze: bool = True,
    enums: dict[str, Any] | None = None,
) -> dict[str, Any]:
    r"""
    Load simple arrays from a MATLAB v7.3 HDF5 based *.mat file.

    Parameters
    ----------
    filename : class pathlib.Path
        Name of the file to load
    varlist : list of str, optional
        Name of the variables to load
    squeeze : bool, optional, default is True
        Whether to squeeze any singleton vectors down a dimension

    Returns
    -------
    out : dict
        Equivalent structure as python dictionary

    Examples
    --------
    >>> from dstauffman import load_matlab, get_tests_dir
    >>> filename = get_tests_dir() / "test_numbers.mat"
    >>> out = load_matlab(filename)
    >>> print(out["row_nums"][1])
    2.2

    """

    def _load(
        file: h5py.Group,
        varlist: list[str] | set[str] | tuple[str, ...] | None,
        squeeze: bool,
        enums: dict[str, Any] | None,
    ) -> dict[str, Any]:
        r"""Wrapped subfunction so it can be called recursively."""
        # initialize output
        out: dict[str, Any] = {}
        # loop through keys, keys are the MATLAB variable names, like TELM
        for key in file:
            # skip keys that are not in the given varlist
            if varlist is not None and key not in varlist:
                continue
            # if no varlist (thus loading every key), still skip those that start with #
            if varlist is None and key in {"#refs#", "#subsystem#"}:
                continue
            # alias this group
            grp = file[key]
            # check if this is a dataset, meaning its just an array and not a structure
            if isinstance(grp, h5py.Dataset):
                # Note: data is transposed due to how Matlab stores columnwise
                values = grp[()].T
                # check for cell array references
                if isinstance(values.flat[0], h5py.Reference):
                    # TODO: for now, always collapse to 1D cell array as a list
                    temp = [file[item] for item in values.flat]
                    temp2 = []
                    for x in temp:
                        if isinstance(x, h5py.Group):
                            temp2.append(load_matlab(x, varlist=None, squeeze=squeeze, enums=enums))
                        else:
                            data = x[()].T
                            temp2.append(np.squeeze(data) if squeeze else data)
                    out[key] = temp2
                else:
                    out[key] = np.squeeze(values) if squeeze else values
            elif "EnumerationInstanceTag" in grp:
                # likely a MATLAB enumerator???
                class_name = grp.attrs["MATLAB_class"].decode()
                if enums is None or class_name not in enums:
                    raise ValueError(
                        f'Tried to load a MATLAB enumeration class called "{class_name}" without a decoder ring, pass in via `enums`.'
                    )
                ix = grp["ValueIndices"][()].T
                values = np.array([enums[class_name][x] for x in ix.flatten()]).reshape(ix.shape)
                out[key] = np.squeeze(values) if squeeze else values
            else:
                # call recursively
                out[key] = load_matlab(grp, varlist=None, squeeze=squeeze, enums=enums)
        return out

    if not isinstance(filename, h5py.Group):
        with h5py.File(filename, "r") as file:
            # normal method
            out = _load(file=file, varlist=varlist, squeeze=squeeze, enums=enums)
    else:
        # recursive call method where the file is already opened to a given group
        out = _load(file=filename, varlist=varlist, squeeze=squeeze, enums=enums)
    return out


# %% orth
def orth(A: _M) -> _M:
    r"""
    Orthogonalization basis for the range of A.

    That is, Q'*Q = I, the columns of Q span the same space as the columns of A, and the number of
    columns of Q is the rank of A.

    Parameters
    ----------
    A : 2D ndarray
        Input matrix

    Returns
    -------
    Q : 2D ndarray
        Orthogonalization matrix of A

    Notes
    -----
    #.  Based on the Matlab orth.m function.

    Examples
    --------
    >>> from dstauffman import orth
    >>> import numpy as np

    Full rank matrix
    >>> A = np.array([[1, 0, 1], [-1, -2, 0], [0, 1, -1]])
    >>> r = np.linalg.matrix_rank(A)
    >>> print(r)
    3

    >>> Q = orth(A)
    >>> with np.printoptions(precision=8):
    ...     print(Q)
    [[-0.12000026 -0.80971228  0.57442663]
     [ 0.90175265  0.15312282  0.40422217]
     [-0.41526149  0.5664975   0.71178541]]

    Rank deficient matrix
    >>> A = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    >>> r = np.linalg.matrix_rank(A)
    >>> print(r)
    2

    >>> Q = orth(A)
    >>> print(Q.round(8) + 0)
    [[-0.70710678  0.        ]
     [ 0.          1.        ]
     [-0.70710678  0.        ]]

    """
    # compute the SVD
    (Q, S, _) = np.linalg.svd(A, full_matrices=False)
    # calculate a tolerance based on the first eigenvalue (instead of just using a small number)
    tol = np.max(A.shape) * S[0] * _EPS
    # sum the number of eigenvalues that are greater than the calculated tolerance
    r = np.count_nonzero(S > tol, axis=0)
    # return the columns corresponding to the non-zero eigenvalues
    Q = Q[:, np.arange(r)]
    return Q


# %% subspace
def subspace(A: _M, B: _M) -> float:
    r"""
    Angle between two subspaces specified by the columns of A and B.

    Parameters
    ----------
    A : 2D ndarray
        Matrix A
    B : 2D ndarray
        Matrix B

    Returns
    -------
    theta : scalar
        Angle between the column subspaces of A and B.

    Notes
    -----
    #.  Based on the Matlab subspace.m function.

    References
    ----------
    #.  A. Bjorck & G. Golub, Numerical methods for computing
        angles between linear subspaces, Math. Comp. 27 (1973), pp. 579-594.
    #.  P.-A. Wedin, On angles between subspaces of a finite
        dimensional inner product space, in B. Kagstrom & A. Ruhe (Eds.),
        Matrix Pencils, Lecture Notes in Mathematics 973, Springer, 1983, pp. 263-285.

    Examples
    --------
    >>> from dstauffman import subspace
    >>> import numpy as np
    >>> A = np.array([[1, 1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1], [1, 1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1]])
    >>> B = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1], [-1, -1, -1, -1], \
    ...     [-1, 1, -1, 1], [-1, -1, 1, 1], [-1, 1, 1, -1]])
    >>> theta = subspace(A, B)
    >>> print("{:.4f}".format(theta))
    1.5708

    """
    # compute orthonormal bases (avoids problems when A and/or B is nearly rank deficient)
    A = orth(A)
    B = orth(B)
    # check rank and swap
    if A.shape[1] > B.shape[1]:
        (B, A) = (A, B)
    # compute the projection according to Ref 1
    B = B - A @ (A.T @ B)
    # make sure it's magnitude is less than 1 and compute arcsin
    theta = np.arcsin(np.minimum(1.0, np.linalg.norm(B)))
    return theta  # type: ignore[no-any-return]


# %% Functions - ecdf
def ecdf(y: float | list[float] | _N, /) -> tuple[_N, _N]:
    r"""
    Calculate the empirical cumulative distribution function, as in Matlab's ecdf function.

    Parameters
    ----------
    array_like of float
        Input samples

    Returns
    -------
    x : ndarray of float
        cumulative probability
    f : ndarray of float
        function values evaluated at the points returned in x

    Notes
    -----
    #.  Written by David C. Stauffer in February 2021.

    Examples
    --------
    >>> from dstauffman import ecdf
    >>> import numpy as np
    >>> y = np.random.default_rng().random(1000)
    >>> (x, f) = ecdf(y)
    >>> exp = np.arange(0.001, 1.001, 0.001)
    >>> print(np.max(np.abs(f - exp)) < 0.05)
    True

    """
    f, counts = np.unique(y, return_counts=True)
    x = np.cumsum(counts) / np.size(y)
    return (x, f)


# %% mat_divide
@ncjit
def mat_divide(a: _M, b: _N, rcond: float = _EPS) -> _N | _M:
    r"""
    Solves the least square solution for x in A*x = b.

    In Matlab, this is: A\b = inv(A)*b
    np.linalg.lstsq(a, b)[0] Computes the vector x that approximatively solves the equation a @ x = b ( or x = inv(a) @ b)
    However, you have to remember the rcond=None to avoid warnings, and the [0] to get the actual solution part you want,
    thus the benefit of having this function.

    Parameters
    ----------
    a : (M, N) array_like
        Coefficient matrix
    b : {(M, ), (M, K)} array_like
        Ordinate or dependent variable values. If b is two-dimensional, the least-squares solution
        is calculated for each of the K columns of b.
    rcond : float, optional
        Cut-off ratio for small singular values of a.

    Returns
    -------
    x {(N, ), (N, K)} ndarray
        Least-squares solution

    See Also
    --------
    numpy.linalg.lstsq

    Notes
    -----
    #.  Written by David C. Stauffer in July 2020.

    Examples
    --------
    >>> from dstauffman import mat_divide
    >>> import numpy as np
    >>> a = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> x = np.array([1.0, -1.0])
    >>> b = a @ x
    >>> out = mat_divide(a, b)
    >>> with np.printoptions(precision=8):
    ...     print(out)
    [ 1. -1.]

    """
    return np.linalg.lstsq(a, b, rcond=rcond)[0]


# %% find_first
def find_first(x: _B, /) -> int:
    r"""
    Finds the location of the first true occurrence in a logical array.

    Return -1 if none are found.
    Returns linear index position for higher order arrays.

    Parameters
    ----------
    x : (N,) array of bool
        Logical array

    Returns
    -------
    int
        Location of first true value, returns -1 if not found

    See Also
    --------
    np.argmax, np.argmin

    Notes
    -----
    #.  Written by David C. Stauffer in May 2023.

    Examples
    --------
    >>> from dstauffman import find_first
    >>> x = np.array([0, 0, 1, 0, 1, 0, 0], dtype=bool)
    >>> ix = find_first(x)
    >>> print(ix)
    2

    """
    if ~np.any(x):
        return -1
    return int(np.argmax(x))


# %% find_last
def find_last(x: _B, /) -> int:
    r"""
    Finds the location of the last true occurrence in a logical array.

    Return -1 if none are found.
    Returns linear index position for higher order arrays.

    Parameters
    ----------
    x : (N,) array of bool
        Logical array

    Returns
    -------
    int
        Location of last true value, returns ? if not found

    See Also
    --------
    np.argmax, np.argmin

    Notes
    -----
    #.  Written by David C. Stauffer in May 2023.

    Examples
    --------
    >>> from dstauffman import find_last
    >>> x = np.array([0, 0, 1, 0, 1, 0, 0], dtype=bool)
    >>> ix = find_last(x)
    >>> print(ix)
    4

    """
    if ~np.any(x):
        return -1
    return int(x.size - 1 - np.argmax(np.flip(x)))


# %% prepend
@overload
def prepend(vec: _B, new: bool) -> _B: ...
@overload
def prepend(vec: _I, new: int) -> _I: ...
@overload
def prepend(vec: _N, new: float) -> _N: ...
def prepend(vec: _B | _I | _N, new: bool | int | float) -> _B | _I | _N:
    r"""
    Add a value to the beginning of an array.

    Parameters
    ----------
    vec : (N,) numpy.ndarray
        Original array
    new : float
        Value to prepend

    Returns
    -------
    (N+1,) numpy.ndarray
        Prepended array

    See Also
    --------
    np.hstack

    Notes
    -----
    #.  Written by David C. Stauffer in October 2023.

    Examples
    --------
    >>> from dstauffman import prepend
    >>> import numpy as np
    >>> vec = np.array([2, 3, 5])
    >>> new = 1
    >>> x = prepend(vec, new)
    >>> print(x)
    [1 2 3 5]

    """
    return np.hstack([new, vec])


# %% postpend
@overload
def postpend(vec: _B, new: bool) -> _B: ...
@overload
def postpend(vec: _I, new: int) -> _I: ...
@overload
def postpend(vec: _N, new: float) -> _N: ...
def postpend(vec: _B | _I | _N, new: bool | int | float) -> _B | _I | _N:
    r"""
    Add a value to the beginning of an array.

    Parameters
    ----------
    vec : (N,) numpy.ndarray
        Original array
    new : float
        Value to append at the end (postpend)

    Returns
    -------
    (N+1,) numpy.ndarray
        Appended (new) array

    See Also
    --------
    np.hstack

    Notes
    -----
    #.  Written by David C. Stauffer in October 2023.

    Examples
    --------
    >>> from dstauffman import postpend
    >>> import numpy as np
    >>> vec = np.array([2, 3, 5])
    >>> new = 7
    >>> x = postpend(vec, new)
    >>> print(x)
    [2 3 5 7]

    """
    return np.hstack([vec, new])


# %% Unit test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_matlab", exit=False)
    doctest.testmod(verbose=False)
