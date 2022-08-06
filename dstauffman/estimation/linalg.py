r"""
Define useful linear algebra methods that are not in NumPy or SciPy.

Notes
-----
#.  Written by David C. Stauffer in May 2016.  Mostly based on Matlab routines.
"""

#%% Imports
from __future__ import annotations

import doctest
from typing import TYPE_CHECKING, Union
import unittest

from dstauffman import HAVE_NUMPY
from dstauffman.nubs import ncjit

if HAVE_NUMPY:
    import numpy as np

if TYPE_CHECKING:
    _N = np.typing.NDArray[np.float64]
    _M = np.typing.NDArray[np.float64]  # 2D

#%% Constants
_EPS = float(np.finfo(float).eps) if HAVE_NUMPY else 2.220446049250313e-16

#%% orth
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
    >>> from dstauffman.estimation import orth
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


#%% subspace
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
    >>> from dstauffman.estimation import subspace
    >>> import numpy as np
    >>> A = np.array([[1, 1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1], [1, 1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, 1]])
    >>> B = np.array([[1, 1, 1, 1], [1, -1, 1, -1],[1, 1, -1, -1], [1, -1, -1, 1], [-1, -1, -1, -1], \
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


#%% mat_divide
@ncjit
def mat_divide(a: _M, b: _N, rcond: float = _EPS) -> Union[_N, _M]:
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
    >>> from dstauffman.estimation import mat_divide
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


#%% Unit test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_estimation_linalg", exit=False)
    doctest.testmod(verbose=False)
