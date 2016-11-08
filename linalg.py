# -*- coding: utf-8 -*-
r"""
Linalg module file for the dstauffman code.  It defines useful linear algebra methods that are not in
NumPy or SciPy.

Notes
-----
#.  Written by David C. Stauffer in May 2016.  Mostly based on Matlab routines.
"""
# pylint: disable=E1101, C0301, C0326

#%% Imports
import doctest
import numpy as np
from scipy.linalg import norm, svd
import unittest

#%% orth
def orth(A):
    r"""
    Orthogonalization basis for the range of A.  That is, Q'*Q = I, the columns of Q span the same
    space as the columns of A, and the number of columns of Q is the rank of A.

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
    >>> print(Q)
    [[-0.12000026 -0.80971228  0.57442663]
     [ 0.90175265  0.15312282  0.40422217]
     [-0.41526149  0.5664975   0.71178541]]

    Rank deficient matrix
    >>> A = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    >>> r = np.linalg.matrix_rank(A)
    >>> print(r)
    2

    >>> Q = orth(A)
    >>> print(np.array2string(Q, suppress_small=True))
    [[-0.70710678 -0.        ]
     [ 0.          1.        ]
     [-0.70710678  0.        ]]

    """
    (Q, S, _) = svd(A, full_matrices=False)
    if S.size > 0:
        tol = np.max(A.shape) * S[0] * np.finfo(float).eps
        r = np.sum(S > tol, axis=0)
        Q = Q[:, np.arange(r)]
    return Q

#%% subspace
def subspace(A, B):
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
    >>> B = np.array([[1, 1, 1, 1], [1, -1, 1, -1],[1, 1, -1, -1], [1, -1, -1, 1], [-1, -1, -1, -1], \
    ...     [-1, 1, -1, 1], [-1, -1, 1, 1], [-1, 1, 1, -1]])
    >>> theta = subspace(A, B)
    >>> print('{:.4f}'.format(theta))
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
    theta = np.arcsin(np.minimum(1, norm(B)))
    return theta

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='tests.test_linalg', exit=False)
    doctest.testmod(verbose=False)
