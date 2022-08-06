r"""
Functions related to Kalman Filter analysis.

Notes
-----
#.  Written by David C. Stauffer in April 2019.
"""

#%% Imports
from __future__ import annotations

import doctest
from typing import Literal, overload, Optional, Tuple, TYPE_CHECKING, Union
import unittest

from dstauffman import HAVE_NUMPY
from dstauffman.estimation.linalg import mat_divide
from dstauffman.nubs import ncjit

if HAVE_NUMPY:
    import numpy as np

if TYPE_CHECKING:
    _N = np.typing.NDArray[np.float64]
    _M = np.typing.NDArray[np.float64]  # 2D

#%% Functions - calculate_kalman_gain
@overload
def calculate_kalman_gain(P: _M, H: _M, R: _M, *, use_inverse: bool, return_innov_cov: Literal[False] = ...) -> _M:
    ...


@overload
def calculate_kalman_gain(P: _M, H: _M, R: _M, *, use_inverse: bool, return_innov_cov: Literal[True]) -> Tuple[_M, _M]:
    ...


def calculate_kalman_gain(
    P: _M, H: _M, R: _M, *, use_inverse: bool = False, return_innov_cov: bool = False
) -> Union[_M, Tuple[_M, _M]]:
    r"""
    Calculates K, the Kalman Gain matrix.

    Parameters
    ----------
    P : (N, N) ndarray
        Covariance Matrix
    H : (A, B) ndarray
        Measurement Update Matrix
    R : () ndarray
        Measurement Noise Matrix
    use_inverse : bool, optional
        Whether to explicitly calculate the inverse or not, default is False

    Returns
    -------
    K : (N, ) ndarray
        Kalman Gain Matrix
    Pz : (N, N) ndarray
        Innovation Covariance Matrix

    Notes
    -----
    #.  Written by David C Stauffer in December 2018.

    Examples
    --------
    >>> from dstauffman.estimation import calculate_kalman_gain
    >>> import numpy as np
    >>> P = 1e-3 * np.eye(5)
    >>> H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5], [0, 0, 0.1]]).T
    >>> R = 0.5 * np.eye(3)
    >>> K = calculate_kalman_gain(P, H, R)

    """
    # calculate the innovation covariance
    Pz = H @ P @ H.T + R
    if use_inverse:
        # explicit version with inverse
        K = (P @ H.T) @ np.linalg.inv(Pz)
    else:
        # implicit solver
        K = mat_divide(Pz.T, (P @ H.T).T).T
    # return desired results
    if return_innov_cov:
        return (K, Pz)
    return K


@ncjit
def calculate_kalman_gain_opt(P: _M, H: _M, R: _M) -> Tuple[_M, _M]:
    r"""Calculate the Kalman gain, in a way optimized for use with numba."""
    Pz = H @ P @ H.T + R
    K = mat_divide(Pz.T, (P @ H.T).T).T
    return (K, Pz)


#%% Functions - calculate_prediction
@ncjit
def calculate_prediction(H: _M, state: _N, const: Optional[_N] = None) -> _N:
    r"""
    Calculates u, the measurement prediction.

    Parameters
    ----------
    H : (A, B) ndarray
        Measurement Update matrix
    state : (A, ) ndarray
        State vector
    const : (A, ) ndarray, optional
        Constant state vector offsets

    Returns
    -------
    (A, ) ndarray
        Delta state vector

    Notes
    -----
    #.  Written by David C. Stauffer in September 2020.

    Examples
    --------
    >>> from dstauffman.estimation import calculate_prediction
    >>> import numpy as np
    >>> H = np.array([[1., 0.], [0., 1.], [0., 0.]])
    >>> state = np.array([1e-3, 5e-3])
    >>> u_pred = calculate_prediction(H, state)
    >>> print(u_pred) # doctest: +NORMALIZE_WHITESPACE
    [0.001 0.005 0. ]

    """
    if const is None:
        return H @ state
    return H @ (state + const)


#%% Functions - calculate_innovation
@ncjit
def calculate_innovation(u_meas: _N, u_pred: _N) -> _N:
    r"""
    Calculates z, the Kalman Filter innovation.

    Parameters
    ----------
    u_meas : (A, ) ndarray
        Measured state vector
    u_pred : (A, ) ndarray
        Predicted state vector

    Returns
    -------
    (A, ) ndarray
        Kalman Filter innovation

    Notes
    -----
    #.  Written by David C. Stauffer in September 2020.

    Examples
    --------
    >>> from dstauffman.estimation import calculate_innovation
    >>> import numpy as np
    >>> u_meas = np.array([1., 2.1, -3.])
    >>> u_pred = np.array([1.1, 2.0, -3.1])
    >>> z = calculate_innovation(u_meas, u_pred)
    >>> with np.printoptions(precision=8):
    ...     print(z) # doctest: +NORMALIZE_WHITESPACE
    [-0.1 0.1 0.1]

    """
    return u_meas - u_pred


#%% Functions - calculate_normalized_innovation
@ncjit
def calculate_normalized_innovation(z: _N, Pz: _M, use_inverse: bool = False) -> _N:
    r"""
    Calculates nu, the Normalized Kalman Filter Innovation.

    Parameters
    ----------
    z : (A, ) ndarray
        Kalman Filter innovation
    Pz : (A, A) ndarray
        Kalman Filter innovation covariance
    use_inverse : bool, optional
        Whether to explicitly calculate the inverse or not, default is False

    Returns
    -------
    (A, ) ndarray
        Normalized innovation

    Notes
    -----
    #.  Written by David C. Stauffer in September 2020.

    Examples
    --------
    >>> from dstauffman.estimation import calculate_normalized_innovation
    >>> import numpy as np
    >>> z = np.array([0.1, 0.05, -0.2])
    >>> Pz = np.array([[0.1, 0.01, 0.001], [0.01, 0.1, 0.001], [0., 0., 0.2]])
    >>> nu = calculate_normalized_innovation(z, Pz)
    >>> with np.printoptions(precision=8):
    ...     print(nu) # doctest: +NORMALIZE_WHITESPACE
    [ 0.96868687 0.41313131 -1. ]

    """
    if use_inverse:
        return np.linalg.inv(Pz) @ z
    return mat_divide(Pz, z)  # type: ignore[no-any-return]


#%% Functions - calculate_delta_state
@ncjit
def calculate_delta_state(K: _M, z: _N) -> _N:
    r"""
    Calculates dx, the delta state for a given measurement.

    Parameters
    ----------
    K : (A, B) ndarray
        Kalman Gain Matrix
    z : (A, ) ndarray
        Kalman Filter innovation

    Notes
    -----
    #.  Written by David C. Stauffer in September 2020.

    Examples
    --------
    >>> from dstauffman.estimation import calculate_delta_state
    >>> import numpy as np
    >>> K = np.array([[0.1, 0.01, 0.001], [0.01, 0.1, 0.001], [0., 0., 0.2]])
    >>> z = np.array([0.1, 0.05, -0.2])
    >>> dx = calculate_delta_state(K, z)
    >>> with np.printoptions(precision=8):
    ...     print(dx) # doctest: +NORMALIZE_WHITESPACE
    [ 0.0103 0.0058 -0.04 ]

    """
    return K @ z


#%% Functions - propagate_covariance
def propagate_covariance(P: _M, phi: _M, Q: _M, *, gamma: Optional[_M] = None, inplace: bool = True) -> _M:
    r"""
    Propagates the covariance forward in time.

    Parameters
    ----------
    P :
        Covariance matrix
    phi :
        State transition matrix
    Q :
        Process noise matrix
    gamma :
        Shaping matrix?
    inplace : bool, optional, default is True
        Whether to update the value inplace or as a new output

    Returns
    -------
    (N, N) ndarray
        Updated covariance matrix

    Notes
    -----
    #.  Written by David C. Stauffer in December 2018.
    #.  Updated by David C. Stauffer in July 2020 to have inplace option.

    Examples
    --------
    >>> from dstauffman.estimation import propagate_covariance
    >>> import numpy as np
    >>> P = 1e-3 * np.eye(6)
    >>> phi = np.diag([1., 1, 1, -1, -1, -1])
    >>> Q = np.diag([1e-3, 1e-3, 1e-5, 1e-7, 1e-7, 1e-7])
    >>> _ = propagate_covariance(P, phi, Q)
    >>> print(P[0, 0])
    0.002

    """
    if gamma is None:
        out = phi @ P @ phi.T + Q
    else:
        out = phi @ P @ phi.T + gamma @ Q @ gamma.T
    if inplace:
        P[:] = out
        return P
    return out


@ncjit
def propagate_covariance_opt(P: _M, phi: _M, Q: _M, gamma: Optional[_M] = None) -> None:
    r"""Propagate the covariance in time, in a way optimized for use with numba."""
    if gamma is None:
        P[:] = phi @ P @ phi.T + Q
    else:
        P[:] = phi @ P @ phi.T + gamma @ Q @ gamma.T


#%% Functions - update_covariance
def update_covariance(P: _M, K: _M, H: _M, *, inplace: bool = True) -> _M:
    r"""
    Updates the covariance for a given measurement.

    Parameters
    ----------
    P : (N, N) ndarray
        Covariance Matrix
    K : (N, ) ndarray
        Kalman Gain Matrix
    H : (A, N) ndarray
        Measurement Update Matrix
    inplace : bool, optional, default is True
        Whether to update the value inplace or as a new output

    Returns
    -------
    P_out : (N, N) ndarray
        Updated Covariance Matrix

    Notes
    -----
    #.  Written by David C Stauffer in December 2018.
    #.  Updated by David C. Stauffer in July 2020 to have inplace option.

    Examples
    --------
    >>> from dstauffman.estimation import update_covariance
    >>> import numpy as np
    >>> P = 1e-3 * np.eye(6)
    >>> P[0, -1] = 5e-2
    >>> K = np.ones((6, 3))
    >>> H = np.hstack((np.eye(3), np.eye(3)))
    >>> _ = update_covariance(P, K, H)
    >>> print(P[-1, -1])
    -0.05

    """
    out = (np.eye(P.shape[0], P.shape[1]) - K @ H) @ P
    if inplace:
        P[:] = out
        return P
    return out


@ncjit
def update_covariance_opt(P: _M, K: _M, H: _M) -> None:
    r"""Propagate the covariance in time, in a way optimized for use with numba."""
    P[:] = (np.eye(P.shape[0], P.shape[1]) - K @ H) @ P


#%% Unit Test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_estimation_kalman", exit=False)
    doctest.testmod(verbose=False)
