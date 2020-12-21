r"""
Functions related to Kalman Filter analysis.

Notes
-----
#.  Written by David C. Stauffer in April 2019.
"""

#%% Imports
import doctest
import unittest

from dstauffman import HAVE_NUMPY

from dstauffman.estimation.linalg import mat_divide

if HAVE_NUMPY:
    import numpy as np

#%% Functions - calculate_kalman_gain
def calculate_kalman_gain(P, H, R, *, use_inverse=False, return_innov_cov=False):
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

#%% Functions - calculate_prediction
def calculate_prediction(H, state, const=None):
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
def calculate_innovation(u_meas, u_pred):
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
def calculate_normalized_innovation(z, Pz, use_inverse=False):
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
    return mat_divide(Pz, z)

#%% Functions - calculate_delta_state
def calculate_delta_state(K, z):
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
def propagate_covariance(P, phi, Q, *, gamma=None, inplace=True):
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
    >>> propagate_covariance(P, phi, Q)
    >>> print(P[0, 0])
    0.002

    """
    if gamma is None:
        out = phi @ P @ phi.T + Q
    else:
        out = phi @ P @ phi.T + gamma @ Q @ gamma.T
    if inplace:
        P[:] = out
    else:
        return out

#%% Functions - update_covariance
def update_covariance(P, K, H, *, inplace=True):
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
    >>> update_covariance(P, K, H)
    >>> print(P[-1, -1])
    -0.05

    """
    out = (np.eye(*P.shape) - K @ H) @ P
    if inplace:
        P[:] = out
    else:
        return out

#%% Unit Test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_estimation_kalman', exit=False)
    doctest.testmod(verbose=False)
