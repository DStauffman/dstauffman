r"""
Functions related to Kalman Filter analysis.

Notes
-----
#.  Written by David C. Stauffer in April 2019.
"""

#%% Imports
import doctest
import unittest

import numpy as np

#%% Functions - calc_kalman_gain
def calc_kalman_gain(P, H, R, use_inverse=False, return_innov_cov=False):
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
    >>> from dstauffman.aerospace import calc_kalman_gain
    >>> import numpy as np
    >>> P = 1e-3 * np.eye(5)
    >>> H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0.5], [0, 0, 0.1]]).T
    >>> R = 0.5 * np.eye(3)
    >>> K = calc_kalman_gain(P, H, R)

    """
    # calculate the innovation covariance
    Pz = H @ P @ H.T + R
    if use_inverse:
        # explicit version with inverse
        K = (P @ H.T) @ np.linalg.inv(Pz)
    else:
        # implicit solver
        K = np.linalg.lstsq(Pz.T, (P @ H.T).T, rcond=None)[0].T
    # return desired results
    if return_innov_cov:
        return (K, Pz)
    return K

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
    >>> from dstauffman.aerospace import propagate_covariance
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
def update_covariance(P, K, H, inplace=True):
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
    >>> from dstauffman.aerospace import update_covariance
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
    unittest.main(module='dstauffman.tests.test_aerospace_kalman', exit=False)
    doctest.testmod(verbose=False)
