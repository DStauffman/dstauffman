r"""
Functions related to doing a backwards information smoother over a Kalman filter analysis.

Notes
-----
#.  Written by David C. Stauffer in July 2020.
"""

#%% Imports
import doctest
import unittest

from dstauffman import HAVE_NUMPY

from dstauffman.estimation.linalg import mat_divide

if HAVE_NUMPY:
    import numpy as np

#%% _update_information
def _update_information(H, Pz, z, K, lambda_bar, LAMBDA_bar):
    r"""
    Update information vector and matrix using innovation.

    Reference
    ---------
    #.  Bierman, Gerald J., "Kalman Filter Related Smoothers," Appendix X.A of "Factorization
        Methods for Discrete Sequential Estimation", Academic Press, 1977.
        (Equations A.15 and A.16)

    Paramaters
    ----------
    H : (n_meas, n_state) ndarray
        Measurement distribution matrix
    Pz : (n_meas, n_meas) ndarray
        Innovation covariance matrix
    z : (n_meas, ) ndarray
        innovation vector
    K : (n_state, n_meas) ndarray
        Kalman gain matrix
    lambda_bar : (n_state,) ndarray
        information vector before measurement update
    LAMBDA_bar : (n_state, n_state) ndarray
        information matrix before measurement update

    Returns
    -------
    lambda_hat : (n_state,) ndarray
        updated information vector
    LAMBDA_hat : (n_state, n_state) ndarray
        updated information matrix

    Examples
    --------
    >>> from dstauffman.estimation.smoother import _update_information
    >>> # TODO: write the rest

    """
    delta_lambda = -H.T @ (mat_divide(Pz, z) + K.T @ lambda_bar)
    I            = np.hstack((np.eye(K.shape[0]), np.zeros((K.shape[0], H.shape[1]-K.shape[0]))))
    I_minus_KH   = I - K @ H
    lambda_hat   = lambda_bar + delta_lambda
    LAMBDA_hat   = I_minus_KH.T @ LAMBDA_bar @ I_minus_KH + H.T @ mat_divide(Pz, H)
    return (lambda_hat, LAMBDA_hat)

#%% bf_smoother
def bf_smoother(kf_record, lambda_bar=None, LAMBDA_bar=None):
    r"""
    Modified Bryson Frasier smoother.

    Parameters
    ----------
    kf_record : class dstauffman.estimation.KfRecord
        Record of all the interval Kalman Filter calculations
    lambda_bar : (n_state, 1) ndarray, optional
        final boundary condition on information vector
    LAMBDA_bar :
        final boundary condition on information matrix

    Returns
    -------
    x_delta        : (N, M) ndarray
        Change to state computed by smoothing. Add this to the filtered state to get the smoothed state.
    lambda_bar     : (N, ) ndarray
        initial information vector
    LAMBDA_bar : (N, N) ndarray
        initial information matrix

    Notes
    -----
    #.  Implements a modified Bryson-Frasier (BF) backward smoother
    #.  Inputs define state dynamics and measurement matrices and innovation vector from the
        forward pass of a Kalman filter. Output is the smoothed state.
    #.  BF smoother requires input of the Kalman gain K used on the forward pass as well as the
        forward pass state and innovation covariances.

    References
    ----------
    #.  Bierman, Gerald J., "Kalman Filter Related Smoothers," Appendix X.A of "Factorization
        Methods for Discrete Sequential Estimation", Academic Press, 1977.

    Examples
    --------
    >>> from dstauffman.aerospace import KfRecord
    >>> from dstauffman.estimation import bf_smoother
    >>> import numpy as np
    >>> num_points = 5
    >>> num_states = 6
    >>> num_axes = 2
    >>> stm = np.eye(num_states)
    >>> P = np.eye(num_states)
    >>> H = np.ones((num_axes, num_states))
    >>> Pz = np.eye(num_axes, num_axes)
    >>> K = np.ones((num_states, num_axes))
    >>> z = np.ones(num_axes)
    >>> lambda_bar_final = np.ones(num_states)
    >>> kf_record = KfRecord(num_points=num_points, num_active=num_states, num_states=num_states, num_axes=num_axes)
    >>> for i in range(num_points):
    ...     kf_record.time[i] = float(num_points)
    ...     kf_record.stm[:, :, i] = stm
    ...     kf_record.P[:, :, i] = P
    ...     kf_record.H[:, :, i] = H
    ...     kf_record.Pz[:, :, i] = Pz
    ...     kf_record.K[:, :, i] = K
    ...     kf_record.z[:, i] = z
    >>> (x_delta, lambda_bar_initial, LAMBDA_bar_initial) = bf_smoother(kf_record)

    """
    #% Initialization, set defaults
    n_state    = kf_record.H.shape[1]
    n_active   = kf_record.K.shape[0]
    n_time     = kf_record.time.size

    # Storage for smoothed state updates
    x_delta = np.zeros((n_active, n_time), dtype=float)
    if LAMBDA_bar is None:
        # If starting at the end of a datafile, set
        # Terminal boundary condition for
        # backward information matrix. Otherwise, use input value
        LAMBDA_bar = np.zeros((n_state, n_state), dtype=float)
    if lambda_bar is None:
        # If starting at the end of a datafile, set
        # Terminal boundary condition for
        # backward information vector. Otherwise, use input value
        lambda_bar = np.zeros(n_state, dtype=float)
    if kf_record.H.size == 0:
        # if all H matricies are empty, no stars were seen
        # compute all x_delta with same lambda_bar, then return
        for i in range(n_time):
            x_delta[:, i] = -kf_record.P[:, :, i] @ lambda_bar
        return (x_delta, lambda_bar, LAMBDA_bar)

    #% Final point
    #Compute smoothed state delta x_delta from  backward propagated information vector
    x_delta[:,-1]  = -kf_record.P[:, :, -1] @ lambda_bar
    #Update information vector using innovation
    H   = kf_record.H[:, :, -1]
    Pz  = kf_record.Pz[:, :, -1]
    K   = kf_record.K[:, :, -1]
    z   = kf_record.z[:, -1]
    #Update information
    (lambda_hat, LAMBDA_hat) = _update_information(H, Pz, z, K, lambda_bar, LAMBDA_bar)

    #% Backwards time loop
    for i in range(n_time-2, 0, -1):
        stm = kf_record.stm[:, :, i+1] # TODO: why is this i+1? # TODO: allow unpacking function here?
        # Create local copies of current filter data matrices
        P   = kf_record.P[:, :, i]
        H   = kf_record.H[:, :, i]
        Pz  = kf_record.Pz[:, :, i]
        K   = kf_record.K[:, :, i]
        z   = kf_record.z[:, i]
        # Propagate information vector and matrix backwards
        lambda_bar = stm.T @ lambda_hat
        LAMBDA_bar = stm.T @ LAMBDA_hat @ stm
        # %Compute smoothed state update from forward filtered state and backward
        # propagated information vector
        x_delta[:, i] = -P @ lambda_bar
        # Update information
        (lambda_hat, LAMBDA_hat) = _update_information(H, Pz, z, K, lambda_bar, LAMBDA_bar)

    return (x_delta, lambda_bar, LAMBDA_bar)

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_estimation_smoother', exit=False)
    doctest.testmod(verbose=False)
