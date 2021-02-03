r"""
Contains generic quaternion utilities that are optimized for runtime performance.

Notes
-----
#.  Written by David C. Stauffer in February 2021.
"""

#%% Imports
import doctest
import unittest

from dstauffman import HAVE_NUMBA, HAVE_NUMPY

if HAVE_NUMBA:
    from numba import njit
else:
    from dstauffman.constants import fake_decorator

    @fake_decorator
    def njit(func, *args, **kwargs):
        r"""Fake njit decorator for when numba isn't installed."""
        return func

if HAVE_NUMPY:
    import numpy as np

#%% Functions - qrot_single
@njit(cache=True)
def qrot_single(axis, angle):
    r"""
    Construct a quaternion expressing a rotation about a single axis.

    Parameters
    ----------
    axis : int
        Axis about which rotation is being made, from {1, 2, 3}
            (1) for x-axis
            (2) for y-axis
            (3) for z-axis
    angle : array_like
        angle of rotation in radians

    Returns
    -------
    quat : ndarray, (4,) or (4, N)
        quaternion representing the given rotations

    Notes
    -----
    #.  Optimized for a single rotation by David C. Stauffer in February 2021.

    References
    ----------
    .. [1]  Wertz, James R. (editor), Equations 12.11 in Parameterization of the Attitude,
            Section 12.1, Spacecraft Attitude Determination and Control,
            Kluwer Academic Publishers, 1978.

    Examples
    --------
    >>> from dstauffman.aerospace import qrot_single
    >>> import numpy as np
    >>> quat = qrot_single(3, np.pi/2)
    >>> with np.printoptions(precision=8):
    ...     print(quat) # doctest: +NORMALIZE_WHITESPACE
    [0. 0. 0.70710678  0.70710678]

    """
    quat = np.array([0., 0., 0., np.cos(angle/2)])
    quat[axis-1] = np.sin(angle/2)
    return quat

#%% Functions - quat_interp_single
@njit(cache=True)
def quat_interp_single(time, quat, ti):
    r"""
    Interpolate quaternions from a monotonic time series of quaternions.

    Parameters
    ----------
    time : ndarray, (A, )
        monotonically increasing time series [sec]
    quat : ndarray, (4, A)
        quaternion series
    ti : ndarray (B, )
        desired time of interpolation, also monotonically increasing [sec]

    Returns
    -------
    qout : ndarray (4, B)
        interpolated quaternion at ti

    Notes
    -----
    #.  Optimized for a single lookup by David C. Stauffer in February 2021.

    Examples
    --------
    >>> from dstauffman.aerospace import quat_interp_single
    >>> import numpy as np
    >>> time  = np.array([1., 5.])
    >>> quat = np.array([[0., 0.5], [0., -0.5], [0., -0.5], [1., 0.5]], order='F')
    >>> ti = 3.
    >>> qout = quat_interp_single(time, quat, ti)
    >>> print(np.array_str(qout, precision=8, suppress_small=True)) # doctest: +NORMALIZE_WHITESPACE
    [ 0.28867513 -0.28867513 -0.28867513  0.8660254 ]

    """
    # check for desired times that are outside the time vector
    if (ti < time[0]) | (ti > time[-1]):
        raise ValueError('Desired time not found within input time vector.')

    # pull out bounding times and quaternions
    t1 = time[0]
    t2 = time[1]
    q1 = quat[:, 0]
    q2 = quat[:, 1]
    # calculate delta quaternion
    dq12       = quat_norm_single(quat_mult_single(q2, quat_inv_single(q1)))
    # find delta quaternion axis of rotation
    vec        = dq12[0:3]
    norm_vec   = np.sqrt(np.sum(vec**2))
    # check for zero norm vectors
    norm_fix   = norm_vec if norm_vec != 0. else 1.
    ax         = vec / norm_fix
    # find delta quaternion rotation angle
    ang        = 2*np.arcsin(norm_vec)
    # scale rotation angle based on time
    scaled_ang = ang*(ti-t1) / (t2-t1)
    # find scaled delta quaternion
    sin        = np.sin(scaled_ang/2)
    dq         = np.array([ax[0]*sin, ax[1]*sin, ax[2]*sin, np.cos(scaled_ang/2)])
    # calculate desired quaternion
    qout       = quat_norm_single(quat_mult_single(dq, q1))
    # enforce positive scalar component
    if qout[3] < 0:
        qout = -qout
    return qout

#%% Functions - quat_inv_single
@njit(cache=True)
def quat_inv_single(q1):
    r"""
    Return the inverse of a normalized quaternions.

    Parameters
    ----------
    q1 : ndarray, (4,) or (4, N)
        input quaternion

    Returns
    -------
    q2 : ndarray, (4,) or (4, N)
        inverse quaterion

    See Also
    --------
    quat_inv

    Notes
    -----
    #.  Adapted from GARSE by David C. Stauffer in April 2015.

    Examples
    --------
    >>> from dstauffman.aerospace import qrot_single, quat_inv_single
    >>> from numpy import pi
    >>> q1 = qrot_single(1, pi/2)
    >>> q2 = quat_inv_single(q1)
    >>> with np.printoptions(precision=8):
    ...     print(q2) # doctest: +NORMALIZE_WHITESPACE
    [-0.70710678 -0. -0. 0.70710678]

    """
    return q1 * np.array([-1., -1., -1., 1.])

#%% Functions - quat_mult_single
@njit(cache=True)
def quat_mult_single(a, b):
    r"""
    Multiply quaternions together.

    Parameters
    ----------
    a : ndarray, (4,) or (4, N)
        input quaternion one
    b : ndarray, (4,) or (4, N)
        input quaternion two

    Returns
    -------
    c : ndarray, (4,) or (4, N)
        result of quaternion multiplication

    See Also
    --------
    quat_inv, quat_norm, quat_prop, quat_times_vector, quat_to_dcm, quat_to_euler,
    quat_from_euler

    Notes
    -----
    #.  Additional keyword arguments are passed on to quat_assertions function.
    #.  Adapted from GARSE by David C. Stauffer in April 2015.
    #.  Each of (a, b) may be either a single quaternion (4,) or an array of quaternions (4, N).
        If `a` and `b` are both single quaternions, then return b*a. If either (but not both) is
        an array of quaternions, then return the product of the single quaternion times each element
        of the array. If both are rows of quaternions, multiply corresponding columns.
        `c` will have size (4,) in the first case, and (4, N) in the other cases.
    #.  The quaternions `a` and `b` describe successive reference frame changes, i.e., a is
        expressed in the coordinate system resulting from b, not in the original coordinate system.
        In Don Reid's tutorial, this is called the R- version.

    Examples
    --------
    >>> from dstauffman.aerospace import qrot_single, quat_mult_single
    >>> from numpy import pi
    >>> a = qrot_single(1, pi/2)
    >>> b = qrot_single(2, pi)
    >>> c = quat_mult_single(a, b)
    >>> print(np.array_str(c, precision=8, suppress_small=True)) # doctest: +NORMALIZE_WHITESPACE
    [ 0. 0.70710678 -0.70710678 0. ]

    """
    # single quaternion inputs case
    c = np.array([ \
        [ a[3],  a[2], -a[1],  a[0]], \
        [-a[2],  a[3],  a[0],  a[1]], \
        [ a[1], -a[0],  a[3],  a[2]], \
        [-a[0], -a[1], -a[2],  a[3]]]) @ b
    # enforce positive scalar component
    if c[3] < 0:
        c = -c
    c = quat_norm_single(c)
    return c

#%% Functions - quat_norm_single
@njit(cache=True)
def quat_norm_single(x):
    r"""
    Normalize each column of the input matrix.

    Parameters
    ----------
    x : ndarray
        input quaternion

    Returns
    -------
    y : ndarray
        normalized quaternion

    See Also
    --------
    quat_mult, quat_inv, quat_prop, quat_times_vector, quat_to_dcm, quat_to_euler, quat_from_euler

    Notes
    -----
    #.  Additional keyword arguments are passed on to quat_assertions function.
    #.  Adapted from GARSE by David C. Stauffer in April 2015.

    Examples
    --------
    >>> from dstauffman.aerospace import quat_norm_single
    >>> import numpy as np
    >>> x = np.array([0.1, 0., 0., 1.])
    >>> y = quat_norm_single(x)
    >>> with np.printoptions(precision=8):
    ...     print(y) # doctest: +NORMALIZE_WHITESPACE
    [0.09950372 0. 0. 0.99503719]

    """
    # divide input by its column vector norm
    y = x / np.sqrt(np.sum(x*x, axis=0))
    return y

#%% Functions - quat_times_vector_single
@njit(cache=True)
def quat_times_vector_single(quat, v):
    r"""
    Multiply quaternion(s) against vector(s).

    Parameters
    ----------
    quat : ndarray, (4,) or (4, N)
        quaternion(s)
    v : ndarray, (3,) or (3, N)
        input vector(s)

    Returns
    -------
    vec : ndarray, (3,) or (3, N)
        product vector(s)

    See Also
    --------
    quat_mult, quat_inv, quat_norm, quat_prop, quat_to_dcm, quat_to_euler, quat_from_euler

    Notes
    -----
    #.  Adapted from GARSE by David C. Stauffer in April 2015.
    #.  This function will broadcast a single vector or quaternion to the other dimension

    References
    ----------
    Steps to algorithm:
        1.  qv = quat(1:3) x v
        2.  vec = v + 2*[ -( quat(4) * qv ) + (quat(1:3) x qv) ]

    Examples
    --------
    >>> from dstauffman.aerospace import quat_times_vector_single
    >>> import numpy as np
    >>> quat = np.array([0., 1., 0., 0.])
    >>> v = np.array([1., 0., 0.])
    >>> vec = quat_times_vector_single(quat, v)
    >>> print(vec) # doctest: +NORMALIZE_WHITESPACE
    [-1. 0. 0.]

    """
    qv = np.cross(quat[:3], v)
    vec = v + 2*(np.full(3, -quat[3]) * qv + \
        np.cross(quat[:3], qv))
    return vec

#%% Functions - quat_to_dcm
@njit(cache=True)
def quat_to_dcm(quat):
    r"""
    Convert quaternion to a direction cosine matrix.

    Parameters
    ----------
    quat : ndarray (4, 1)
        quaternion

    Returns
    -------
    dcm : ndarray (3, 3)
        direction cosine matrix

    See Also
    --------
    quat_mult, quat_inv, quat_norm, quat_prop, quat_times_vector, quat_to_euler, quat_from_euler

    Notes
    -----
    #.  Adapted from GARSE by David C. Stauffer in April 2015.

    Examples
    --------
    >>> from dstauffman.aerospace import quat_to_dcm
    >>> import numpy as np
    >>> quat = np.array([0.5, -0.5, 0.5, 0.5])
    >>> dcm = quat_to_dcm(quat)
    >>> print(dcm) # doctest: +NORMALIZE_WHITESPACE
    [[ 0.  0.  1.]
     [-1.  0.  0.]
     [ 0. -1.  0.]]

    """
    #build dcm components
    dcm = np.zeros((3, 3))
    dcm[0, 0] = quat[3]**2 + quat[0]**2 - quat[1]**2 - quat[2]**2
    dcm[0, 1] = 2*(quat[0]*quat[1] + quat[2]*quat[3])
    dcm[0, 2] = 2*(quat[0]*quat[2] - quat[1]*quat[3])
    dcm[1, 0] = 2*(quat[0]*quat[1] - quat[2]*quat[3])
    dcm[1, 1] = quat[3]**2 - quat[0]**2 + quat[1]**2 - quat[2]**2
    dcm[1, 2] = 2*(quat[1]*quat[2] + quat[0]*quat[3])
    dcm[2, 0] = 2*(quat[0]*quat[2] + quat[1]*quat[3])
    dcm[2, 1] = 2*(quat[1]*quat[2] - quat[0]*quat[3])
    dcm[2, 2] = quat[3]**2 - quat[0]**2 - quat[1]**2 + quat[2]**2
    return dcm

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_aerospace_quat_opt', exit=False)
    doctest.testmod(verbose=False)
