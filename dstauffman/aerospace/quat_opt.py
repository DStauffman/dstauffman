r"""
Contains generic quaternion utilities that are optimized for runtime performance.

Notes
-----
#.  Written by David C. Stauffer in February 2021.
"""

# %% Imports
from __future__ import annotations

import doctest
from typing import TYPE_CHECKING
import unittest

from nubs import ncjit

from dstauffman import HAVE_NUMPY
from dstauffman.aerospace.vectors import vec_cross

if HAVE_NUMPY:
    import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    _N = NDArray[np.floating]
    _Q = NDArray[np.floating]  # shape (4,)
    _V = NDArray[np.floating]  # shape (3,)
    _DCM = NDArray[np.floating]  # shape (3, 3)


# %% Functions - qrot_single
@ncjit
def qrot_single(axis: int, angle: float) -> _Q:
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
    ...     print(quat)  # doctest: +NORMALIZE_WHITESPACE
    [0. 0. 0.70710678  0.70710678]

    """
    c = np.cos(angle / 2)
    if c < 0:
        quat = np.array([0.0, 0.0, 0.0, -c])
        quat[axis - 1] = -np.sin(angle / 2)
    else:
        quat = np.array([0.0, 0.0, 0.0, c])
        quat[axis - 1] = np.sin(angle / 2)
    return quat


# %% Functions - quat_from_axis_angle_single
@ncjit
def quat_from_axis_angle_single(axis: _V, angle: float) -> _Q:
    r"""
    Construct a quaternion expressing the given rotation about the given axis.

    Parameters
    ----------
    axis : (3, ) numpy.ndarray of float
        Unit vector
    angle : float
        angle of rotation in radians

    Returns
    -------
    quat : ndarray, (4,)
        quaternion representing the given rotation

    Notes
    -----
    #.  Written by David C. Stauffer in April 2021.

    References
    ----------
    #.  A quaternion is given by [x*s, y*s, z*s, c] where c = cos(theta/2) and sin=(theta/2)
        See: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    Examples
    --------
    >>> from dstauffman.aerospace import quat_from_axis_angle_single
    >>> import numpy as np
    >>> axis = np.sqrt([9/50, 16/50, 0.5])
    >>> angle = 5/180*np.pi
    >>> quat = quat_from_axis_angle_single(axis, angle)
    >>> with np.printoptions(precision=8):
    ...     print(quat)  # doctest: +NORMALIZE_WHITESPACE
    [0.01850614 0.02467485 0.03084356 0.99904822]

    """
    if axis[0] == 0.0 and axis[1] == 0.0 and axis[2] == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0])
    c = np.cos(angle / 2.0)
    s = np.sin(angle / 2.0)
    if c < 0.0:
        return np.array([-axis[0] * s, -axis[1] * s, -axis[2] * s, -c])
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c])


# %% Functions - quat_from_rotation_vector_single
@ncjit
def quat_from_rotation_vector_single(rv: _V) -> _Q:
    r"""
    Construct a quaternion expressing the given rotation vector.

    Parameters
    ----------
    rv : (3, ) numpy.ndarray of float
        Rotation vector, where the magnitude gives the amount of rotation

    Returns
    -------
    quat : ndarray, (4,)
        quaternion representing the given rotation

    Notes
    -----
    #.  Written by David C. Stauffer in June 2024.

    References
    ----------
    #.  A quaternion is given by [x*s, y*s, z*s, c] where c = cos(theta/2) and sin=(theta/2)
        See: https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation

    Examples
    --------
    >>> from dstauffman.aerospace import quat_from_rotation_vector_single
    >>> import numpy as np
    >>> rv = 5/180*np.pi * np.sqrt([9/50, 16/50, 0.5])
    >>> quat = quat_from_rotation_vector_single(rv)
    >>> with np.printoptions(precision=8):
    ...     print(quat)  # doctest: +NORMALIZE_WHITESPACE
    [0.01850614 0.02467485 0.03084356 0.99904822]

    """
    if rv[0] == 0.0 and rv[1] == 0.0 and rv[2] == 0.0:
        return np.array([0.0, 0.0, 0.0, 1.0])
    angle = np.sqrt(np.sum(rv * np.conj(rv)))
    axis = rv / angle
    c = np.cos(angle / 2.0)
    s = np.sin(angle / 2.0)
    if c < 0.0:
        return np.array([-axis[0] * s, -axis[1] * s, -axis[2] * s, -c])
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c])


# %% Functions - quat_angle_diff_single
@ncjit
def quat_angle_diff_single(quat1: _Q, quat2: _Q) -> _Q:
    r"""
    Calculate the angular difference between two quaternions.

    This function takes two quaternions and calculates a delta quaternion between them.
    It then uses the delta quaternion to generate both a total angular difference, and an
    angular difference expressed in X, Y, Z components based on the axis of rotation,
    expressed in the original frame of the quat1 input quaternion.  This function uses full
    trignometric functions instead of any small angle approximations.

    Parameters
    ----------
    quat1 : ndarray (4,)
        quaternion one
    quat2 : ndarray (4,)
        quaternion two

    Returns
    -------
    comp  : ndarray (3,)
        angle components in x, y, z frame [rad]

    References
    ----------
    This function is based on this representation of a unit quaternion:
    quat = [[nx * sin(theta/2)]
            [ny * sin(theta/2)]
            [nz * sin(theta/2)]
            [   cos(theta/2)  ]]
    Where: <nx, ny, nz> are the three components of a unit vector of rotation axis and
           theta is the angle of rotation

    Notes
    -----
    #.  Additional keyword arguments are passed on to quat_assertions function.
    #.  Adapted from GARSE by David C. Stauffer in April 2015.
    #.  Split into compiled single version by David C. Stauffer in October 2024.

    Examples
    --------
    >>> from dstauffman.aerospace import qrot_single, quat_mult_single, quat_angle_diff_single
    >>> import numpy as np
    >>> quat1 = np.array([0.5, 0.5, 0.5, 0.5])
    >>> dq1 = qrot_single(1, 0.001)
    >>> dq2 = qrot_single(2, 0.05)
    >>> quat2 = quat_mult_single(dq1, quat1)
    >>> comp = quat_angle_diff_single(quat1, quat2)
    >>> with np.printoptions(precision=8):
    ...     print(comp)  # doctest: +NORMALIZE_WHITESPACE
    [0.001 0.    0.   ]

    """
    # calculate delta quaternion
    dq = quat_mult_single(quat2, quat_inv_single(quat1))

    # pull vector components out of delta quaternion
    dv = dq[0:3]

    # sum vector components to get sin(theta/2)^2
    mag2 = np.sum(dv**2)

    # take square root to get sin(theta/2)
    mag = np.sqrt(mag2)

    # take inverse sine to get theta/2
    theta_over_2 = np.arcsin(mag)

    # multiply by 2 to get theta
    theta = 2 * theta_over_2

    # set any magnitude that is identically 0 to be 1 instead
    # to avoid a divide by zero warning.
    if mag == 0:
        mag = 1

    # normalize vector components
    nv = dv / mag

    # find angle expressed in x, y, z components based on normalized vector
    comp = nv * theta

    return comp  # type: ignore[no-any-return]


# %% Functions - quat_interp_single
@ncjit
def quat_interp_single(time: _N, quat: _Q, ti: _N) -> _Q:
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
    >>> quat = np.array([[0., 0.5], [0., -0.5], [0., -0.5], [1., 0.5]], order="F")
    >>> ti = 3.
    >>> qout = quat_interp_single(time, quat, ti)
    >>> print(np.array_str(qout, precision=8, suppress_small=True))  # doctest: +NORMALIZE_WHITESPACE
    [ 0.28867513 -0.28867513 -0.28867513  0.8660254 ]

    """
    # check for desired times that are outside the time vector
    if (ti < time[0]) | (ti > time[-1]):
        raise ValueError("Desired time not found within input time vector.")

    # pull out bounding times and quaternions
    t1 = time[0]
    t2 = time[1]
    q1 = quat[:, 0].copy()
    q2 = quat[:, 1].copy()
    # calculate delta quaternion
    dq12 = quat_norm_single(quat_mult_single(q2, quat_inv_single(q1)))
    # find delta quaternion axis of rotation
    vec = dq12[0:3]
    norm_vec = np.sqrt(np.sum(vec**2))
    # check for zero norm vectors
    norm_fix = norm_vec if norm_vec != 0.0 else 1.0
    ax = vec / norm_fix
    # find delta quaternion rotation angle
    ang = 2 * np.arcsin(norm_vec)
    # scale rotation angle based on time
    scaled_ang = ang * (ti - t1) / (t2 - t1)
    # find scaled delta quaternion
    dq = quat_from_axis_angle_single(ax, scaled_ang)
    # calculate desired quaternion
    qout: _Q = quat_norm_single(quat_mult_single(dq, q1))
    # enforce positive scalar component
    if qout[3] < 0:
        qout[:] = -qout
    return qout


# %% Functions - quat_inv_single
@ncjit
def quat_inv_single(q1: _Q, inplace: bool = False) -> _Q:
    r"""
    Return the inverse of a normalized quaternions.

    Parameters
    ----------
    q1 : ndarray, (4, )
        input quaternion
    inplace : bool, optional, default is False
        Whether to modify the input in-place

    Returns
    -------
    ndarray, (4, )
        inverse quaternion

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
    ...     print(q2)  # doctest: +NORMALIZE_WHITESPACE
    [-0.70710678 -0. -0. 0.70710678]

    """
    if inplace:
        q1 *= np.array([-1.0, -1.0, -1.0, 1.0])
        return q1
    return q1 * np.array([-1.0, -1.0, -1.0, 1.0])  # type: ignore[no-any-return]


# %% Functions - quat_mult_single
@ncjit
def quat_mult_single(a: _Q, b: _Q, inplace: bool = False) -> _Q:
    r"""
    Multiply quaternions together.

    Parameters
    ----------
    a : ndarray, (4,) or (4, N)
        input quaternion one
    b : ndarray, (4,) or (4, N)
        input quaternion two
    inplace : bool, optional, default is False
        Whether to modify the input in-place

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
    >>> print(np.array_str(c, precision=8, suppress_small=True))  # doctest: +NORMALIZE_WHITESPACE
    [ 0. 0.70710678 -0.70710678 0. ]

    """
    c = a if inplace else a.copy()
    # fmt: off
    c[:] = np.array([
        +b[0] * a[3] + b[1] * a[2] - b[2] * a[1] + b[3] * a[0],
        -b[0] * a[2] + b[1] * a[3] + b[2] * a[0] + b[3] * a[1],
        +b[0] * a[1] - b[1] * a[0] + b[2] * a[3] + b[3] * a[2],
        -b[0] * a[0] - b[1] * a[1] - b[2] * a[2] + b[3] * a[3],
    ])
    # fmt: on
    # enforce positive scalar component
    if c[3] < 0:
        c[:] = -c
    quat_norm_single(c, inplace=True)
    return c


# %% Functions - quat_norm_single
@ncjit
def quat_norm_single(x: _Q, inplace: bool = False) -> _Q:
    r"""
    Normalize each column of the input matrix.

    Parameters
    ----------
    x : ndarray
        input quaternion
    inplace : bool, optional, default is False
        Whether to modify the input in-place

    Returns
    -------
    ndarray
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
    ...     print(y)  # doctest: +NORMALIZE_WHITESPACE
    [0.09950372 0. 0. 0.99503719]

    """
    # divide input by its column vector norm
    if inplace:
        x /= np.sqrt(np.sum(x * x, axis=0))
        return x
    return x / np.sqrt(np.sum(x * x, axis=0))  # type: ignore[no-any-return]


# %% Functions - quat_prop_single
@ncjit
def quat_prop_single(quat: _Q, delta_ang: _V, use_approx: bool = False, inplace: bool = False, renorm: bool = True) -> _Q:
    r"""
    Approximate propagation of a quaternion using a small delta angle.

    Parameters
    ----------
    quat : ndarray, (4,)
        normalized input quaternion
    delta_ang : ndarray, (3,)
        delta angles in x, y, z order [rad]
    inplace : bool, optional, default is False
        Whether to modify the input in-place

    Returns
    -------
    quat_new : ndarray, (4,)
        propagated quaternion, optionally re-normalized

    See Also
    --------
    quat_mult_single, quat_inv_single, quat_norm_single, quat_times_vector_single, quat_to_dcm

    Notes
    -----
    #.  Adapted from GARSE by David C. Stauffer in April 2015.
    #.  Optimized for numba by David C. Stauffer in February 2021.

    Examples
    --------
    >>> from dstauffman.aerospace import quat_norm_single, quat_prop_single
    >>> import numpy as np
    >>> quat      = np.array([0., 0., 0., 1.])
    >>> delta_ang = np.array([0.01, 0.02, 0.03])
    >>> quat_new  = quat_prop_single(quat, delta_ang, use_approx=True, renorm=False)
    >>> print(quat_new)  # doctest: +NORMALIZE_WHITESPACE
    [0.005 0.01 0.015 1. ]

    >>> quat_new = quat_prop_single(quat, delta_ang)
    >>> with np.printoptions(precision=8):
    ...     print(quat_new)  # doctest: +NORMALIZE_WHITESPACE
    [0.00499971 0.00999942 0.01499913 0.99982501]

    >>> quat_new = quat_prop_single(quat, delta_ang, use_approx=True)
    >>> with np.printoptions(precision=8):
    ...     print(quat_new)  # doctest: +NORMALIZE_WHITESPACE
    [0.00499913 0.00999825 0.01499738 0.99982505]

    """
    quat_new = quat if inplace else quat.copy()
    # compute angle rate matrix (note: transposed to make "F" order), use it to compute a delta
    # quaternion, and then propagate by adding the delta
    # fmt: off
    if use_approx:
        quat_new += 0.5 * np.array([
            [      0      , -delta_ang[2],  delta_ang[1], -delta_ang[0]],
            [ delta_ang[2],       0      , -delta_ang[0], -delta_ang[1]],
            [-delta_ang[1],  delta_ang[0],      0       , -delta_ang[2]],
            [ delta_ang[0],  delta_ang[1],  delta_ang[2],       0      ],
        ]).T @ quat
    # fmt: on
    else:
        dq_mag = np.sqrt(np.sum(delta_ang**2))
        delta_quat = np.array([0.0, 0.0, 0.0, 1.0])
        if dq_mag > 1e-14:
            fact = np.sin(dq_mag / 2) / dq_mag
            delta_quat[0] = fact * delta_ang[0]
            delta_quat[1] = fact * delta_ang[1]
            delta_quat[2] = fact * delta_ang[2]
            delta_quat[3] = np.cos(dq_mag / 2)
        quat_new[:] = quat_mult_single(delta_quat, quat)
    # ensure positive scalar component
    if quat_new[3] < 0:
        quat_new[:] = -quat_new
    # renormalize and return
    if renorm:
        quat_norm_single(quat_new, inplace=True)
    return quat_new


# %% Functions - quat_times_vector_single
@ncjit
def quat_times_vector_single(quat: _Q, v: _V, inplace: bool = False) -> _Q:
    r"""
    Multiply quaternion(s) against vector(s).

    Parameters
    ----------
    quat : ndarray, (4,) or (4, N)
        quaternion(s)
    v : ndarray, (3,) or (3, N)
        input vector(s)
    inplace : bool, optional, default is False
        Whether to modify the input in-place

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
    >>> print(vec)  # doctest: +NORMALIZE_WHITESPACE
    [-1. 0. 0.]

    """
    vec = v if inplace else v.copy()
    skew = vec_cross(quat[:3])
    qv = skew @ v
    vec += 2 * (-quat[3] * qv + (skew @ qv))
    return vec


# %% Functions - quat_to_dcm
@ncjit
def quat_to_dcm(quat: _Q) -> _DCM:
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
    >>> print(dcm)  # doctest: +NORMALIZE_WHITESPACE
    [[ 0.  0.  1.]
     [-1.  0.  0.]
     [ 0. -1.  0.]]

    """
    # build dcm components
    dcm = np.zeros((3, 3))
    dcm[0, 0] = quat[3] ** 2 + quat[0] ** 2 - quat[1] ** 2 - quat[2] ** 2
    dcm[0, 1] = 2 * (quat[0] * quat[1] + quat[2] * quat[3])
    dcm[0, 2] = 2 * (quat[0] * quat[2] - quat[1] * quat[3])
    dcm[1, 0] = 2 * (quat[0] * quat[1] - quat[2] * quat[3])
    dcm[1, 1] = quat[3] ** 2 - quat[0] ** 2 + quat[1] ** 2 - quat[2] ** 2
    dcm[1, 2] = 2 * (quat[1] * quat[2] + quat[0] * quat[3])
    dcm[2, 0] = 2 * (quat[0] * quat[2] + quat[1] * quat[3])
    dcm[2, 1] = 2 * (quat[1] * quat[2] - quat[0] * quat[3])
    dcm[2, 2] = quat[3] ** 2 - quat[0] ** 2 - quat[1] ** 2 + quat[2] ** 2
    return dcm


# %% Unit test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_aerospace_quat_opt", exit=False)
    doctest.testmod(verbose=False)
