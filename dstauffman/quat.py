# -*- coding: utf-8 -*-
r"""
Contains generic quaternion utilities that can be independently defined and used by other modules.

Notes
-----
#.  Written by David C. Stauffer in April 2015.

"""

#%% Imports
import doctest
import unittest

import numpy as np

from dstauffman.constants import INT_TOKEN, QUAT_SIZE

#%% Master flags
use_assertions = True

#%% Functions - _quat_assertions
def _quat_assertions(quat):
    r"""
    Check assertions about valid quaternions.

    Parameters
    ----------
    quat : ndarray, (4,) or (4, N)
        Quaternion

    Examples
    --------
    >>> from dstauffman.quat import _quat_assertions
    >>> import numpy as np
    >>> quat = np.array([0.5, 0.5, -0.5, 0.5])
    >>> _quat_assertions(quat)

    """
    if not use_assertions:
        return # pragma: no cover
    # hard-coded values
    precision = 1e-12
    # get sizes
    qsize = quat.size
    qndim = quat.ndim
    # check sizes and dimensions
    if qndim == 1:
        assert (qsize == 0 or qsize == QUAT_SIZE), 'Quaternion has invalid size: "{}"'.format(qsize)
    elif qndim == 2:
        assert (quat.shape[0] == QUAT_SIZE), 'Quaternion has invalid size for first ' + \
            'dimension: "{}"'.format(quat.shape[0])
    else:
        assert False, 'Quaternion has too many dimensions: "{}".'.format(qndim)
    # if a null quaternion, then checks are done
    if qsize == 0:
        return
    # check that values are all real
    assert np.all(np.isreal(quat)), 'Quaternion is not real'
    # check ranges
    if qndim == 1:
        assert (-1 <= quat[0] <= 1), 'Quaternion has bad range in x value: "{}"'.format(quat[0])
        assert (-1 <= quat[1] <= 1), 'Quaternion has bad range in y value: "{}"'.format(quat[1])
        assert (-1 <= quat[2] <= 1), 'Quaternion has bad range in z value: "{}"'.format(quat[2])
        assert ( 0 <= quat[3] <= 1), 'Quaternion has bad range in s value: "{}"'.format(quat[3])
    else:
        assert np.all(-1 <= quat[0,:]) and np.all(quat[0,:] <= 1), 'Quaternion has bad range ' + \
            'in x value, min: "{}", max:"{}"'.format(np.min(quat[0,:]), np.max(quat[0,:]))
        assert np.all(-1 <= quat[1,:]) and np.all(quat[1,:] <= 1), 'Quaternion has bad range ' + \
            'in y value, min: "{}", max:"{}"'.format(np.min(quat[1,:]), np.max(quat[1,:]))
        assert np.all(-1 <= quat[2,:]) and np.all(quat[2,:] <= 1), 'Quaternion has bad range ' + \
            'in z value, min: "{}", max:"{}"'.format(np.min(quat[2,:]), np.max(quat[2,:]))
        assert np.all( 0 <= quat[3,:]) and np.all(quat[3,:] <= 1), 'Quaternion has bad range ' + \
            'in s value, min: "{}", max:"{}"'.format(np.min(quat[3,:]), np.max(quat[3,:]))

    # check normalization
    q_norm_err = np.abs(1 - np.sum(quat**2, axis=0))
    assert np.all(q_norm_err <= precision), 'Quaternion has invalid normalization ' + \
        'error "{}".'.format(np.max(q_norm_err))

#%% Functions - qrot
def qrot(axis, angle):
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
    #.  Adapted from GARSE by David C. Stauffer in April 2015.

    References
    ----------
    .. [1]  Wertz, James R. (editor), Equations 12.11 in Parameterization of the Attitude,
            Section 12.1, Spacecraft Attitude Determination and Control,
            Kluwer Academic Publishers, 1978.

    Examples
    --------
    >>> from dstauffman import qrot
    >>> import numpy as np
    >>> quat = qrot(3, np.pi/2)
    >>> print(quat) # doctest: +NORMALIZE_WHITESPACE
    [0. 0. 0.70710678  0.70710678]

    """
    # assertions
    try:
        axis_set = set(axis)
    except TypeError:
        axis_set = {axis}
    assert len(axis_set - {1, 2, 3}) == 0, 'axis_set = {}'.format(axis_set)
    # calculations
    if np.isscalar(angle) and np.isscalar(axis):
        # optimized scalar case
        quat = np.array([0, 0, 0, np.cos(angle/2)])
        quat[axis-1] = np.sin(angle/2)
    elif np.isscalar(axis):
        # single axis, multiple angle case
        quat = np.vstack((np.zeros((3, len(angle))), np.expand_dims(np.cos(angle/2), axis=0)))
        quat[axis-1, :] = np.sin(angle/2)
    elif np.isscalar(angle):
        # single angle, multiple axis case
        quat = np.array([[0], [0], [0], [np.cos(angle/2)]]) * np.ones((1, len(axis))) # TODO: formulate better
        for (i, this_axis) in enumerate(axis): # TODO: eliminate this for loop somehow?
            quat[this_axis-1, i] = np.sin(angle/2)
    else:
        # multiple axis, multiple angle case
        assert len(axis) == len(angle)
        quat = np.vstack((np.zeros((3, len(angle))), np.expand_dims(np.cos(angle/2), axis=0)))
        for i in range(len(axis)): # TODO: eliminate this for loop somehow?
            quat[axis[i]-1, i] = np.sin(angle[i]/2)
    _quat_assertions(quat)
    return quat

#%% Functions - quat_angle_diff
def quat_angle_diff(quat1, quat2):
    r"""
    Calculate the angular difference between two quaternions.

    This function takes a two quaternions and calculates a delta quaternion between them.
    It then uses the delta quaternion to generate both a total angular difference, and an
    an angular difference expressed in X, Y, Z components based on the axis of rotation,
    expressed in the original frame of the quat1 input quaternion.  This function uses full
    trignometric functions instead of any small angle approximations.

    Parameters
    ----------
    quat1 : ndarray (4,) or (4, N)
        quaternion one
    quat2 : ndarray (4,) or (4, N)
        quaternion two

    Returns
    -------
    theta : ndarray (1, N)
        angular different [rad]
    comp  : ndarray (3, N)
        angle components in x, y, z frame [rad]

    References
    ----------
    This function is based on this representation of a unit quaternion:
    quat = [[nx * sin(theta/2)]
            [ny * sin(theta/2)]
            [nz * sin(theta/2)]
            [   cos(theta/2)  ]]
    Where: <nx,ny,nz> are the three components of a unit vector of rotation axis and
           theta is the angle of rotation

    Notes
    -----
    #.  Adapted from GARSE by David C. Stauffer in April 2015.

    Examples
    --------
    >>> from dstauffman import qrot, quat_mult, quat_angle_diff
    >>> import numpy as np
    >>> quat1 = np.array([0.5, 0.5, 0.5, 0.5])
    >>> dq1 = qrot(1, 0.001)
    >>> dq2 = qrot(2, 0.05)
    >>> quat2 = np.column_stack((quat_mult(dq1,quat1), quat_mult(dq2,quat1)))
    >>> (theta, comp) = quat_angle_diff(quat1, quat2)
    >>> print(theta) # doctest: +NORMALIZE_WHITESPACE
    [0.001  0.05 ]
    >>> print(comp) # doctest: +NORMALIZE_WHITESPACE
    [[0.001  0.   ]
     [0.     0.05 ]
     [0.     0.   ]]

    """
    # check assertions
    _quat_assertions(quat1)
    _quat_assertions(quat2)

    # check for null quaternions
    if quat1.size == 0 or quat2.size == 0:
        return (None, np.array([None, None, None]))

    # calculate delta quaternion
    dq = quat_mult(quat2, quat_inv(quat1))

    # pull vector components out of delta quaternion
    if dq.ndim == 1:
        dv = dq[0:3]
    else:
        dv = dq[0:3, :]

    # sum vector components to get sin(theta/2)^2
    mag2 = np.sum(dv**2, axis=0)

    # take square root to get sin(theta/2)
    mag = np.sqrt(mag2)

    # take inverse sine to get theta/2
    theta_over_2 = np.arcsin(mag)

    # multiply by 2 to get theta
    theta = 2*theta_over_2

    # set any magnitude that is identically 0 to be 1 instead
    # to avoid a divide by zero warning.
    if np.isscalar(mag):
        if mag == 0:
            mag = 1
    else:
        mag[mag == 0] = 1

    # normalize vector components
    nv = dv / mag

    # find angle expressed in x, y, z components based on normalized vector
    comp = nv * theta

    return (theta, comp)

#%% Functions - quat_from_euler
def quat_from_euler(angles, seq=None):
    r"""
    Convert set(s) of euler angles to quaternion(s).

    Assumes angles are of (3 1 2) euler order and converts accordingly unless the
    optional "seq" argument defines a different euler order. This function will
    also take more than three angle sequences if desired.

    Parameters
    ----------
    angles : ndarray, (A, N)
        Euler angels [rad]
    seq : ndarray, (A, 1), optional
        Euler angle sequence, where:
            1 = X axis, or roll
            2 = Y axis, or pitch
            3 = Z axis, or yaw

    Returns
    -------
    quat : ndarray (4,) or (4, N)
        quaternion representing the euler rotation(s)

    Notes
    -----
    #.  Adapted from GARSE by David C. Stauffer in April 2015.
    #.  This function will take one angle sequence, but an arbitrary number of angles.
    #.  Enumerated values are some selective permutation of (1, 2, 3) without successive
            repetition such as (3, 1, 2) or (3, 1, 3) but not (3, 1, 1) wherein 1, 1 is a successive
            repetition.  By default, it expects (3, 1, 2).

    Examples
    --------
    >>> from dstauffman import quat_from_euler
    >>> import numpy as np
    >>> a   = np.array([0.01, 0.02, 0.03])
    >>> b   = np.array([0.04, 0.05, 0.06])
    >>> angles = np.column_stack((a, b))
    >>> seq = np.array([3, 2, 1])
    >>> quat = quat_from_euler(angles, seq)
    >>> print(quat) # doctest: +NORMALIZE_WHITESPACE
    [[0.01504849  0.03047982]
     [0.00992359  0.02438147]
     [0.00514916  0.02073308]
     [0.99982426  0.99902285]]

    """
    # check for optional inputs
    if seq is None:
        seq = np.array([3, 1, 2])
    # check for different combinations of angles (scalar, 1D, 2D)
    try:
        ndim = angles.ndim
    except AttributeError:
        if np.isscalar(angles):
            # assume this is an integer and turn into ndarray with one element
            angles = np.array([angles])
            ndim = 1
        else:
            raise # pragma: no cover
    # need this check for scalar float64 objects (but not float objects, don't know why)
    if ndim == 0:
        angles = np.array([angles])
        ndim = 1
    if ndim == 1:
        # angles is a 1D, note and make 2D
        is_vector = True
        angles = np.expand_dims(angles, axis=1)
    elif ndim == 2:
        # note that was 2D
        is_vector = False
    else:
        raise ValueError('Unexpected number of dimensions in angle: "{}"'.format(ndim))
    # get the number of quaternions to end up making
    num = angles.shape[1]
    # initialize output
    quat = np.zeros((QUAT_SIZE, num))
    # check that seq is iterable
    try:
        len(seq)
    except TypeError:
        seq = np.array([seq])
    # loop through quaternions
    for i in range(num):
        q_temp = np.array([0, 0, 0, 1])
        # apply each rotation
        for j in range(len(seq)):
            q_single = qrot(seq[j], angles[j, i])
            q_temp = quat_mult(q_temp, q_single)
        # save output
        quat[:, i] = q_temp
    # optionally flatten result
    if is_vector and num == 1:
        quat = quat.flatten()
    return quat

#%% Functions - quat_interp
def quat_interp(time, quat, ti, inclusive=True):
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
    inclusive : bool {True, False}, optional
        Whether ti must be only inclusive to the `time` vector.

    Returns
    -------
    qout : ndarray (4, B)
        interpolated quaternion at ti

    Notes
    -----
    #.  Adapted from GARSE by David C. Stauffer in April 2015.

    Examples
    --------
    >>> from dstauffman import quat_interp, qrot
    >>> import numpy as np
    >>> time  = np.array([1, 3, 5])
    >>> quat = np.column_stack((qrot(1, 0), qrot(1, np.pi/2), qrot(1, np.pi)))
    >>> ti = np.array([1, 2, 4.5, 5])
    >>> qout = quat_interp(time, quat, ti)
    >>> print(np.array_str(qout, precision=8, suppress_small=True)) # doctest: +NORMALIZE_WHITESPACE
    [[0. 0.38268343 0.98078528 1. ]
     [0. 0.         0.         0. ]
     [0. 0.         0.         0. ]
     [1. 0.92387953 0.19509032 0. ]]

    """
    # Initializations
    # number of data points to find
    try:
        num   = len(ti)
    except TypeError:
        if np.isscalar(ti):
            ti = np.array([ti])
            num = len(ti)
        else: # pragma: no cover
            raise

    # initialize output
    qout  = np.full((QUAT_SIZE, num), np.nan, dtype=float)

    # Simple cases
    if num == 0:
        # optimization for when ti is empty
        return qout
    elif num == 1:
        # optimization for simple use case(s), where ti is a scalar and contained in time
        if ti in time:
            ix = np.where(ti == time)[0]
            if not ix: # pragma: no branch
                qout = quat[:, ix]
                return qout

    # Check time bounds
    # check for desired times that are outside the time vector
    ix_exclusive = (ti < time[0]) | (ti > time[-1])
    if np.any(ix_exclusive):
        if inclusive:
            raise ValueError('Desired time not found within input time vector.')
        else:
            print('Desired time not found within input time vector.')

    # Given times
    # find desired points that are contained in input time vector
    ix_known = np.in1d(ti, time, assume_unique=True)
    ix_input = np.in1d(time, ti, assume_unique=True)

    # set quaternions directly to known values
    qout[:, ix_known] = quat[:, ix_input]

    # find other points to be calculated
    ix_calc = ~ix_known & ~ix_exclusive

    # Calculations
    # find index within time to surround ti, accounting for the end of the vector
    index = np.full(num, INT_TOKEN, dtype=int)
    for i in np.flatnonzero(ix_calc):
        temp = np.flatnonzero(ti[i] <= time)
        if temp[0] != len(time)-1:
            index[i] = temp[0] + 1
        else:
            index[i] = temp[0]

    # remove points that are NaN, either they weren't in the time vector, or they were next to a
    # drop out and cannot be interpolated.
    index = index[index != INT_TOKEN]
    # pull out bounding times and quaternions
    t1 = time[index-1]
    t2 = time[index]
    q1 = quat[:, index-1]
    q2 = quat[:, index]
    # calculate delta quaternion
    dq12       = quat_norm(quat_mult(q2, quat_inv(q1)))
    # find delta quaternion axis of rotation
    vec        = dq12[0:3, :]
    norm_vec   = np.sqrt(np.sum(vec**2, axis=0))
    # check for zero norm vectors
    norm_fix   = norm_vec
    norm_fix[norm_fix == 0] = 1
    ax         = vec / norm_fix
    # find delta quaternion rotation angle
    ang        = 2*np.arcsin(norm_vec)
    # scale rotation angle based on time
    scaled_ang = ang*(ti[ix_calc]-t1) / (t2-t1)
    # find scaled delta quaternion
    dq         = np.concatenate((ax*np.sin(scaled_ang/2), np.expand_dims(np.cos(scaled_ang/2),0)), axis=0)
    # calculate desired quaternion
    qout_temp  = quat_norm(quat_mult(dq,q1))
    # store into output structure
    qout[:, ix_calc] = qout_temp

    # Sign convention
    # Enforce sign convention on scalar quaternion element.
    # Scalar element (fourth element) of quaternion must not be negative.
    # So change sign on entire quaternion if qout(4) is less than zero.
    negs = np.zeros(qout.shape[1], dtype=bool)
    np.less(qout[3, :], 0, out=negs, where=~np.isnan(qout[3, :]))
    qout[:, negs] = -qout[:, negs]

    return qout

#%% Functions - quat_inv
def quat_inv(q1):
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
    quat_norm, quat_mult, quat_prop, quat_times_vector, quat_to_dcm, quat_to_euler,
    quat_from_euler

    Notes
    -----
    #.  Adapted from GARSE by David C. Stauffer in April 2015.

    Examples
    --------
    >>> from dstauffman import qrot, quat_inv
    >>> from numpy import pi
    >>> q1 = qrot(1, pi/2)
    >>> q2 = quat_inv(q1)
    >>> print(q2) # doctest: +NORMALIZE_WHITESPACE
    [-0.70710678 -0. -0. 0.70710678]

    """
    # check for empty case
    if q1.size == 0:
        q2 = np.zeros(q1.shape)
        return q2
    # size check
    _quat_assertions(q1)
    # invert the quaternions
    if q1.ndim == 1:
        # optimized single quaternion case
        q2 = q1 * np.array([-1, -1, -1, 1])
    else:
        # general case
        q2 = np.concatenate((-q1[0, :], -q1[1, :], -q1[2, :], q1[3, :]), axis=0).reshape(QUAT_SIZE, q1.shape[1])
    _quat_assertions(q2)
    return q2

#%% Functions - quat_mult
def quat_mult(a, b):
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
    >>> from dstauffman import qrot, quat_mult
    >>> from numpy import pi
    >>> a = qrot(1, pi/2)
    >>> b = qrot(2, pi)
    >>> c = quat_mult(a, b)
    >>> print(c) # doctest: +NORMALIZE_WHITESPACE
    [ 4.32978028e-17 7.07106781e-01 -7.07106781e-01 4.32978028e-17]

    """
    # check for vectorized inputs
    is_single_a = a.ndim == 1 and a.size == QUAT_SIZE
    is_single_b = b.ndim == 1 and b.size == QUAT_SIZE
    # check for null case
    if a.size * b.size == 0:
        if min(a.shape[0], b.shape[0]) == 0:
            c = np.array([])
        else:
            if a.size == 0:
                c = np.zeros(a.shape)
            else:
                c = np.zeros(b.shape)
        _quat_assertions(c)
        return c
    # single quaternion inputs case
    if is_single_a and is_single_b:
        c = np.array([ \
            [ a[3],  a[2], -a[1],  a[0]], \
            [-a[2],  a[3],  a[0],  a[1]], \
            [ a[1], -a[0],  a[3],  a[2]], \
            [-a[0], -a[1], -a[2],  a[3]]]) @ b[:, np.newaxis]
        # flatten back to a 1D vector
        c = c.flatten()
        # enforce positive scalar component
        if c[3] < 0:
            c = -c
    # vectorized inputs
    else:
        # make A and B both 2D arrays
        if is_single_a:
            a = a[:, np.newaxis]
        if is_single_b:
            b = b[:, np.newaxis]
        # alias the rows
        a1 = a[0, :]
        a2 = a[1, :]
        a3 = a[2, :]
        a4 = a[3, :]
        b1 = b[0, :]
        b2 = b[1, :]
        b3 = b[2, :]
        b4 = b[3, :]
        # compute the combine multiplication result
        c = np.array([ \
             b1*a4 + b2*a3 - b3*a2 + b4*a1, \
            -b1*a3 + b2*a4 + b3*a1 + b4*a2, \
             b1*a2 - b2*a1 + b3*a4 + b4*a3, \
            -b1*a1 - b2*a2 - b3*a3 + b4*a4])
        # enforce positive scalar component
        c[:, c[3, :]<0] = -c[:, c[3, :]<0]
    c = quat_norm(c)
    _quat_assertions(c)
    return c

#%% Functions - quat_norm
def quat_norm(x):
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
    #.  Adapted from GARSE by David C. Stauffer in April 2015.

    Examples
    --------
    >>> from dstauffman import quat_norm
    >>> import numpy as np
    >>> x = np.array([0.1, 0, 0, 1])
    >>> y = quat_norm(x)
    >>> print(y) # doctest: +NORMALIZE_WHITESPACE
    [0.09950372 0. 0. 0.99503719]

    """
    # divide input by its column vector norm
    y = x / np.sqrt(np.sum(x*x, axis=0))
    _quat_assertions(y)
    return y

#%% Functions - quat_prop
def quat_prop(quat, delta_ang, renorm=True):
    r"""
    Approximate propagation of a quaternion using a small delta angle.

    Parameters
    ----------
    quat : ndarray, (4, 1)
        normalized input quaternion
    delta_ang : ndarray, (3, 1)
        delta angles in x, y, z order [rad]
    renorm : bool {True, False}, optional

    Returns
    -------
    quat_new : ndarray, (4, 1)
        propagated quaternion, optionally re-normalized

    See Also
    --------
    quat_mult, quat_inv, quat_norm, quat_times_vector, quat_to_dcm, quat_to_euler, quat_from_euler

    Notes
    -----
    #.  Adapted from GARSE by David C. Stauffer in April 2015.

    Examples
    --------
    >>> from dstauffman import quat_prop
    >>> import numpy as np
    >>> quat      = np.array([0, 0, 0, 1])
    >>> delta_ang = np.array([0.01, 0.02, 0.03])
    >>> quat_new  = quat_prop(quat, delta_ang)
    >>> print(quat_new) # doctest: +NORMALIZE_WHITESPACE
    [0.00499913  0.00999825  0.01499738  0.99982505]

    """
    #compute angle rate matrix
    omega = np.array([ \
        [      0      ,   delta_ang[2],   -delta_ang[1],   delta_ang[0]], \
        [-delta_ang[2],        0      ,    delta_ang[0],   delta_ang[1]], \
        [ delta_ang[1],  -delta_ang[0],        0       ,   delta_ang[2]], \
        [-delta_ang[0],  -delta_ang[1],   -delta_ang[2],        0      ]])
    #compute delta quaternion
    delta_quaternion = 0.5 * omega.dot(quat)
    # propagate over delta
    quat_new = quat + delta_quaternion
    # renormalize and return
    if renorm:
        quat_new = quat_norm(quat_new)
    _quat_assertions(quat_new)
    return quat_new

#%% Functions - quat_times_vector
def quat_times_vector(quat, v):
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
    >>> from dstauffman import quat_times_vector
    >>> import numpy as np
    >>> quat = np.array([[0, 1, 0, 0], [1, 0, 0, 0]]).T
    >>> v = np.array([[1, 0, 0], [2, 0, 0]]).T
    >>> vec = quat_times_vector(quat, v)
    >>> print(vec) # doctest: +NORMALIZE_WHITESPACE
    [[-1.  2.]
     [ 0.  0.]
     [ 0.  0.]]

    """
    # assume single inputs until proven otherwise
    is_single = True
    # determine input sizes
    if quat.ndim == 1:
        quat = quat[:, np.newaxis]
    else:
        is_single = False
    if v.ndim == 1:
        v = v[:, np.newaxis]
    else:
        is_single = False
    # Multiple quaternions, multiple vectors
    qv  = np.array([ \
        quat[1, :]*v[2, :] - quat[2, :]*v[1, :], \
        quat[2, :]*v[0, :] - quat[0, :]*v[2, :], \
        quat[0, :]*v[1, :] - quat[1, :]*v[0, :]]) # TODO: use cross product?
    vec = v + 2*(-(np.ones((3, 1)).dot(np.expand_dims(quat[3, :], 0))) * qv + \
        np.array([ \
            quat[1, :]*qv[2, :] - quat[2, :]*qv[1, :], \
            quat[2, :]*qv[0, :] - quat[0, :]*qv[2, :], \
            quat[0, :]*qv[1, :] - quat[1, :]*qv[0, :]])) # TODO: check this, and use cross product?
    if is_single:
        vec = vec.flatten()
    return vec

#%% Functions - quat_to_dcm
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
    >>> from dstauffman import quat_to_dcm
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

#%% Functions - quat_to_euler
def quat_to_euler(quat, seq=None):
    r"""
    Convert quaternion to Euler angles for one of 6 input angle sequences.

    Parameters
    ----------
    quat : ndarray, (4,) or (4, N)
        quaternion
    seq : ndarray, {(3, 1, 2), (1, 2, 3), (2, 3, 1), (1, 3, 2), (2, 1, 3), (3, 2, 1)}
        euler angle sequence, where:
            1 = X axis, or roll
            2 = Y axis, or pitch
            3 = Z axis, or yaw

    Returns
    -------
    euler : ndarray, (3, N)
        Euler angles [rad]

    See Also
    --------
    quat_mult, quat_inv, quat_norm, quat_prop, quat_times_vectore, quat_from_euler

    Notes
    -----
    #.  Adapted from GARSE by David C. Stauffer in April 2015.

    References
    ----------
    #.  Appendix E of Wertz, page 764
    #.  Appendix I of Kane, Likins, and Levinson, page 423

    Examples
    --------
    >>> from dstauffman import quat_to_euler
    >>> import numpy as np
    >>> quat = np.array([[0, 1, 0, 0], [0, 0, 1, 0]]).T
    >>> seq = [3, 1, 2]
    >>> euler = quat_to_euler(quat, seq)
    >>> print(euler) # doctest: +NORMALIZE_WHITESPACE
    [[-0.         -3.14159265]
     [ 0.          0.        ]
     [ 3.14159265 -0.        ]]

    """
    # check for optional inputs
    if seq is None:
        seq = np.array([3, 1, 2])
    # assert quaternion checks
    _quat_assertions(quat)
    assert len(seq) == 3, 'Sequence must have len of 3, not "{}"'.format(len(seq))
    if quat.ndim == 1:
        # quat is a 1D
        is_vector = True
        quat = np.expand_dims(quat, axis=1)
    else:
        # note that was 2D
        is_vector = False

    # initialize output
    num   = quat.shape[1]
    euler = np.zeros((3, num))

    # Loop through quaternions
    for i in range(num):
        # calculate DCM from quaternion
        dcm = quat_to_dcm(quat[:, i])
        # build sequence str
        seq_str = str(int(seq[0])) + str(int(seq[1])) + str(int(seq[2]))
        # calculate terms based on sequence order
        if seq_str == '123':
            #Identical to KLL pg 423
            c2_c3                       =  dcm[0, 0]
            s1_s2_c3_plus_s3_c1         =  dcm[1, 0]
            minus_c1_s2_c3_plus_s3_s1   =  dcm[2, 0]
            minus_c2_s3                 =  dcm[0, 1]
            minus_s1_s2_s3_plus_c3_c1   =  dcm[1, 1]
            c1_s2_s3_plus_c3_s1         =  dcm[2, 1]
            s2                          =  dcm[0, 2]
            s1_c2                       =  dcm[1, 2]
            c1_c2                       =  dcm[2, 2]
            group = 1
        elif seq_str == '231':
            c1_c2                       =  dcm[0, 0]
            minus_c1_s2_c3_plus_s3_s1   =  dcm[0, 1]
            c1_s2_s3_plus_c3_s1         =  dcm[0, 2]
            s2                          =  dcm[1, 0]
            c2_c3                       =  dcm[1, 1]
            minus_c2_s3                 =  dcm[1, 2]
            s1_c2                       =  dcm[2, 0]
            s1_s2_c3_plus_s3_c1         =  dcm[2, 1]
            minus_s1_s2_s3_plus_c3_c1   =  dcm[2, 2]
            group = 1
        elif seq_str == '312':
            s1_s2_c3_plus_s3_c1         =  dcm[0, 2]
            minus_c1_s2_c3_plus_s3_s1   =  dcm[1, 2]
            minus_c2_s3                 =  dcm[2, 0]
            minus_s1_s2_s3_plus_c3_c1   =  dcm[0, 0]
            c1_s2_s3_plus_c3_s1         =  dcm[1, 0]
            s2                          =  dcm[2, 1]
            s1_c2                       =  dcm[0, 1]
            c1_c2                       =  dcm[1, 1]
            c2_c3                       =  dcm[2, 2]
            group = 1
        elif seq_str == '132':
            c2_c3                        =  dcm[0, 0]
            minus_c1_s2_c3_plus_s3_s1    =  dcm[1, 0]
            s1_s2_c3_plus_s3_c1          = -dcm[2, 0]
            s2                           = -dcm[0, 1]
            c1_c2                        =  dcm[1, 1]
            s1_c2                        =  dcm[2, 1]
            minus_c2_s3                  = -dcm[0, 2]
            c1_s2_s3_plus_c3_s1          = -dcm[1, 2]
            minus_s1_s2_s3_plus_c3_c1    =  dcm[2, 2]
            group = 2
        elif seq_str == '213':
            s1_s2_c3_plus_s3_c1          = -dcm[0, 1]
            minus_c1_s2_c3_plus_s3_s1    =  dcm[2, 1]
            minus_c2_s3                  = -dcm[1, 0]
            minus_s1_s2_s3_plus_c3_c1    =  dcm[0, 0]
            c1_s2_s3_plus_c3_s1          = -dcm[2, 0]
            s2                           = -dcm[1, 2]
            s1_c2                        =  dcm[0, 2]
            c1_c2                        =  dcm[2, 2]
            c2_c3                        =  dcm[1, 1]
            group = 2
        elif seq_str == '321':
            s1_s2_c3_plus_s3_c1          = -dcm[1, 2]
            minus_c1_s2_c3_plus_s3_s1    =  dcm[0, 2]
            minus_c2_s3                  = -dcm[2, 1]
            minus_s1_s2_s3_plus_c3_c1    =  dcm[1, 1]
            c1_s2_s3_plus_c3_s1          = -dcm[0, 1]
            s2                           = -dcm[2, 0]
            s1_c2                        =  dcm[1, 0]
            c1_c2                        =  dcm[0, 0]
            c2_c3                        =  dcm[2, 2]
            group = 2
        else:
            raise ValueError('Invalid axis rotation sequence: "{}"'.format(seq_str))

        # Compute angles
        if s1_c2 == 0 and c1_c2 == 0:
            theta1 = 0
        else:
            if group == 1:
                theta1 = np.arctan2(-s1_c2, c1_c2)
            else:
                theta1 = np.arctan2( s1_c2, c1_c2)
        # compute sin and cos
        s1 = np.sin(theta1)
        c1 = np.cos(theta1)
        # build remaining thetas
        s3     = s1_s2_c3_plus_s3_c1*c1       +  minus_c1_s2_c3_plus_s3_s1*s1
        c3     = minus_s1_s2_s3_plus_c3_c1*c1 +        c1_s2_s3_plus_c3_s1*s1
        theta3 = np.arctan2(s3, c3)
        c2     = c2_c3*c3 - minus_c2_s3*s3
        theta2 = np.arctan2(s2, c2)

        # Store output
        euler[:,i] = np.array([theta1, theta2, theta3])

    # optionally flatten result and then return answer
    if is_vector:
        euler = euler.flatten()
    return euler

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='tests.test_quat', exit=False)
    doctest.testmod(verbose=False)
