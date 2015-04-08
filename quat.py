# -*- coding: utf-8 -*-
r"""
Quat module file for the "dstauffman" library.  It contains generic quaternion
utilities that can be independently defined and used by other modules.

Notes
-----
#.  Written by David C. Stauffer in April 2015.
"""

#%% Imports
from __future__ import print_function
from __future__ import division
import doctest
import numpy as np
import unittest

#%% Functions - qrot
def qrot(k, a):
    r"""
    Construct a quaternion expressing a rotation about a single axis

    Parameters
    ----------
    k : int
        Axis about which rotation is being made, from {1, 2, 3}
            (1) for x-axis
            (2) for y-axis
            (3) for z-axis
    a : array_like
        angle of rotation in radians

    Returns
    -------
    q : ndarray, (4, N)
        quaternion representing the given rotations

    References
    ----------
    #.  Wertz, James R. (editor), Equations 12.11 in Parameterization of the Attitude,
            Section 12.1, Spacecraft Attitude Determination and Control,
            Kluwer Academic Publishers, 1978.

    Notes
    -----
    #.  Adapted from GARSE by David C. Stauffer in April 2015.

    Examples
    --------
    >>> from dstauffman import qrot
    >>> import numpy as np
    >>> q = qrot(3, np.pi/2)
    >>> print(q) # doctest: +NORMALIZE_WHITESPACE
    [ 0. 0. 0.70710678  0.70710678]

    """
    # assertions
    assert k in {1, 2, 3}
    # calculations
    if np.isscalar(a):
        # optimized scalar case
        q = np.array([0, 0, 0, np.cos(a/2)])
        q[k-1] = np.sin(a/2)
    else:
        # general case
        q = np.concatenate((np.zeros((len(a), 3)), np.vstack(np.cos(a/2))), axis=1).T
        q[k-1, :] = np.sin(a/2)
    return q


#%% Functions - quat_angle_diff
def quat_angle_diff(q1, q2):
    r"""
    Calculates the angular difference between two quaternions

    This function takes a two quaternions and calculates a delta quaternion between them.
    It then uses the delta quaternion to generate both a total angular difference, and an
    an angular difference expressed in X,Y,Z components based on the axis of rotation,
    expressed in the original frame of the q1 input quaternion.  This function uses full
    trignometric functions instead of any small angle approximations.

    Parameters
    ----------
    q1 : ndarray (4, N)
        quaternion one
    q2 : ndarray (4, N)
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
    q = [nx * sin(theta/2);
         ny * sin(theta/2);
         nz * sin(theta/2);
           cos(theta/2)  ];
    Where: <nx,ny,nz> are the three components of a unit vector of rotation axis and
           theta is the angle of rotation

    Notes
    -----
    #.  Adapted from GARSE by David C. Stauffer in April 2015.

    Examples
    --------
    >>> from dstauffman import qrot, quat_mult, quat_angle_diff
    >>> import numpy as np
    >>> q1    = np.array([0.5, 0.5, 0.5, 0.5])
    >>> dq1   = qrot(1, 0.001)
    >>> dq2   = qrot(2, 0.05)
    >>> q2    = [quat_mult(dq1,q1), quat_mult(dq2,q1)]
    >>> (theta, comp) = quat_angle_diff(q1,q2)
    >>> print(theta) # doctest: +NORMALIZE_WHITESPACE
    []
    >>> print(comp) # doctest: +NORMALIZE_WHITESPACE
    []

    """
    # calculate delta quaternion
    dq = quat_mult(q2,quat_inv(q1));

    # pull vector components out of delta quaternion
    dv = dq[0:2, :]

    # sum vector components to get sin(theta/2)^2
    mag2 = np.sum(dv**2, axis=0)

    # take square root to get sin(theta/2)
    mag = np.sqrt(mag2)

    # take inverse sine to get theta/2
    theta_over_2 = np.asin(mag)

    # multiply by 2 to get theta
    theta = 2*theta_over_2

    # set any magnitude that is identically 0 to be 1 instead
    # to avoid a divide by zero warning.
    mag[mag == 0] = 1

    # normalize vector components
    nv = dv / mag

    # find angle expressed in x,y,z components based on normalized vector
    comp = nv * theta

    return (theta, comp)

#%% Functions - quat_from_euler
def quat_from_euler(angles, seq=[3, 1, 2]):
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
    dq : ndarray (4, N)
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
    >>> angles = np.concatenate((a, b)).reshape(2, 3).T
    >>> seq = np.array([3, 2, 1])
    >>> q = quat_from_euler([a,b], seq)
    >>> print(q) # doctest: +NORMALIZE_WHITESPACE

    """
    # get number of angles
    n = angles.shape[1] # TODO: can be vector?
    # initialize output
    q = np.zeros((4, n))
    # loop through quaternions
    for i in range(n):
        q_temp = np.array([[0], [0], [0], [1]])
        # apply each rotation
        for j in range(len(seq)):
            q_single = qrot(seq[j], angles[j,i])
            q_temp = quat_mult(q_temp, q_single)
        # save output
        q[:, i] = q_temp
    return q

#%% Functions - quat_interp
def quat_interp(t, q, ti, inclusive=True):
    r"""
    Interpolate quaternions from a monotonic time series of quaternions.

    Parameters
    ----------
    t : ndarray, (A, )
        monotonically increasing time series [sec]
    q : ndarray, (4, A)
        quaternion series
    ti : ndarray (B, )
        desired time of interpolation [sec]
    inclusive : bool {True, False}, optional

    Returns
    -------
    qout : ndarray (4, B)
        interpolated quaternion at ti

    Notes
    -----
    #.  Adapted from GARSE by David C. Stauffer in April 2015.

    Examples
    --------
    >>> from dstauffman import quat_interp
    >>> import numpy as np
    >>> t  = np.array([1, 3, 5])
    >>> q  = np.array([[0, 0, 0, 1], [0, 0, 0.1961, 0.9806], [0.5, -0.5, -0.5, 0.5]]).T
    >>> ti = np.array([1, 2, 4.5, 5])
    >>> qout = quat_interp(t, q, ti)
    >>> print(qout) # doctest: +NORMALIZE_WHITESPACE
    []

    """
    # Initializations
    # number of data points to find
    num   = len(ti)

    # initialize output
    qout  = np.nan(4, num)

    # Scalar case
    # optimization for simple use case(s), where ti is a scalar and contained in t
    #if num == 0:
    #    return qout
    #elif num == 1:
    #    ix = find(ti == t,1,'first');
    #    if not isempty(ix):
    #        qout = q[:, ix]
    #        return qout
    #
    ## Check time bounds
    ## check for desired times that are outside the time vector
    #ix_exclusive = ti < t(1) | ti > t(end);
    #if any(ix_exclusive)
    #    if inclusive
    #        warning('utils:QuatInterpExtrap','Desired time not found within input time vector.');
    #    else
    #        error('Desired time not found within input time vector.');
    #    end
    #end
    #
    ## Given times
    ## find desired points that are contained in input time vector
    #[ix_known,ix_input] = ismember(ti,t);
    #
    ## set quaternions directly to known values
    #qout(:,ix_known) = q(:,ix_input(ix_known));
    #
    ## find other points to be calculated
    #ix_calc = ~ix_known & ~ix_exclusive;
    #
    ## Calculations
    ## find index within t to surround ti
    #index = nan(1,num);
    ## If not compiling, then you can do a for i = find(ix_calc) and skip the if ix_calc(i) line,
    ## which may make the non-compiled matlab version faster
    #for i = find(ix_calc)
    #    temp = find(ti(i) <= t,1,'first');
    #    if temp(1) ~= 1
    #        index(i) = temp(1);
    #    else
    #        index(i) = temp(1) + 1;
    ## If you want to compile this function, then you need this instead of the last for loop,
    ## plus a coder.extrinsic('warning') line.  These are not kept, because it makes the MATLAB-only
    ## version less efficient:
    ## for i = 1:length(ix_calc)
    ##     if ix_calc(i)
    ##         temp = find(ti(i) <= t,1,'first');
    ##         if temp(1) ~= 1
    ##             index(i) = temp(1);
    ##         else
    ##             index(i) = temp(1) + 1;
    ##         end
    ##     end
    ## end
    #
    ## remove points that are NaN, either they weren't in the time vector, or they were next to a drop out
    ## and cannot be interpolated.
    #index(isnan(index)) = [];
    ## pull out bounding times and quaternions
    #t1 = t(index-1);
    #t2 = t(index);
    #q1 = q(:,index-1);
    #q2 = q(:,index);
    ## calculate delta quaternion
    #dq12       = quat_norm(quat_mult(q2,quat_inv(q1)));
    ## find delta quaternion axis of rotation
    #vec        = dq12(1:3,:);
    #norm_vec   = realsqrt(sum(vec.^2));
    ## check for zero norm vectors
    #norm_fix   = norm_vec;
    #norm_fix(norm_fix == 0) = 1;
    #ax         = bsxfun(@rdivide,vec,norm_fix);
    ## find delta quaternion rotation angle
    #ang        = 2*asin(norm_vec);
    ## scale rotation angle based on time
    #scaled_ang = ang.*(ti(ix_calc)-t1)./(t2-t1);
    ## find scaled delta quaternion
    #dq         = [bsxfun(@times,ax,sin(scaled_ang/2)); cos(scaled_ang/2)];
    ## calculate desired quaternion
    #qout_temp  = quat_norm(quat_mult(dq,q1));
    ## store into output structure
    #qout(:,ix_calc) = qout_temp;
    #
    ## Sign convention
    ## Enforce sign convention on scalar quaternion element.
    ## Scalar element (fourth element) of quaternion must not be negative.
    ## So change sign on entire quaternion if qout(4) is less than zero.
    #qout(:,qout(4,:) < 0) = -qout(:,qout(4,:) < 0);

    return qout

#%% Functions - quat_inv
def quat_inv(q1):
    r"""
    Returns the inverse of a normalized quaternions

    Parameters
    ----------
    q1 : ndarray, (4xN)
        input quaternion

    Returns
    -------
    q2 : ndarray, (4xN)
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
    []

    """
    # invert the quaternions
    if q1.ndim == 1:
        # optimized single quaternion case
        q2 = q1 * np.array([-1, -1, -1, 1])
    else:
        # general case
        q2 = np.concatenate((-q1[0, :], -q1[1, :], -q1[2, :], q1[3, :]), axis=0)
    return q2

#%% Functions - quat_mult
def quat_mult(a, b):
    r"""
    Multiplies quaternions together.

    Parameters
    ----------
    a : ndarray, (4xN)
        input quaternion one
    b : ndarray, (4xN)
        input quaternion two

    Returns
    -------
    c : ndarray, (4xN)
        result of quaternion multiplication

    See Also
    --------
    quat_inv, quat_norm, quat_prop, quat_times_vector, quat_to_dcm, quat_to_euler,
    quat_from_euler

    Notes
    -----
    #.  Adapted from GARSE by David C. Stauffer in April 2015.

    #.  Each of (a, b) may be either a single quaternion (4, 1) or an array of quaternions (4, N).
        If `a` and `b` are both single quaternions, then return b*a. If either (but not both) is
        an array of quaternions, then return the product of the single quaternion times each element
        of the array. If both are rows of quaternions, multiply corresponding columns.
        `c` will have size (4, 1) in the first case, and (4, N) in the other cases.

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
    []

    """
    # check for vectorized inputs
    na = a.size
    nb = b.size
    # single quaternion inputs case
    if (na==4) & (nb==4):
        c = np.array([ \
            [ a[3],  a[2], -a[1],  a[0]], \
            [-a[2],  a[3],  a[0],  a[1]], \
            [ a[1], -a[0],  a[3],  a[2]], \
            [-a[0], -a[1], -a[2],  a[3]]]).dot(b) # TODO: replace with @ in Python 3.5
        # enforce positive scalar component
        if c[3] < 0:
            c = -c
    # vectorized inputs
    else:
        if a.ndim == 1:
            a = a[:, np.newaxis]
        a1 = a[0, :]
        a2 = a[1, :]
        a3 = a[2, :]
        a4 = a[3, :]
        if b.ndim == 1:
            b = b[:, np.newaxis]
        b1 = b[0, :]
        b2 = b[1, :]
        b3 = b[2, :]
        b4 = b[3, :]
        c = np.array([ \
            [ b1*a4 + b2*a3 - b3*a2 + b4*a1], \
            [-b1*a3 + b2*a4 + b3*a1 + b4*a2], \
            [ b1*a2 - b2*a1 + b3*a4 + b4*a3], \
            [-b1*a1 - b2*a2 - b3*a3 + b4*a4]]) # TODO: bug here
        c[:, c[3, :]<0] = -c[:, c[3, :]<0]
        if c.shape[1] == 1:
            c = c.flatten()
    return c

#%% Functions - quat_norm
def quat_norm(x):
    r"""
    Normalizes each column of the input matrix

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
    [ 0.09950372 0. 0. 0.99503719]

    """
    # divide input by its column vector norm
    y = x / np.sqrt(np.sum(x*x, axis=0))
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
    >>> quat      = np.array([0, 0, 0, 1])
    >>> delta_ang = np.array([0.01, 0.02, 0.03])
    >>> quat_new  = quat_prop(quat, delta_ang)
    >>> print(quat_new) # doctest: +NORMALIZE_WHITESPACE
    []

    """
    #compute angle rate matrix
    W = np.array([ \
        [      0      ,   delta_ang[2],   -delta_ang[1],   delta_ang[0]], \
        [-delta_ang[2],        0      ,    delta_ang[0],   delta_ang[1]], \
        [ delta_ang[1],  -delta_ang[0],        0       ,   delta_ang[2]], \
        [-delta_ang[0],  -delta_ang[1],   -delta_ang[2],        0      ]])
    #compute delta quaternion
    delta_quaternion = 0.5 * W.dot(quat)
    # propagate over delta
    quat_new = quat + delta_quaternion
    # renormalize and return
    if renorm:
        quat_new = quat_norm(quat_new)
    return quat_new

#%% Functions - quat_times_vector
def quat_times_vector(q, v):
    r"""
    Multiply quaternion(s) against vector(s)

    Parameters
    ----------
    q : ndarray, (4, N) or (4, )
        quaternion(s)
    v : ndarray, (3, N) or (3 ,)
        input vector(s)

    Returns
    -------
    vec : ndarray, (3, N)
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
    Steps to algorithm
    #.  qv = q(1:3) x v
    #.  vec = v + 2*[ -( q(4) * qv ) + (q(1:3) x qv) ]

    Examples
    --------
    >>> from dstauffman import quat_times_vector
    >>> import numpy as np
    >>> q = np.array([[0, 1, 0, 0], [1, 0, 0, 0]]).T
    >>> v = np.array([[1, 0, 0], [2, 0, 0]]).T
    >>> vec = quat_times_vector(q, v)
    >>> print(vec) # doctest: +NORMALIZE_WHITESPACE
    []

    """
    # determine input sizes
    if q.ndim == 1:
        q = q[:, np.newaxis]
    if v.ndim == 1:
        v = v[:, np.newaxis]
    # Multiple quaternions, multiple vectors
    qv  = np.array([ \
        [q[1, :]*v[2, :] - q[2, :]*v[1, :]], \
        [q[2, :]*v[0, :] - q[0, :]*v[2, :]], \
        [q[0, :]*v[1, :] - q[1, :]*v[0, :]]]) # TODO: use cross product?
    vec = v + 2*(-(np.ones((3,1)).dot(q[3, :])) * qv + \
        np.array([ \
            [q[1, :]*qv[2, :] - q[2, :]*qv[1, :]], \
            [q[2, :]*qv[0, :] - q[0, :]*qv[2, :]], \
            [q[0, :]*qv[1, :] - q[1, :]*qv[0, :]]])) # TODO: check this, and use cross product?
    return vec

#%% Functions - quat_to_dcm
def quat_to_dcm(q):
    r"""
    Converts quaternion to a direction cosine matrix

    Parameters
    ----------
    q : ndarray (4, 1)
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
    >>> q = np.array([0.5, -0.5, 0.5, 0.5])
    >>> dcm = quat_to_dcm(q)
    >>> print(dcm) # doctest: +NORMALIZE_WHITESPACE

    """
    #build dcm components
    dcm = np.zeros((3, 3))
    dcm[0, 0] = q[3]**2 + q[0]**2 - q[1]**2 - q[2]**2
    dcm[0, 1] = 2*(q[0]*q[1] + q[2]*q[3])
    dcm[0, 2] = 2*(q[0]*q[2] - q[1]*q[3])
    dcm[1, 0] = 2*(q[0]*q[1] - q[2]*q[3])
    dcm[1, 1] = q[3]**2 - q[0]**2 + q[1]**2 - q[2]**2
    dcm[1, 2] = 2*(q[1]*q[2] + q[0]*q[3])
    dcm[2, 0] = 2*(q[0]*q[2] + q[1]*q[3])
    dcm[2, 1] = 2*(q[1]*q[2] - q[0]*q[3])
    dcm[2, 2] = q[3]**2 - q[0]**2 - q[1]**2 + q[2]**2
    return dcm

#%% Functions - quat_to_euler
def quat_to_euler(q,seq):
    r"""
    Converts quaternion to Euler angles for one of 6 input angle sequences.

    Parameters
    ----------
    q : ndarray, (4, N)
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
    >>> q = np.array([[0, 1, 0, 0], [0, 0, 1, 0]]).T
    >>> seq = [3, 1, 2]
    >>> euler = quat_to_euler(q, seq)
    >>> print(euler) # doctest: +NORMALIZE_WHITESPACE
    []

    """

    # initialize output
    n     = q.shape[1] # TODO: vector case?
    euler = np.zeros((3, n))

    # Loop through quaternions
    for i in range(n):
        # calculate DCM from quaternion
        C = quat_to_dcm(q[:, i])
        # Find values of dir cosine matrix terms
        seq_str = str(int(seq[0])) + str(int(seq[1])) + str(int(seq[2]))
        # calculate terms based on sequence order
        if seq_str == '123':
            #Identical to KLL pg 423
            c2_c3                       =  C[0, 0]
            s1_s2_c3_plus_s3_c1         =  C[1, 0]
            minus_c1_s2_c3_plus_s3_s1   =  C[2, 0]
            minus_c2_s3                 =  C[0, 1]
            minus_s1_s2_s3_plus_c3_c1   =  C[1, 1]
            c1_s2_s3_plus_c3_s1         =  C[2, 1]
            s2                          =  C[0, 2]
            s1_c2                       =  C[1, 2]
            c1_c2                       =  C[2, 2]
            group = 1
        if seq_str == '231':
            c1_c2                       =  C[0, 0]
            minus_c1_s2_c3_plus_s3_s1   =  C[0, 1]
            c1_s2_s3_plus_c3_s1         =  C[0, 2]
            s2                          =  C[1, 0]
            c2_c3                       =  C[1, 1]
            minus_c2_s3                 =  C[1, 2]
            s1_c2                       =  C[2, 0]
            s1_s2_c3_plus_s3_c1         =  C[2, 1]
            minus_s1_s2_s3_plus_c3_c1   =  C[2, 2]
            group = 1
        if seq_str == '312':
            s1_s2_c3_plus_s3_c1         =  C[0, 2]
            minus_c1_s2_c3_plus_s3_s1   =  C[1, 2]
            minus_c2_s3                 =  C[2, 0]
            minus_s1_s2_s3_plus_c3_c1   =  C[0, 0]
            c1_s2_s3_plus_c3_s1         =  C[1, 0]
            s2                          =  C[2, 1]
            s1_c2                       =  C[0, 1]
            c1_c2                       =  C[1, 1]
            c2_c3                       =  C[2, 2]
            group = 1
        if seq_str == '132':
            c2_c3                        =  C[0, 0]
            minus_c1_s2_c3_plus_s3_s1    =  C[1, 0]
            s1_s2_c3_plus_s3_c1          = -C[2, 0]
            s2                           = -C[0, 1]
            c1_c2                        =  C[1, 1]
            s1_c2                        =  C[2, 1]
            minus_c2_s3                  = -C[0, 2]
            c1_s2_s3_plus_c3_s1          = -C[1, 2]
            minus_s1_s2_s3_plus_c3_c1    =  C[2, 2]
            group = 2
        if seq_str == '213':
            s1_s2_c3_plus_s3_c1          = -C[0, 1]
            minus_c1_s2_c3_plus_s3_s1    =  C[2, 1]
            minus_c2_s3                  = -C[1, 0]
            minus_s1_s2_s3_plus_c3_c1    =  C[0, 0]
            c1_s2_s3_plus_c3_s1          = -C[2, 0]
            s2                           = -C[1, 2]
            s1_c2                        =  C[0, 2]
            c1_c2                        =  C[2, 2]
            c2_c3                        =  C[1, 1]
            group = 2
        if seq_str == '321':
            s1_s2_c3_plus_s3_c1          = -C[1, 2]
            minus_c1_s2_c3_plus_s3_s1    =  C[0, 2]
            minus_c2_s3                  = -C[2, 1]
            minus_s1_s2_s3_plus_c3_c1    =  C[1, 1]
            c1_s2_s3_plus_c3_s1          = -C[0, 1]
            s2                           = -C[2, 0]
            s1_c2                        =  C[1, 0]
            c1_c2                        =  C[0, 0]
            c2_c3                        =  C[2, 2]
            group = 2
        else:
            raise ValueError('Invalid axis rotation sequence: ' + seq_str)

        # Compute angles
        if s1_c2 == 0 and c1_c2 == 0:
            theta1 = 0
        else:
            if group == 1:
                theta1 = np.artcan2(-s1_c2, c1_c2)
            else:
                theta1 = np.arctan2( s1_c2, c1_c2)
        # compute sin and cos
        s1 = np.sin(theta1)
        c1 = np.cos(theta1)
        # build remaining thetas
        s3     = s1_s2_c3_plus_s3_c1*c1       +  minus_c1_s2_c3_plus_s3_s1*s1;
        c3     = minus_s1_s2_s3_plus_c3_c1*c1 +        c1_s2_s3_plus_c3_s1*s1;
        theta3 = np.arctan2(s3, c3)
        c2     = c2_c3*c3 - minus_c2_s3*s3
        theta2 = np.arctan2(s2, c2)

        # Store output
        euler[:,i] = np.array([[theta1], [theta2], [theta3]])
    return euler

def concat_vectors(v1, v2):
    r"""
    Concatenate two vectors as if they were column vectors.

    Notes
    -----
    #.  TODO:Assumes 'C' vectors as opposed to 'F' vectors?
    """
    out = np.concatenate((v1, v2)).reshape(2, len(v1)).T
    return out

#%% Unit test
if __name__ == '__main__':
    q1 = qrot(1, np.pi/2)
    q2 = qrot(2, np.array([0, np.pi/2, np.pi/3]))
    q3 = quat_inv(q1)
    q4 = quat_mult(q1, q2)
    q5 = quat_norm(q4)
    dcm = quat_to_dcm(q1)

    print('q1 = ' + str(q1))
    print('q2 = ' + str(q2))
    print('q3 = ' + str(q3))
    print('q4 = ' + str(q4))
    print('q5 = ' + str(q5))
    print('dcm = ' + str(dcm))

    #unittest.main(module='tests.test_quat', exit=False)
    #doctest.testmod(verbose=False)
