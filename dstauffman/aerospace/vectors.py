r"""
Contains generic vector utilities that can be independently defined and used by other modules.

Notes
-----
#.  Written by David C. Stauffer in March 2020.
"""

#%% Imports
import doctest
import unittest

import numpy as np

from dstauffman import unit

#%% Functions - rot
def rot(axis, angle):
    r"""
    Direction cosine matrix for rotation about a single axis.

    Parameters
    ----------
    axis : int
        axis about which rotation is being made [enum]
             enumerated choices are (1, 2, or 3)
             corresponding to        x, y, or z axis
    angle : float
        angle of rotation [radians]

    Returns
    -------
    dcm : (3x3) ndarray
        direction cosine matrix

    See Also
    --------
    drot

    Notes
    -----
    1.  Incorporated by David C. Stauffer into dstauffman in March 2020 based on Matlab version.

    Examples
    --------
    Simple 90deg z-rotation
    >>> from dstauffman.aerospace import rot
    >>> import numpy as np
    >>> axis = 3
    >>> angle = np.pi/2
    >>> dcm = rot(axis, angle)
    >>> print(np.array_str(dcm, precision=4, suppress_small=True))
    [[ 0.  1.  0.]
     [-1.  0.  0.]
     [ 0.  0.  1.]]

    """
    # sines of angle
    ca = np.cos(angle)
    sa = np.sin(angle)

    # build direction cosine matrix
    if axis == 1:
        dcm = np.array([[1., 0., 0.], [0., ca, sa], [0., -sa, ca]], dtype=float)
    elif axis == 2:
        dcm = np.array([[ca, 0., -sa], [0., 1., 0.], [sa, 0., ca]], dtype=float)
    elif axis == 3:
        dcm = np.array([[ca, sa, 0.], [-sa, ca, 0.], [0., 0., 1.]], dtype=float)
    else:
        raise ValueError('Unexpected value for axis of: "{}".'.format(axis))
    return dcm

#%% Functions - vec_cross
def vec_cross(vec):
    r"""
    Returns the equivalent 3x3 matrix that would perform a cross product when multiplied.

    Parameters
    ----------
    vec : (3, ) ndarray
        3 element vector

    Returns
    -------
    (3, 3) ndarray
        3x3 matrix representation

    Notes
    -----
    #.  Written by David C. Stauffer in December 2018.

    Examples
    --------
    >>> from dstauffman.aerospace import vec_cross
    >>> import numpy as np
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([-2, -3, -4])
    >>> mat = vec_cross(a)
    >>> print(mat)
    [[ 0 -3  2]
     [ 3  0 -1]
     [-2  1  0]]

    """
    return np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])

#%% Functions - vec_angle
def vec_angle(vec1, vec2, use_cross=True, normalized=True):
    r"""
    Calculates the angle between two unit vectors.

    Parameters
    ----------
    vec1 : (3, ) or (3, N) ndarray
        Vector 1
    vec2 : (3, ) or (3, N) ndarray
        Vector 2
    use_cross : bool, optional, default is True
        Use cross product in calculation
    normalized : bool, optional, default is True
        Whether vectors are normalized

    Returns
    -------
    scalar or (N, ) ndarray
        Angle between vectors

    Notes
    -----
    #.  Note that the cross product method is more computationally expensive, but is more accurate
        for vectors with small angular differences, as the arcsin is Taylor series is order 2
        instead of order 1 error for the arcsin.
    #.  Written by David C. Stauffer in September 2020.

    Examples
    --------
    >>> from dstauffman.aerospace import rot, vec_angle
    >>> import numpy as np
    >>> vec1 = np.array([1., 0., 0.])
    >>> vec2 = rot(2, 1e-5) @ vec1
    >>> vec3 = np.array([0., 1., 0.])
    >>> angle = vec_angle(vec1, vec2)
    >>> print(angle)
    1e-05

    >>> angle2 = vec_angle(vec1, vec3, use_cross=False)
    >>> print(f'{angle2:12.12f}')
    1.570796326795

    """
    if not normalized:
        vec1 = unit(vec1)
        vec2 = unit(vec2)
    if use_cross:
        return np.arcsin(np.sqrt(np.sum(np.cross(vec1, vec2) ** 2, axis=-1)))
    return np.arccos(np.sum(np.multiply(vec1, vec2), axis=-1)) # Note: using sum and multiply instead of dot for 2D case

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_aerospace_vectors', exit=False)
    doctest.testmod(verbose=False)
