# -*- coding: utf-8 -*-
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
    >>> from dstauffman import rot
    >>> import numpy as np
    >>> axis = 3
    >>> angle = np.pi/2
    >>> dcm = rot(axis, angle)

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

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_vector', exit=False)
    doctest.testmod(verbose=False)
