r"""
Contains generic vector utilities that can be independently defined and used by other modules.

Notes
-----
#.  Written by David C. Stauffer in March 2020.
"""

#%% Imports
from __future__ import annotations

import doctest
from typing import List, overload, Tuple, TYPE_CHECKING, Union
import unittest

from nubs import ncjit

from dstauffman import HAVE_NUMPY, unit

if HAVE_NUMPY:
    import numpy as np

#%% Constants
if TYPE_CHECKING:
    _Numbers = Union[float, np.ndarray]
    _Lists = Union[List[np.ndarray], Tuple[np.ndarray, ...], np.ndarray]
    _N = np.typing.NDArray[np.float64]

#%% Functions - rot
@ncjit
def rot(axis: int, angle: float) -> np.ndarray:
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
    # fmt: off
    if axis == 1:
        dcm = np.array([[1.0, 0.0, 0.0], [0.0,  ca,  sa], [0.0, -sa,  ca]])
    elif axis == 2:
        dcm = np.array([[ca,  0.0, -sa], [0.0, 1.0, 0.0], [ sa, 0.0,  ca]])
    elif axis == 3:
        dcm = np.array([[ca,  sa,  0.0], [-sa,  ca, 0.0], [0.0, 0.0, 1.0]])
    else:
        # Axis value not listed, so it can compile in nopython mode
        raise ValueError("Unexpected value for axis.")
    # fmt: on
    return dcm


#%% Functions - drot
@ncjit
def drot(axis: int, angle: float) -> np.ndarray:
    r"""
    Derivative of transformation matrix for rotation about a single axis.

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
    trans : (3x3) ndarray
        transformation matrix

    See Also
    --------
    rot

    Notes
    -----
    1.  Incorporated by David C. Stauffer into dstauffman in December 2020 based on Matlab version.

    Examples
    --------
    Simple 90deg z-rotation
    >>> from dstauffman.aerospace import drot
    >>> import numpy as np
    >>> axis = 3
    >>> angle = np.pi/2
    >>> dcm = drot(axis, angle)
    >>> print(np.array_str(dcm, precision=4, suppress_small=True))
    [[-1.  0.  0.]
     [-0. -1.  0.]
     [ 0.  0.  0.]]

    """
    # sines of angle
    ca = np.cos(angle)
    sa = np.sin(angle)

    # build direction cosine matrix
    # fmt: off
    if axis == 1:
        trans = np.array([[0.0, 0.0, 0.0], [0.0, -sa,  ca], [0.0, -ca, -sa]])
    elif axis == 2:
        trans = np.array([[-sa, 0.0, -ca], [0.0, 0.0, 0.0], [ ca, 0.0, -sa]])
    elif axis == 3:
        trans = np.array([[-sa,  ca, 0.0], [-ca, -sa, 0.0], [0.0, 0.0, 0.0]])
    else:
        raise ValueError("Unexpected value for axis.")
    # fmt: on
    return trans


#%% Functions - vec_cross
@ncjit
def vec_cross(vec: np.ndarray) -> np.ndarray:
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
    >>> vec = np.array([1, 2, 3])
    >>> skew = vec_cross(vec)
    >>> print(skew)
    [[ 0 -3  2]
     [ 3  0 -1]
     [-2  1  0]]

    """
    return np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])


#%% Functions - vec_angle
def vec_angle(vec1: _Lists, vec2: _Lists, use_cross: bool = True, normalized: bool = True) -> Union[float, _N]:
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
    >>> print(f"{angle2:12.12f}")
    1.570796326795

    """
    # process lists and tuples into numpy arrays
    if isinstance(vec1, (list, tuple)):
        vec1 = np.vstack(vec1).T
    if isinstance(vec2, (list, tuple)):
        vec2 = np.vstack(vec2).T
    # normalize if desired, otherwise assume it already is
    assert isinstance(vec1, np.ndarray)
    assert isinstance(vec2, np.ndarray)
    if not normalized:
        vec1 = unit(vec1)
        vec2 = unit(vec2)
    # calculate the result using dot products
    # Note: using sum and multiply instead of dot for 2D case
    dot_prod = np.multiply(vec1.T, np.conj(vec2).T).T
    dot_result = np.arccos(np.sum(dot_prod, axis=0))
    if not use_cross:
        return dot_result  # type: ignore[no-any-return]
    # if desired, use cross product result, which is more accurate for small differences, but has
    # an ambiguity for angles greater than pi/2 (90 deg).  Use the dot product result to resolve
    # the ambiguity.
    cross_prod = np.cross(vec1.T, vec2.T).T
    temp = np.sqrt(np.sum(cross_prod**2, axis=0))
    # return dot product result for sums greater than 1 (due to small numerical inconsistencies near 180 deg separation)
    cross_result = np.arcsin(temp, out=dot_result.copy(), where=temp <= 1.0)
    return np.where(dot_result > np.pi / 2, np.pi - cross_result, cross_result)


#%% Functions - cart2sph
@overload
def cart2sph(x: float, y: float, z: float) -> Tuple[float, float, float]:
    ...


@overload
def cart2sph(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


@ncjit
def cart2sph(x: _Numbers, y: _Numbers, z: _Numbers) -> Tuple[_Numbers, _Numbers, _Numbers]:
    r"""
    Converts cartesian X, Y, Z components to spherical Az, El, Radius.

    Parameters
    ----------
    x : ndarray of float
        X coordinate
    y : ndarray of float
        Y coordinate
    z : ndarray of float
        Z coordinate

    Returns
    -------
    az : ndarray of float
        Azimuth angle [radians]
    el : ndarray of float
        Elevation angle [radians]
    rad : ndarray of float
        Radius [ndim]

    Notes
    -----
    #.  Written by David C. Stauffer in December 2020.

    Examples
    --------
    >>> from dstauffman.aerospace import cart2sph
    >>> (az, el, rad) = cart2sph(3, 4, 5)

    """
    xy2 = x**2 + y**2
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.sqrt(xy2))
    rad = np.sqrt(xy2 + z**2)
    return (az, el, rad)


#%% Functions - sph2cart
@overload
def sph2cart(az: float, el: float, rad: float) -> Tuple[float, float, float]:
    ...


@overload
def sph2cart(az: np.ndarray, el: np.ndarray, rad: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ...


@ncjit
def sph2cart(az: _Numbers, el: _Numbers, rad: _Numbers) -> Tuple[_Numbers, _Numbers, _Numbers]:
    r"""
    Converts spherical Az, El and Radius to cartesian X, Y, Z components.

    Parameters
    ----------
    az : ndarray of float
        Azimuth angle [radians]
    el : ndarray of float
        Elevation angle [radians]
    rad : ndarray of float
        Radius [ndim]

    Returns
    -------
    x : ndarray of float
        X coordinate
    y : ndarray of float
        Y coordinate
    z : ndarray of float
        Z coordinate

    Notes
    -----
    #.  Written by David C. Stauffer in December 2020.

    Examples
    --------
    >>> from dstauffman.aerospace import sph2cart
    >>> from numpy import pi
    >>> az = pi/2
    >>> el = -pi
    >>> rad = 1
    >>> (x, y, z) = sph2cart(az, el, rad)

    """
    rcos_el = rad * np.cos(el)
    x = rcos_el * np.cos(az)
    y = rcos_el * np.sin(az)
    z = rad * np.sin(el)
    return (x, y, z)


#%% Funcctions - rv2dcm
@ncjit
def rv2dcm(vec: np.ndarray) -> np.ndarray:
    r"""
    Convert a rotation vector into a direction cosine matrix.

    Parameters
    ----------
    vec : (3, ) ndarray
        Rotation vector

    Returns
    -------
    dcm : (3, 3) ndarray
        Direction Cosine Matrix

    Notes
    -----
    #.  Written by Jason Hull in July 2008.
    #.  Translated from Matlab into Python by David C. Stauffer in January 2021.
    #.  The rotation vector `vec` defines the Euler rotation axis, and its magnitude defines then
        Euler rotation angle.

    Examples
    --------
    >>> from dstauffman.aerospace import rv2dcm
    >>> import numpy as np
    >>> vec = np.array([np.pi/2, 0., 0.])
    >>> dcm = rv2dcm(vec)
    >>> print(np.array_str(dcm, precision=4, suppress_small=True))
    [[ 1.  0.  0.]
     [ 0.  0.  1.]
     [ 0. -1.  0.]]

    """
    dcm: np.ndarray = np.eye(3)
    # Use np.dot instead of np.inner here as they are the same for 1D arrays, and np.dot compiles with numba
    mag = np.sqrt(np.dot(vec, vec))
    if mag != 0:
        v = vec / mag
        c = np.cos(mag)
        s = np.sin(mag)
        dcm *= c
        dcm += -s * vec_cross(v) + (1 - c) * np.outer(v, v)
    return dcm


#%% Unit test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_aerospace_vectors", exit=False)
    doctest.testmod(verbose=False)
