r"""
Contains generic vector utilities that can be independently defined and used by other modules.

Notes
-----
#.  Written by David C. Stauffer in March 2020.

"""

# %% Imports
from __future__ import annotations

import doctest
from typing import Callable, Literal, NotRequired, overload, TYPE_CHECKING, TypedDict, Unpack
import unittest

from nubs import ncjit

from dstauffman import HAVE_NUMPY, is_datetime, linear_interp, linear_lowpass_interp, unit

if HAVE_NUMPY:
    import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    _D = NDArray[np.datetime64]
    _N = NDArray[np.floating]
    _V = NDArray[np.floating]  # shape (3,)
    _DCM = NDArray[np.floating]  # shape (3, 3)
    _FN = float | np.floating | _N
    _Lists = list[_N] | tuple[_N, ...] | _N

    class _InterpKwArgs(TypedDict):
        btype: NotRequired[Literal["linear", "lowpass", "highpass", "bandpass", "bandstop"]]
        left: NotRequired[int | float | None]
        right: NotRequired[int | float | None]
        analog: NotRequired[bool]
        filt_order: NotRequired[int]
        filt_freq: NotRequired[float]
        filt_samp: NotRequired[float]


# %% Functions - rot
@ncjit
def rot(axis: int, angle: float) -> _DCM:
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


# %% Functions - drot
@ncjit
def drot(axis: int, angle: float) -> _DCM:
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


# %% Functions - vec_cross
@ncjit
def vec_cross(vec: _V) -> _DCM:
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


# %% Functions - vec_angle
def vec_angle(vec1: _Lists, vec2: _Lists, use_cross: bool = True, normalized: bool = True) -> float | _N:
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
    temp = np.sum(dot_prod, axis=0)
    # handle small floating point errors (but let others still crash)
    if temp.ndim == 0:
        if temp > 1.0 and temp < 1 + 3 * np.finfo(float).eps:  # pylint: disable=chained-comparison
            temp = 1.0
    else:
        temp[(temp > 1.0) & (temp < 1 + 3 * np.finfo(float).eps)] = 1.0
    dot_result = np.arccos(temp)  # np.arccos(temp, out=np.full(temp.shape, np.nan), where=temp <= 1.0)
    if not use_cross:
        return dot_result  # type: ignore[no-any-return]
    # if desired, use cross product result, which is more accurate for small differences, but has
    # an ambiguity for angles greater than pi/2 (90 deg).  Use the dot product result to resolve
    # the ambiguity.
    cross_prod: _N
    if vec1.shape[0] == 2 and vec2.shape[0] == 2:
        cross_prod = (vec1.T[..., 0] * vec2.T[..., 1] - vec1.T[..., 1] * vec2.T[..., 0]).T
    else:
        cross_prod = np.cross(vec1.T, vec2.T).T
    temp = np.sqrt(np.sum(cross_prod**2, axis=0))
    # return dot product result for sums greater than 1 (due to small numerical inconsistencies near 180 deg separation)
    if temp.ndim == 0:
        # Note: this seems like a bug in numpy to have to have this branch
        cross_result = np.arcsin(temp) if temp <= 1.0 else dot_result
    else:
        cross_result = np.arcsin(temp, out=dot_result.copy(), where=temp <= 1.0)
    return np.where(dot_result > np.pi / 2, np.pi - cross_result, cross_result)


# %% Functions - cart2sph
@overload
def cart2sph(x: float, y: float, z: float) -> tuple[float, float, float]: ...
@overload
def cart2sph(x: np.floating, y: np.floating, z: np.floating) -> tuple[np.floating, np.floating, np.floating]: ...
@overload
def cart2sph(x: _N, y: _N, z: _N) -> tuple[_N, _N, _N]: ...
@ncjit
def cart2sph(x: _FN, y: _FN, z: _FN) -> tuple[_FN, _FN, _FN]:
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


# %% Functions - sph2cart
@overload
def sph2cart(az: float, el: float, rad: float) -> tuple[float, float, float]: ...
@overload
def sph2cart(az: np.floating, el: np.floating, rad: np.floating) -> tuple[np.floating, np.floating, np.floating]: ...
@overload
def sph2cart(az: _N, el: _N, rad: _N) -> tuple[_N, _N, _N]: ...
@ncjit
def sph2cart(az: _FN, el: _FN, rad: _FN) -> tuple[_FN, _FN, _FN]:
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


# %% Funcctions - rv2dcm
@ncjit
def rv2dcm(vec: _V) -> _DCM:
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
    dcm: _DCM = np.eye(3)
    # Use np.dot instead of np.inner here as they are the same for 1D arrays, and np.dot compiles with numba
    mag = np.sqrt(np.dot(vec, vec))
    if mag != 0:
        v = vec / mag
        c = np.cos(mag)
        s = np.sin(mag)
        dcm *= c
        dcm += -s * vec_cross(v) + (1 - c) * np.outer(v, v)
    return dcm


# %% interp_vector
def interp_vector(
    x: _N | _D,
    xp: _N | _D,
    yp: _N | _V,
    *,
    assume_sorted: bool = False,
    extrapolate: bool = False,
    **kwargs: Unpack[_InterpKwArgs],
) -> _V:
    r"""
    Interpolates each component of a vector, with options for convert numpy datetimes to int64.

    Parameters
    ----------
    x : array_like
        The x-coordinates at which to evaluate the interpolated values.
    xp: 1-D sequence of floats
        The x-coordinates of the data points, must be increasing if argument period is not specified. Otherwise, xp is internally sorted after normalizing the periodic boundaries with xp = xp % period.
    yp: 1-D sequence of float or complex
        The y-coordinates of the data points, same length as xp.
    assume_sorted : bool, optional, default is False
        Whether you can assume the data is sorted and do simpler (i.e. faster) calculations
    extrapolate : bool, optional, default is False
        Whether to allow function to extrapolate data on either end
    left: int or float, optional, default is yp[0]
        Value to use for any value less than all points in xp
    right: int or float, optional, default is yp[-1]
        Value to use for any value greater than all points in xp
    filt_order : int, optional, default is 2
        Low pass filter order
    filt_freq : float, optional, default is 0.01
        Default filter frequency
    filt_samp : float, optional, default is 1.0
        Default filter sample rate
    kwargs : Any
        Additional key-word arguments to pass through to scipy.signal.butter

    Returns
    -------
    y : float or complex (corresponding to yp) or ndarray
        The interpolated values, same shape as x.

    Notes
    -----
    #.  Written by David C. Stauffer in December 2024.

    Examples
    --------
    >>> from dstauffman.aerospace import interp_vector
    >>> import numpy as np
    >>> xp = np.array([0.0, 111.0, 2000.0, 5000.0])
    >>> yp = np.array([0.0, 1.0, -2.0, 3.0])
    >>> x = np.arange(0.0, 6001.0)
    >>> y = interp_vector(x, xp, yp, extrapolate=True)

    """
    x_is_datetime = is_datetime(x)
    assert x_is_datetime ^ (not is_datetime(xp)), "Both x and xp must be datetime if either one is."

    func: Callable[[_N | _D, _N | _D, _N | _V, Unpack[_InterpKwArgs]], _V]
    # fmt: off
    if "btype" not in kwargs or kwargs["btype"] == "linear":
        if x_is_datetime:
            func = lambda x, xp, yp, **kwargs: linear_interp(x.view(np.int64), xp.view(np.int64), yp, assume_sorted=assume_sorted, extrapolate=extrapolate, **kwargs)  # pylint: disable=unnecessary-lambda-assignment  # noqa: E731
        else:
            func = lambda x, xp, yp, **kwargs: linear_interp(x, xp, yp, assume_sorted=assume_sorted, extrapolate=extrapolate, **kwargs)  # pylint: disable=unnecessary-lambda-assignment  # noqa: E731
    else:
        if x_is_datetime:
            func = lambda x, xp, yp, **kwargs: linear_lowpass_interp(x.view(np.int64), xp.view(np.int64), yp, assume_sorted=assume_sorted, extrapolate=extrapolate, **kwargs)  # pylint: disable=unnecessary-lambda-assignment  # noqa: E731
        else:
            func = lambda x, xp, yp, **kwargs: linear_lowpass_interp(x, xp, yp, assume_sorted=assume_sorted, extrapolate=extrapolate, **kwargs)  # pylint: disable=unnecessary-lambda-assignment  # noqa: E731
    # fmt: on

    if yp.ndim == 1:
        # if not vectorized, then just call the functions directly
        return func(x, xp, yp, **kwargs)

    num_axis = yp.shape[0]
    num_pts = np.size(x)
    out = np.empty((num_axis, num_pts), dtype=yp.dtype)
    for i in range(num_axis):
        out[i, :] = func(x, xp, yp[i, :], **kwargs)
    return out


# %% Unit test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_aerospace_vectors", exit=False)
    doctest.testmod(verbose=False)
