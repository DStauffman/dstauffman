r"""
Contains utilities related to Earth transformations.

Notes
-----
#.  Written by David C. Stauffer in June 2021.
"""

# %% Imports
from __future__ import annotations

import doctest
from typing import Literal, Optional, overload, Tuple, TYPE_CHECKING, Union
import unittest

from dstauffman import HAVE_NUMPY, M2FT

if HAVE_NUMPY:
    import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

    _N = NDArray[np.float64]

# %% Constants
# Earth's semi-major axis (ft)
_a = 6378137.0
# Inverse flattening
_finv = 298.257223563
# Earth's semi-minor axis (ft)
_b = _a * (1.0 - 1.0 / _finv)


# %% Functions - geod2ecf
@overload
def geod2ecf(
    lat: ArrayLike, lon: Literal[None] = ..., alt: Literal[None] = ..., *, units: str = ..., output: str = ...
) -> Union[_N, Tuple[_N, _N, _N]]: ...
@overload
def geod2ecf(lat: ArrayLike, lon: ArrayLike, alt: ArrayLike, *, units: str = ..., output: Literal["combined"] = ...) -> _N: ...
@overload
def geod2ecf(
    lat: ArrayLike, lon: ArrayLike, alt: ArrayLike, *, units: str = ..., output: Literal["split"]
) -> Tuple[_N, _N, _N]: ...
def geod2ecf(
    lat: ArrayLike,
    lon: Optional[ArrayLike] = None,
    alt: Optional[ArrayLike] = None,
    *,
    units: str = "m",
    output: str = "combined",
) -> Union[_N, Tuple[_N, _N, _N]]:
    r"""
    Converts geodetic latitude, longitude, and altitude to earth centered, earth fixed (ECF) coordinates.

    Assuming the X axis of the ECF frame is lined up with the prime meridian, the Z axis is through
    the north pole, and the Y axis completes a right handed coordinate system.

    Parameters
    ----------
    lat : (N, ) ndarray
        Geodetic latitude [rad]
    lon : (N, ) ndarray
        Geodetic longitude [rad]
    alt : (N, ) ndarray
        Geodetic altitude [m]
    units : str
        Units, default of "m" for meters, "ft" for feet is also valid
    output : str
        Whether output is "combined" for a (3, N) or "split" for (x, y, z) outputs

    Returns
    -------
    x : (3, N) ndarray or (N, ) if `output` == "split"
        Position XYZ (or X) vector [m]
    y : (N, ) ndarray, optional
        Position XYZ vector [m]
    z : (N, ) ndarray, optional
        Position XYZ vector [m]

    Notes
    -----
    #.  Written by David C. Stauffer in July 2021.  Includes options for "m" or "ft", and 3xN
        matrices or three 1xN vectors for both inputs and outputs.

    Examples
    --------
    >>> from dstauffman.aerospace import geod2ecf
    >>> import numpy as np
    >>> lla = np.array([np.pi/2, 0, 0])
    >>> xyz = geod2ecf(lla)
    >>> print(np.round(xyz))  # doctest: +NORMALIZE_WHITESPACE
    [ 0. 0. 6356752.]

    """
    # determine units (TODO: do I really want to support both?)
    if units == "m":
        a = _a
        b = _b
    elif units == "ft":
        a = _a * M2FT
        b = _b * M2FT
    else:
        raise ValueError(f'Unexpected value for units: "{units}"')
    # pull longitude and altitude out of lat if it is the only one given (assumed to be lla)
    if lon is None and alt is None:
        assert isinstance(lat, np.ndarray)
        lon = lat[1, ...]
        alt = lat[2, ...]
        lat = lat[0, ...]
    assert lon is not None
    assert alt is not None
    # intermediate calculations
    Ne = a**2 / np.sqrt(a**2 * np.cos(lat) ** 2 + b**2 * np.sin(lat) ** 2)
    # determine output shape
    shape = (3,) if np.size(lat) == 1 else (3, np.size(lat))
    xyz = np.full(shape, np.nan)
    # create output
    xyz[0, ...] = (Ne + alt) * np.cos(lat) * np.cos(lon)  # type: ignore[operator]
    xyz[1, ...] = (Ne + alt) * np.cos(lat) * np.sin(lon)  # type: ignore[operator]
    xyz[2, ...] = ((b / a) ** 2 * Ne + alt) * np.sin(lat)  # type: ignore[operator]
    if output == "combined":
        return xyz
    if output == "split":
        return (xyz[0, ...], xyz[1, ...], xyz[2, ...])
    raise ValueError(f'Unexpected value for output: "{output}"')


# %% Functions - ecf2geod
@overload
def ecf2geod(
    x: ArrayLike, y: Literal[None] = ..., z: Literal[None] = ..., *, units: str = ..., output: str = ..., algorithm: str = ...
) -> Union[_N, Tuple[_N, _N, _N]]: ...
@overload
def ecf2geod(
    x: ArrayLike, y: ArrayLike, z: ArrayLike, *, units: str = ..., output: Literal["combined"] = ..., algorithm: str = ...
) -> _N: ...
@overload
def ecf2geod(
    x: ArrayLike, y: ArrayLike, z: ArrayLike, *, units: str = ..., output: Literal["split"], algorithm: str = ...
) -> Tuple[_N, _N, _N]: ...
def ecf2geod(
    x: ArrayLike,
    y: Optional[ArrayLike] = None,
    z: Optional[ArrayLike] = None,
    *,
    units: str = "m",
    output: str = "combined",
    algorithm: str = "gersten",
) -> Union[_N, Tuple[_N, _N, _N]]:
    r"""
    Converts a vector of earth centered, earth fixed coordinates to geodetic latitude, longitude, and altitude.

    The "gersten" algorithm is based off of:
    1.  Chobotov, Vladimir, ed., "Orbital Mechanics", pp. 87-88, AIAA, 1991
    2.  Gersten, Robert, "Geodetic Sub-Latitude and Altitude of a Space
        Vehicle", Journal of the Astronautical Sciences, pp. 28-29

    Assumes:
    1.  World Geodetic System 1984 (WGS84) values.
    2.  Magnitude of r_ecf is nonzero (no divide-by-zero protection).
    3.  Solution is accurate to within limits imposed by elliptical Earth model.
    4.  Altitudes are not significantly negative (intended for spacecraft).

    The "sofair" algorithm is based off of this original paper:
    Sofair, I., Improved Method for Calculating Exact Geodetic Latitude and Altitude,
    Journal of Guidance, Control, and Dynamics, Vol. 20, No. 4, 1997, pp. 824826.
    and Sofair's revised version:
    Sofair, I., Improved Method for Calculating Exact Geodetic Latitude and Altitude Revisited,
    Journal of Guidance, Control, and Dynamics, Vol. 23, No. 2, 2000, pp. 369.

    Assumes:
    X axis of the ECEF frame is lined up with the prime meridian,
    the Z axis is through the north pole, and the Y axis completes a right handed
    coordinate system. It also assumes q > 0 in the algorithm which corresponds
    to the magnitude of x being greater than ~43 km.

    Parameters
    ----------
    x : (3, N) ndarray or (N, ) if `output` == "split"
        Position XYZ (or X) vector [m]
    y : (N, ) ndarray, optional
        Position XYZ vector [m]
    z : (N, ) ndarray, optional
        Position XYZ vector [m]
    units : str, optional, default is "m"
        Units, default of "m" for meters, "ft" for feet is also valid
    output : str, optional, default is "combined"
        Whether output is "combined" for a (3, N) or "split" for (x, y, z) outputs
    algorithm : str, optional, default is "gersten"
        Algorithm to use, from {"gersten", "sofair"}

    Returns
    -------
    lat : (N, ) ndarray
        Geodetic latitude [rad]
    lon : (N, ) ndarray
        Geodetic longitude [rad]
    alt : (N, ) ndarray
        Geodetic altitude [m]

    Notes
    -----
    #.  Written by David C. Stauffer in June 2021 based on the "gersten" version in ssc_toolbox
        and the "sofair" version in Redy.
    #.  Expanded by David C. Stauffer in July 2021 to allow for "m" or "ft", and 3xN matrices or
        three 1xN vectors for both inputs and outputs.

    Examples
    --------
    >>> from dstauffman.aerospace import ecf2geod
    >>> import numpy as np
    >>> xyz = np.array([6378137. + 10000, 0., 0.])
    >>> lla = ecf2geod(xyz)
    >>> print(lla)  # doctest: +NORMALIZE_WHITESPACE
    [ 0. 0. 10000.]

    """
    # determine units (TODO: do I really want to support both?)
    if units == "m":
        a = _a
        b = _b
    elif units == "ft":
        a = _a * M2FT
        b = _b * M2FT
    else:
        raise ValueError(f'Unexpected value for units: "{units}"')
    # pull Y and Z out of X if it is the only one given (assumed to be xyz)
    if y is None and z is None:
        assert isinstance(x, np.ndarray)
        y = x[1, ...]
        z = x[2, ...]
        x = x[0, ...]
    assert y is not None
    assert z is not None
    if algorithm == "sofair":
        # Sofair's algorithm
        # fmt: off
        e2   = 1 - (b / a) ** 2
        eps2 = (a / b) ** 2 - 1
        r0   = np.sqrt(np.square(x) + np.square(y))
        p    = np.abs(z) / eps2
        s    = r0**2 / e2 / eps2
        q    = p**2 - b**2 + s
        assert np.all(q > 0), "Q must be greater than zero, you need to avoid places very close to the center of the Earth."
        u    = p / np.sqrt(q)  # q > 0 required
        v    = b**2 * u**2 / q
        P    = 27 * v * s / q
        Q    = (np.sqrt(P + 1) + np.sqrt(P)) ** (2 / 3)
        t    = (1 + Q + 1 / Q) / 6
        c    = np.sqrt(u**2 - 1 + 2 * t)
        w    = (c - u) / 2
        zz   = np.sign(z) * np.sqrt(q) * (w + np.sqrt(np.sqrt(t**2 + v) - u * w - t / 2 - 1 / 4))
        Ne   = a * np.sqrt(1 + eps2 * zz**2 / b**2)
        ang  = (eps2 + 1) * zz / Ne
        lat  = np.arcsin(ang, out=np.full(ang.shape, np.nan), where=np.abs(ang) <= 1.0)
        lon  = np.arctan2(y, x)
        alt  = r0 * np.cos(lat) + z * np.sin(lat) - a**2 / Ne
        # fmt: on
    elif algorithm == "gersten":
        flattening = 1.0 / _finv
        fsq = flattening**2
        r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
        sin_delta = z / r  # sin(delta) = z/r
        delta = np.arcsin(sin_delta)  # delta is declination
        sin_sq_delta = sin_delta**2  # sin(delta)^2
        a_over_r = a / r
        # sin(2*delta)^2 = 4*sin(delta)^2*cos(delta)^2
        #                = 4*sin(delta)^2*(1-sin(delta)^2)
        sin_sq_2delta = 4.0 * sin_sq_delta * (1.0 - sin_sq_delta)
        ang = a_over_r * (flattening * np.sin(2 * delta) + fsq * np.sin(4.0 * delta) * (a_over_r - 0.25))
        # Main calculations
        lat = delta + np.arcsin(ang, out=np.full(ang.shape, np.nan), where=np.abs(ang) <= 1.0)
        lon = np.arctan2(y, x)
        alt = r - a * (1 - flattening * sin_sq_delta - fsq / 2.0 * sin_sq_2delta * (a_over_r - 0.25))
    else:
        raise ValueError(f'Unknown algorithm: "{algorithm}"')

    if output == "combined":
        return np.array([lat, lon, alt]) if lat.size == 1 else np.vstack([lat, lon, alt])
    if output == "split":
        return (lat, lon, alt)
    raise ValueError(f'Unexpected value for output: "{output}"')


# %% Unit test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_aerospace_earth", exit=False)
    doctest.testmod(verbose=False)
