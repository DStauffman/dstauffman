r"""
Functions that support orbital elements in different frames.

Notes
-----
#.  Written by David C. Stauffer in August 2021.
"""

# %% Imports
from __future__ import annotations

import doctest
from typing import overload, Tuple, TYPE_CHECKING, Union
import unittest

from dstauffman import ARCSEC2RAD, DEG2RAD, HAVE_NUMPY, magnitude, NP_DATETIME_UNITS, NP_ONE_DAY, ONE_HOUR, ONE_MINUTE, RAD2DEG
from dstauffman.aerospace.orbit_const import EARTH, JULIAN, PI, TAU

if HAVE_NUMPY:
    import numpy as np

if TYPE_CHECKING:
    _D = np.typing.NDArray[np.datetime64]
    _N = np.typing.NDArray[np.float64]
    _FN = Union[float, _N]


# %% Functions - d_2_r
@overload
def d_2_r(deg: float) -> float:
    ...


@overload
def d_2_r(deg: _N) -> _N:
    ...


def d_2_r(deg: _FN) -> _FN:
    r"""Converts degrees to radians."""
    return DEG2RAD * deg


# %% Functions r_2_d
@overload
def r_2_d(rad: float) -> float:
    ...


@overload
def r_2_d(rad: _N) -> _N:
    ...


def r_2_d(rad: _FN) -> _FN:
    r"""Converts radians to degrees."""
    return RAD2DEG * rad


# %% Functions - norm
def norm(x: _N) -> _N:
    r"""Alias to the magnitude function, calculating the vector magnitude."""
    return np.asanyarray(magnitude(x))


# %% Functions - dot
def dot(x: _N, y: _N) -> _N:
    r"""Dot product between two vectors."""
    return np.sum(x * y, axis=0)  # type: ignore[no-any-return]


# %% Functions - cross
def cross(x: _N, y: _N) -> _N:
    r"""Cross product between two vectors."""
    return np.cross(x.T, y.T).T


# %% Functions - jd_to_numpy
@overload
def jd_to_numpy(time_jd: float) -> np.datetime64:
    ...


@overload
def jd_to_numpy(time_jd: _N) -> _D:
    ...


def jd_to_numpy(time_jd: _FN) -> Union[np.datetime64, _D]:
    r"""
    Converts a julian date to a numpy datetime64.

    Examples
    --------
    >>> from dstauffman.aerospace import jd_to_numpy
    >>> time_jd = 2451546.5
    >>> date = jd_to_numpy(time_jd)
    >>> print(date)
    2000-01-03T00:00:00.000000000

    """
    delta_days = time_jd - JULIAN["jd_2000_01_01"]
    out = np.datetime64("2000-01-01T12:00:00", NP_DATETIME_UNITS) + NP_ONE_DAY * delta_days
    return out


# %% Functions - numpy_to_jd
@overload
def numpy_to_jd(date: np.datetime64) -> float:
    ...


@overload
def numpy_to_jd(date: _D) -> _N:
    ...


def numpy_to_jd(date: Union[np.datetime64, _D]) -> Union[float, _N]:
    r"""
    Converts a numpy datetime64 into a julian date.

    Examples
    --------
    >>> from dstauffman.aerospace import numpy_to_jd
    >>> import numpy as np
    >>> date = np.datetime64("2000-01-02T12:00:00")
    >>> time_jd = numpy_to_jd(date)
    >>> print(time_jd)
    2451546.0

    """
    delta_days = (date - np.datetime64("2000-01-01T12:00:00", NP_DATETIME_UNITS)) / NP_ONE_DAY
    out = delta_days + JULIAN["jd_2000_01_01"]
    return out  # type: ignore[return-value]


# %% Functions - jd_2_century
@overload
def jd_2_century(time_jd: float) -> Tuple[float, float]:
    ...


@overload
def jd_2_century(time_jd: _N) -> Tuple[_N, _N]:
    ...


def jd_2_century(time_jd: _FN) -> Tuple[_FN, _FN]:
    r"""
    Converts a julian day to the fractional julian centuries since J2000.

    Examples
    --------
    >>> from dstauffman.aerospace import jd_2_century, JULIAN
    >>> d0 = JULIAN["jd_2000_01_01"]
    >>> time_jd = 10000 + d0
    >>> (Du, T) = jd_2_century(time_jd)
    >>> print(Du)
    10000.0

    >>> print(T)
    0.2737850787132101

    """
    # delta time in days since J2000
    Du = time_jd - JULIAN["jd_2000_01_01"]
    # Time in Julian centuries
    T = Du * JULIAN["day"] / JULIAN["century"]
    return (Du, T)


# %% Functions - d_2_dms
@overload
def d_2_dms(x: float, /) -> _N:
    ...


@overload
def d_2_dms(x: _N, /) -> _N:
    ...


def d_2_dms(x: _FN, /) -> _N:
    r"""Converts an angle from degrees to degrees, minutes and seconds."""
    if np.ndim(x) > 1:
        raise ValueError("dms_2_d expects a vector as input.")
    # calculate size of array
    n = np.size(x)
    # initialize output
    out = np.empty(3) if n == 1 else np.empty((3, n))
    # find whole degrees
    out[0, ...] = np.floor(x)
    # find whole minutes
    out[1, ...] = np.floor(np.mod(x, 1) * ONE_MINUTE)
    # find fractional seconds
    out[2, ...] = (np.mod(x, 1) - out[1, ...] / ONE_MINUTE) * ONE_HOUR
    return out


# %% Functions - dms_2_d
def dms_2_d(x: _N, /) -> _N:
    r"""Converts an angle from degrees, minutes and seconds to degrees."""
    if x.shape[0] != 3:
        raise ValueError("d_2_dms expects a 3xN array as input.")
    # find fractional degrees by adding parts together
    out = x[0, ...] + x[1, ...] / ONE_MINUTE + x[2, ...] / ONE_HOUR
    return out  # type: ignore[no-any-return]


# %% Functions - hms_2_r
def hms_2_r(x: _N, /) -> _N:
    r"""Converts a time from hours, minutes and seconds to radians."""
    if x.shape[0] != 3:
        raise ValueError("hms_2_r expects a 3xN array as input.")
    # find fractional degrees by adding parts together
    hours = x[0, ...] + x[1, ...] / ONE_MINUTE + x[2, ...] / ONE_HOUR
    out = hours / 24 * TAU
    return out  # type: ignore[no-any-return]


# %% Functions - r_2_hms
@overload
def r_2_hms(x: float, /) -> _N:
    ...


@overload
def r_2_hms(x: _N, /) -> _N:
    ...


def r_2_hms(x: _FN, /) -> _N:
    r"""Converts an angle from radians to hours, minutes and seconds."""
    if np.ndim(x) > 1:
        raise ValueError("r_2_hms expects a vector as input.")
    # calculate size of array
    n = np.size(x)
    # initialize output
    out = np.empty(3) if n == 1 else np.empty((3, n))
    # convert from angle to fractional hours
    hours = 24 * (x / TAU)
    # find whole hours
    out[0, ...] = np.floor(hours)
    # find whole minutes
    out[1, ...] = np.floor(np.mod(hours, 1) * ONE_MINUTE)
    # find fractional seconds
    out[2, ...] = (np.mod(hours, 1) - out[1, ...] / ONE_MINUTE) * ONE_HOUR
    return out


# %% Functions - aer_2_rdr
def aer_2_rdr(aer: _N, geo_loc: _N, time_jd: _N) -> _N:
    r"""Converts Az/El/range to RA/Dec/range."""
    sez = aer_2_sez(aer)
    ijk = sez_2_ijk(sez, geo_loc, time_jd)
    rdr = ijk_2_rdr(ijk)
    return rdr


# %% Functions - aer_2_sez
def aer_2_sez(aer: _N) -> _N:
    r"""Converts Az/El/range to SEZ cartesian."""
    # pull out az, el and range
    az = aer[0, ...]
    el = aer[1, ...]
    rho = aer[2, ...]
    # transform vector
    sez = np.vstack((-rho * np.cos(el) * np.cos(az), rho * np.cos(el) * np.sin(az), rho * np.sin(el)))
    return sez


# %% Functions - geo_loc_2_ijk
def geo_loc_2_ijk(geo_loc: _N, time_jd: _N) -> _N:
    r"""
    Converts a geographic location to IJK coordinates.

    Notes
    -----
    Converts the latitude, longitude and altitude to the IJK frame. This uses the WGS84 ellipsoid.
    #.  Written by David C. Stauffer in April 2007 for AA279.

    Examples
    --------
    >>> from dstauffman.aerospace import geo_loc_2_ijk
    >>> import numpy as np
    >>> R = geo_loc_2_ijk(np.array([0.65, -2.13, 4.]), 2454587)
    >>> with np.printoptions(precision=5):
    ...     print(R) # doctest: +NORMALIZE_WHITESPACE
    [[  641795.75243]
     [-5043096.63067]
     [ 3838833.11215]]

    """
    # pull out data from geo_loc
    L = geo_loc[0, ...]
    lambda_ = geo_loc[1, ...]
    H = geo_loc[2, ...]
    # find x & z
    x = (EARTH["a"] / np.sqrt(1.0 - EARTH["e"] ** 2 * np.sin(L) ** 2) + H) * np.cos(L)
    z = (EARTH["a"] * (1.0 - EARTH["e"] ** 2) / np.sqrt(1.0 - EARTH["e"] ** 2 * np.sin(L) ** 2) + H) * np.sin(L)
    # find theta
    theta = long_2_sidereal(lambda_, time_jd)
    # create vector in IJK frame
    R = np.vstack((x * np.cos(theta), x * np.sin(theta), z))
    return R


# %% Functions - ijk_2_rdr
def ijk_2_rdr(ijk: _N) -> _N:
    r"""Converts IJK cartesian to Ra/Dec/range."""
    # pull out i,j,k row vectors
    i = ijk[0, ...]
    j = ijk[1, ...]
    k = ijk[2, ...]
    # calculate magnitude of vectors
    rang = np.sqrt(i**2 + j**2 + k**2)  # TODO: use magnitude
    # find azimuth
    ra = np.arctan2(j, i)
    # change from [-pi pi] to [0 2*pi]
    ra = np.mod(ra, TAU)
    # find elevation
    de = np.arctan2(k, np.sqrt(i**2 + j**2))
    # output array of values
    rdr = np.vstack((ra, de, rang))
    return rdr


# %% Functions - ijk_2_sez
def ijk_2_sez(ijk: _N, geo_loc: _N, time_jd: _N) -> _N:
    r"""Converts IJK coordinates to SEZ coordinates."""

    def _find_D(L: _FN, theta: _FN) -> _N:
        r"""Calculate the IJK to SEZ transformation matrix from L and theta."""
        # fmt: off
        return np.array([
            [ np.sin(L) * np.cos(theta), np.sin(L) * np.sin(theta), -np.cos(L)],
            [-np.sin(theta)            , np.cos(theta)            , 0.0       ],
            [ np.cos(L) * np.cos(theta), np.cos(L) * np.sin(theta),  np.sin(L)],
        ])
        # fmt: on

    # find vector to observer in IJK frame
    R = geo_loc_2_ijk(geo_loc, time_jd)
    # calculate SEZ vector in IJK frame by subtracting position on Earth in IJK
    sez_in_ijk = ijk - R
    # find the size of the input array
    (m, n) = ijk.shape
    if m != 3:
        raise ValueError("ijk_2_sez expects an array of 3 dimensional vectors for sez")
    (m, length) = geo_loc.shape
    if m != 3:
        raise ValueError("ijk_2_sez expects an array of 3 dimensional vectors for geo_loc")
    # pull out data from geo_loc
    L = geo_loc[0, ...]
    lambda_ = geo_loc[1, ...]
    # find sidereal time
    theta = long_2_sidereal(lambda_, time_jd)
    # initialize output
    sez = np.zeros((m, n))
    # create transformation matrix for four different size combinations, and
    # then transform the SEZ vector from the IJK frame to the SEZ frame
    if n == 1 and length == 1:
        D = _find_D(L, theta)
        sez = D * sez_in_ijk
    elif n == 1 and length != 1:
        for i in range(length):
            D = _find_D(L[i], theta[i])
            sez[:, i] = D @ sez_in_ijk
    elif n != 1 and length == 1:
        D = _find_D(L, theta)
        for i in range(n):
            sez[:, i] = D @ sez_in_ijk[:, i]
    elif n == length:
        for i in range(n):
            D = _find_D(L[i], theta[i])
            sez[:, i] = D @ sez_in_ijk[:, i]
    else:
        raise ValueError("ijk and geo_loc must be arrays of the same length")
    return sez


# %% Functions - long_2_sidereal
def long_2_sidereal(lon: _N, time_jd: _N) -> _N:
    r"""
    Converts a geographic longitude to sidereal longitude.

    Parameters
    ----------
    lon : float or (N, ) ndarray
        Geographic longitude [rad]
    time_jd : float or (N, ) ndarray
        Julian date

    Returns
    -------
    theta : float or (N, ) ndarray
        Sidereal longitude [rad]

    Notes
    -----
    #.  Written by David C. Stauffer for AA279 on 12 May 2007.
    #.  Translated into Python by David C. Stauffer in October 2021.

    Examples
    --------
    >>> from dstauffman.aerospace import long_2_sidereal
    >>> from math import pi
    >>> lon = -2.13
    >>> time_jd = 2454587
    >>> theta = long_2_sidereal(lon, time_jd)
    >>> print(f"{theta:.8f}")
    4.83897078

    """
    # epoch
    to = JULIAN["tg0_2000_time"]
    # theta at epoch
    theta_go = JULIAN["tg0_2000"]
    # earth rate per day
    earth_rate = EARTH["omega"] * JULIAN["day"]
    # find theta
    theta = np.mod(theta_go + earth_rate * (time_jd - to) + lon, TAU)
    return theta


# %% Functions - rdr_2_aer
def rdr_2_aer(rdr: _N, geo_loc: _N, time_jd: _N) -> _N:
    r"""Converts RA/Dec/range to Az/El/range."""
    # transform rdr to ijk frame
    ijk = rdr_2_ijk(rdr)
    # transform ijk frame to sez frame
    sez = ijk_2_sez(ijk, geo_loc, time_jd)
    # transford sez frame to aer frame
    aer = sez_2_aer(sez)
    return aer


# %% Functions - rdr_2_ijk
def rdr_2_ijk(rdr: _N) -> _N:
    r"""Converts Ra/Dec/range to IJK cartesian."""
    # pull out components
    ra = rdr[0, ...]
    de = rdr[1, ...]
    rang = rdr[2, ...]
    # calculate vector in IJK frame
    ijk = np.vstack((rang * np.cos(de) * np.cos(ra), rang * np.cos(de) * np.sin(ra), rang * np.sin(de)))
    return ijk


# %% Functions - sez_2_aer
def sez_2_aer(sez: _N) -> _N:
    r"""Converts SEZ cartesian to Az/El/range."""
    # pull out x,y,z row vectors
    x = sez[0, ...]
    y = sez[1, ...]
    z = sez[2, ...]
    # calculate magnitude of vectors
    rho = np.sqrt(x**2 + y**2 + z**2)  # TODO: use magnitude
    # find azimuth
    az = np.arctan2(y, -x)
    # change range from (-pi,pi) to (0,2*pi)
    az = np.mod(az, TAU)
    # find elevation (note Elevation is in range (-pi/2:pi/2)
    el = np.arctan(z / np.sqrt(x**2 + y**2))
    # output array of values
    aer = np.vstack((az, el, rho))
    return aer


# %% Functions - sez_2_ijk
def sez_2_ijk(sez: _N, geo_loc: _N, time_jd: _N) -> _N:
    r"""Converts SEZ coordinates to IJK coordinates."""

    def _find_D(L: _FN, theta: _FN) -> _N:
        r"""Calculate the SEZ to IJK transformation matrix from L and theta."""
        # fmt: off
        return np.array([
            [ np.sin(L) * np.cos(theta), -np.sin(theta), np.cos(L) * np.cos(theta)],
            [ np.sin(L) * np.sin(theta),  np.cos(theta), np.cos(L) * np.sin(theta)],
            [-np.cos(L)                ,  0.0          , np.sin(L)                ],
        ])
        # fmt: on

    # find the size of the input array
    (m, n) = sez.shape
    if m != 3:
        raise ValueError("ijk_2_sez expects an array of 3 dimensional vectors for sez")
    (m, length) = geo_loc.shape
    if m != 3:
        raise ValueError("ijk_2_sez expects an array of 3 dimensional vectors for geo_loc")
    # pull out data from geo_loc
    L = geo_loc[0, ...]
    lambda_ = geo_loc[1, ...]
    # find sidereal time
    theta = long_2_sidereal(lambda_, time_jd)
    # initialize output
    sez_in_ijk = np.zeros((m, n))
    # create transformation matrix for four different size combinations, and
    # then transform the SEZ vector from the IJK frame to the SEZ frame
    if n == 1 and length == 1:
        D = _find_D(L, theta)
        sez_in_ijk = D @ sez
    elif n == 1 and length != 1:
        for i in range(length):
            D = _find_D(L[i], theta[i])
            sez_in_ijk[:, i] = D @ sez
    elif n != 1 and length == 1:
        D = _find_D(L, theta)
        for i in range(n):
            sez_in_ijk[:, i] = D @ sez[:, i]
    elif n == length:
        for i in range(n):
            D = _find_D(L[i], theta[i])
            sez_in_ijk[:, i] = D @ sez[:, i]
    else:
        raise ValueError("sez and geo_loc must be arrays of the same length")
    # find vector to observer's position
    R = geo_loc_2_ijk(geo_loc, time_jd)
    # add position vector to SEZ position
    ijk = sez_in_ijk + R
    return ijk


# %% Functions - rv_aer_2_ijk
def rv_aer_2_ijk(r_aer: _N, v_aer: _N, geo_loc: _N, time_jd: _N) -> Tuple[_N, _N]:
    r"""Converts position and velocity from Az/El/range to IJK cartesion."""
    # transform to SEZ frame
    (r_sez, v_sez) = rv_aer_2_sez(r_aer, v_aer)
    # transform to IJK frame
    (r_ijk, v_ijk) = rv_sez_2_ijk(r_sez, v_sez, geo_loc, time_jd)
    return (r_ijk, v_ijk)


# %% Functions - rv_aer_2_sez
def rv_aer_2_sez(r_aer: _N, v_aer: _N) -> Tuple[_N, _N]:
    r"""Converts position and velocity from Az/El/range to SEZ cartesion."""
    # tranform position vector
    r_sez = aer_2_sez(r_aer)

    # pull out az, el and range and their derivatives
    # fmt: off
    az  = r_aer[0, ...]
    el  = r_aer[1, ...]
    rho = r_aer[2, ...]
    az_dot  = v_aer[0, ...]
    el_dot  = v_aer[1, ...]
    rho_dot = v_aer[2, ...]

    # calculate velocity transform
    v_sez = np.array([
        -rho_dot * np.cos(el) * np.cos(az) + rho * np.sin(el) * el_dot * np.cos(az) + rho * np.cos(el) * np.sin(az) * az_dot,
        +rho_dot * np.cos(el) * np.sin(az) - rho * np.sin(el) * el_dot * np.sin(az) + rho * np.cos(el) * np.cos(az) * az_dot,
        +rho_dot * np.sin(el) + rho * np.cos(el) * el_dot,
    ])
    # fmt: on
    return (r_sez, v_sez)


# %% Functions - rv_ijk_2_aer
def rv_ijk_2_aer(r_ijk: _N, v_ijk: _N, geo_loc: _N, time_jd: _N) -> Tuple[_N, _N]:
    r"""Converts position and velocity IJK cartesion to Az/El/range."""
    # transform from IJK frame to SEZ frame
    (r_sez, v_sez) = rv_ijk_2_sez(r_ijk, v_ijk, geo_loc, time_jd)
    # transform from SEZ frame to AER frame
    (r_aer, v_aer) = rv_sez_2_aer(r_sez, v_sez)
    return (r_aer, v_aer)


# %% Functions - rv_ijk_2_sez
def rv_ijk_2_sez(r_ijk: _N, v_ijk: _N, geo_loc: _N, time_jd: _N) -> Tuple[_N, _N]:
    r"""Converts position and velocity from IJK to SEZ cartesian."""
    # transform position from SEZ to IJK frame
    r_sez = sez_2_ijk(r_ijk, geo_loc, time_jd)
    # TODO: is this really true?
    v_sez = v_ijk
    return (r_sez, v_sez)


# %% Functions - rv_sez_2_aer
def rv_sez_2_aer(r_sez: _N, v_sez: _N) -> Tuple[_N, _N]:
    r"""Converts position and velocity SEZ cartesion to Az/El/range."""
    # transform position from SEZ to AER frame
    r_aer = sez_2_aer(r_sez)
    # TODO: calculate v_aer - don't think this is correct.
    v_aer = sez_2_aer(v_sez)
    return (r_aer, v_aer)


# %% Functions - rv_sez_2_ijk
def rv_sez_2_ijk(r_sez: _N, v_sez: _N, geo_loc: _N, time_jd: _N) -> Tuple[_N, _N]:
    r"""Converts position and velocity SEZ to IJK cartesion."""
    # transform position from SEZ to IJK frame
    r_ijk = sez_2_ijk(r_sez, geo_loc, time_jd)
    # express SEZ velocity in IJK frame
    v_sez_in_ijk = sez_2_ijk(v_sez, np.array([0.0, 0.0, -EARTH["a"]]), time_jd)
    # number of vectors
    n = r_sez.shape[1]
    # calculate omega vector (Earth spins about k axis)
    omega_vector = np.repeat(np.array([0.0, 0.0, EARTH["omega"]]), (1, n))
    # calculate velocity in IJK frame
    v_ijk = v_sez_in_ijk + cross(omega_vector, r_ijk)
    return (r_ijk, v_ijk)


# %% Functions - get_sun_radec_approx
def get_sun_radec_approx(time_jd: _N) -> Tuple[_N, _N]:
    r"""
    Approximates the right ascension and declination angles to the Sun for the given julian time.

    Guaranteed to be accurate to one arcminute (~0.000291 radians, ~0.0166667 degrees) of error between 1950-2050.

    Parameters
    ----------
    time_jd : np.ndarray
        Julian date

    Returns
    -------
    ra : np.ndarray
        Right ascension [rad]
    dec : np.ndarray
        Declination [rad]

    Notes
    -----
    #.  Algorithm from Astronomical Almanac 2023, page C5.
    #.  Written by David C. Stauffer in Apr 2023.

    Examples
    --------
    >>> from dstauffman.aerospace import get_sun_radec_approx, numpy_to_jd
    >>> from dstauffman import convert_datetime_to_np
    >>> import datetime
    >>> date = datetime.datetime(2010, 6, 20, 15, 30, 45)
    >>> np_date = convert_datetime_to_np(date)
    >>> time_jd = numpy_to_jd(np_date)
    >>> (ra, dec) = get_sun_radec_approx(time_jd)
    >>> print(f"{ra:.3f}")
    1.556

    >>> print(f"{dec:.3f}")
    0.409

    """
    # delta time in days since J2000
    n = time_jd - JULIAN["jd_2000_01_01"]
    # Mean longitude of the Sun, corrected for aberration
    L = np.mod(4.89495 + 0.017202792 * n, TAU)  # np.mod(DEG2RAD * (280.460 + 0.9856474 * n), TAU)
    # Mean anomaly of the Sun
    g = np.mod(6.2400408 + 0.01720197 * n, TAU)  # np.mod(DEG2RAD * (357.528 + 0.9856003 * n), TAU)
    # Ecliptic longitude
    lamd = L + 0.033423 * np.sin(g) + 0.00034907 * np.sin(2 * g)  #  L + DEG2RAD * (1.915 * np.sin(g) + 0.020 * np.sin(2 * g))
    # obliquity of ecliptic
    epsilon = 0.409088 - 7e-9 * n  # DEG2RAD * (23.439 - 0.000_000_4 * n)

    # right ascension
    t = np.tan(0.5 * epsilon) ** 2
    ra = lamd - t * np.sin(2.0 * lamd) + 0.5 * t**2 * np.sin(4.0 * lamd)
    # declination
    dec = np.arcsin(np.sin(epsilon) * np.sin(lamd))

    return (ra, dec)


# %% Functions - get_sun_radec
@overload
def get_sun_radec(time_jd: float) -> Tuple[float, float]:
    ...


@overload
def get_sun_radec(time_jd: _N) -> Tuple[_N, _N]:
    ...


@overload
def get_sun_radec(time_jd: float, return_early: bool) -> Tuple[float, float]:
    ...


@overload
def get_sun_radec(time_jd: _N, return_early: bool) -> Tuple[_N, _N]:
    ...


def get_sun_radec(time_jd: _FN, return_early: bool = False) -> Tuple[_FN, _FN]:
    r"""
    Gets the right ascension and declination angles to the Sun for the given julian time.

    Parameters
    ----------
    time_jd : np.ndarray
        Julian date
    return_early : bool, optional, default is False
        Whether to return early with solar longitude and obliquity instead of ra and dec

    Returns
    -------
    ra : np.ndarray
        Right ascension [rad]
    dec : np.ndarray
        Declination [rad]
    ~ or ~
    sun_true_longitude : np.ndarray
        Solar longitude [rad]
    obliquity_of_ecliptic : np.ndarray
        Obliquity of the ecliptic plane [rad]

    Notes
    -----
    #.  obliquity of ecliptic from Astronomical Almanac 2010.
    #.  Updated by David C. Stauffer in May 2022 to optional return early with sun ecliptic parameters.

    Examples
    --------
    >>> from dstauffman.aerospace import get_sun_radec, numpy_to_jd
    >>> from dstauffman import convert_datetime_to_np
    >>> import datetime
    >>> date = datetime.datetime(2010, 6, 21, 3, 30, 45)
    >>> np_date = convert_datetime_to_np(date)
    >>> time_jd = numpy_to_jd(np_date)
    >>> (ra, dec) = get_sun_radec(time_jd)
    >>> print(f"{ra:.3f}")
    1.565

    >>> print(f"{dec:.3f}")
    0.409

    """
    # time in julian centuries
    (_, T) = jd_2_century(time_jd)
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    T5 = T4 * T
    # (eq 25.2) Geometric mean longitude of the Sun, referred to the mean equinox of the date
    Lo = np.mod(DEG2RAD * (280.46645 + 36000.76983 * T + 0.0003032 * T2), TAU)
    # (eq 25.3) Mean anomaly of the Sun
    M = np.mod(DEG2RAD * (357.52910 + 35999.05030 * T - 0.0001559 * T2 - 0.00000048 * T3), TAU)
    # Center of Sun
    C = DEG2RAD * (
        (1.914600 - 0.004817 * T - 0.000014 * T2) * np.sin(M)
        + (0.019993 - 0.000101 * T) * np.sin(2 * M)
        + 0.000290 * np.sin(3 * M)
    )
    # true longitude
    sun_true_longitude = np.mod(Lo + C, TAU)
    # (eq 22.2) (arc-sec)
    obliquity_of_ecliptic = ARCSEC2RAD * (
        23 * ONE_HOUR
        + 26 * ONE_MINUTE
        + 21.406
        - 46.836769 * T
        - 0.0001831 * T2
        + 0.00200340 * T3
        - 0.576e-6 * T4
        - 4.34e-8 * T5
    )
    if return_early:
        return (sun_true_longitude, obliquity_of_ecliptic)
    # right ascension
    ra = np.mod(np.arctan2(np.cos(obliquity_of_ecliptic) * np.sin(sun_true_longitude), np.cos(sun_true_longitude)), TAU)
    # declination
    dec = np.arcsin(np.sin(obliquity_of_ecliptic) * np.sin(sun_true_longitude))
    return (ra, dec)


# %% Functions - get_sun_distance
@overload
def get_sun_distance(time_jd: float) -> float:
    ...


@overload
def get_sun_distance(time_jd: _N) -> _N:
    ...


def get_sun_distance(time_jd: _FN) -> _FN:
    r"""
    Calculate the distance to the Sun from the Earth.

    Guaranteed to be accurate to 0.0003 AU of error between 1950-2050.

    Parameters
    ----------
    time_jd : np.ndarray
        Julian date

    Returns
    -------
    np.ndarray
        Distance to the sun in AU

    Notes
    -----
    #.  Algorithm from Astronomical Almanac 2023, page C5.
    #.  Written by David C. Stauffer in August 2023.

    Examples
    --------
    >>> from dstauffman.aerospace import get_sun_distance, numpy_to_jd
    >>> from dstauffman import convert_datetime_to_np
    >>> import datetime
    >>> date = datetime.datetime(2010, 6, 21, 3, 30, 45)
    >>> np_date = convert_datetime_to_np(date)
    >>> time_jd = numpy_to_jd(np_date)
    >>> sun_dist_au = get_sun_distance(time_jd)
    >>> print(f"{sun_dist_au:.5f}")
    1.01624

    """
    # delta time in days since J2000
    n = time_jd - JULIAN["jd_2000_01_01"]
    # Mean anomaly of the Sun
    g = np.mod(6.2400408 + 0.01720197 * n, TAU)
    return 1.000_14 - 0.016_71 * np.cos(g) - 0.000_14 * np.cos(2 * g)  # type: ignore[no-any-return]


# %% Functions - beta_from_oe
def beta_from_oe(raan: _N, inclination: _N, time_jd: _N) -> _N:
    r"""
    Calculates the beta angle between the sun and the orbit plane.

    Parameters
    ----------
    raan : np.ndarray
        Right ascension of the ascending node
    inclination : np.ndarray
        Inclination of the orbit plane
    time_jd : np.ndarray
        Julian date

    Returns
    -------
    np.ndarry
        Beta angle

    Notes
    -----
    #.  Written by David C. Stauffer in May 2022.

    Examples
    --------
    >>> from dstauffman.aerospace import beta_from_oe
    >>> raan = 0.5
    >>> inclination = 1.4
    >>> time_jd = 2455368.5
    >>> beta = beta_from_oe(raan, inclination, time_jd)
    >>> print(f"{beta:0.4f}")
    -0.8068

    """
    (Ls, ob) = get_sun_radec(time_jd, return_early=True)
    sin_beta = (
        np.cos(Ls) * np.sin(raan) * np.sin(inclination)
        - np.sin(Ls) * np.cos(ob) * np.cos(raan) * np.sin(inclination)
        + np.sin(Ls) * np.sin(ob) * np.cos(inclination)
    )
    return np.arcsin(sin_beta)  # type: ignore[no-any-return]


# %% Functions - eclipse_fraction
@overload
def eclipse_fraction(altitude: float, beta: float) -> float:
    ...


@overload
def eclipse_fraction(altitude: float, beta: _N) -> _N:
    ...


@overload
def eclipse_fraction(altitude: _N, beta: float) -> _N:
    ...


@overload
def eclipse_fraction(altitude: _N, beta: _N) -> _N:
    ...


def eclipse_fraction(altitude: _FN, beta: _FN) -> _FN:
    r"""
    Gets the faction of the orbit period for which the satellite is in umbra.

    Parameters
    ----------
    altitude : np.ndarray
        Spacecraft altitude [m]
    beta : np.ndarray
        Sun beta angle [rad]

    Returns
    -------
    eclipse : np.ndarray
        Fraction of orbit period spent in Earth eclipse

    Notes
    -----
    #.  Written by David C. Stauffer in May 2022

    Examples
    --------
    >>> from dstauffman.aerospace import eclipse_fraction
    >>> altitude = 16000.0
    >>> beta = 3.14/6
    >>> eclipse = eclipse_fraction(altitude, beta)
    >>> print(f"{eclipse:.4f}")
    0.4740

    """
    # force inputs to be ndarrays so they can be indexed
    beta = np.asanyarray(beta)
    altitude = np.asanyarray(altitude)
    # alias the Earth radius
    Re = EARTH["a"]
    # find the limit of when you have any eclipse
    ix_good = altitude >= 0
    beta_star = np.full(ix_good.shape, np.nan)
    beta_star = np.divide(Re, Re + altitude, out=beta_star, where=ix_good)
    beta_star = np.arcsin(beta_star, out=beta_star, where=ix_good)
    # initialize the output and get an index into when you have some level of eclipse
    eclipse = np.where(ix_good, 0, np.nan)
    ix = np.abs(beta) < beta_star
    # do the actual eclipse fraction calculation
    h = altitude[ix]
    eclipse[ix] = 1 / PI * np.arccos(np.sqrt(h**2 + 2 * Re * h) / ((Re + h) * np.cos(beta[ix])))
    return eclipse


# %% earth_radius_by_latitude
@overload
def earth_radius_by_latitude(latitude: float) -> float:
    ...


@overload
def earth_radius_by_latitude(latitude: _N) -> _N:
    ...


def earth_radius_by_latitude(latitude: _FN) -> _FN:
    r"""
    Calculates the Earth radius at the given latitude for an ellipsoidal Earth.

    Parameters
    ----------
    latitude: (N,) np.ndarray of float
        Latitude [rad]

    Returns
    -------
    (N,) np.ndarray of float
        Earth radius [m]

    Notes
    -----
    #.  Written by David C. Stauffer in May 2023.

    Examples
    --------
    >>> from dstauffman.aerospace import earth_radius_by_latitude
    >>> import numpy as np
    >>> radius = earth_radius_by_latitude(np.array([0.0, np.pi/2]))
    >>> with np.printoptions(precision=8):
    ...     print(radius)
    [6378137.  6356752.3]

    """
    a = EARTH["a"]
    b = EARTH["b"]
    cos_lat = np.cos(latitude)
    sin_lat = np.sin(latitude)
    return np.sqrt(((a**2 * cos_lat) ** 2 + (b**2 * sin_lat) ** 2) / ((a * cos_lat) ** 2 + (b * sin_lat) ** 2))


# %% Unit Test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_aerospace_orbit_support", exit=False)
    doctest.testmod(verbose=False)
