r"""
Functions that support orbital elements in different frames.

Notes
-----
#.  Written by David C. Stauffer in August 2021.
"""

#%% Imports
from __future__ import annotations
import doctest
from typing import overload, TYPE_CHECKING, Union
import unittest

from dstauffman import DEG2RAD, HAVE_NUMPY, magnitude, NP_DATETIME_UNITS, NP_ONE_DAY, ONE_MINUTE, \
    ONE_HOUR, RAD2DEG

from dstauffman.aerospace.orbit_const import EARTH, JULIAN, TAU

if HAVE_NUMPY:
    import numpy as np

if TYPE_CHECKING:
    _N = Union[float, np.ndarray]

#%% Functions - d_2_r
@overload
def d_2_r(deg: float) -> float: ...
@overload
def d_2_r(deg: np.ndarray) -> np.ndarray: ...

def d_2_r(deg: _N) -> _N:
    return DEG2RAD * deg

#%% Functions r_2_d
@overload
def r_2_d(rad: float) -> float: ...
@overload
def r_2_d(rad: np.ndarray) -> np.ndarray: ...

def r_2_d(rad: _N) -> _N:
    return RAD2DEG * rad

#%% Functions - norm
def norm(x):
    return np.asanyarray(magnitude(x))

#%% Functions - dot
def dot(x, y):
    return np.sum(x * y, axis=0)

#%% Functions - cross
def cross(x, y):
    return np.cross(x.T, y.T).T

#%% Functions - jd_to_numpy
@overload
def jd_to_numpy(jd: float) -> np.datetime64: ...
@overload
def jd_to_numpy(jd: np.ndarray) -> np.ndarray: ...

def jd_to_numpy(jd: _N) -> Union[np.datetime64, np.ndarray]:
    r"""
    Converts a julian date to a numpy datetime64.

    Examples
    --------
    >>> from dstauffman.aerospace import jd_to_numpy
    >>> jd = 2451546.5
    >>> date = jd_to_numpy(jd)
    >>> print(date)
    2000-01-02T12:00:00.000000000

    """
    delta_days = jd - JULIAN['jd_2000_01_01']
    out = np.datetime64('2000-01-01T00:00:00', NP_DATETIME_UNITS) + \
        NP_ONE_DAY * delta_days
    return out

#%% Functions - numpy_to_jd
@overload
def numpy_to_jd(date: np.datetime64) -> np.float64: ...
@overload
def numpy_to_jd(date: np.ndarray) -> np.ndarray: ...

def numpy_to_jd(date: Union[np.datetime64, np.ndarray]) -> Union[np.float64, np.ndarray]:
    r"""
    Converts a numpy datetime64 into a julian date.

    Examples
    --------
    >>> from dstauffman.aerospace import numpy_to_jd
    >>> import numpy as np
    >>> date = np.datetime64('2000-01-02T12:00:00')
    >>> jd = numpy_to_jd(date)
    >>> print(jd)
    2451546.5

    """
    delta_days = (date - np.datetime64('2000-01-01T00:00:00', NP_DATETIME_UNITS)) / NP_ONE_DAY
    out = delta_days + JULIAN['jd_2000_01_01']
    return out

#%% Functions - d_2_dms
def d_2_dms(x, /):
    r"""Converts an angle from degrees to degrees, minutes and seconds."""
    if np.ndim(x) > 1:
        raise ValueError('dms_2_d expects a vector as input.')
    # calculate size of array
    n = np.size(x)
    # initialize output
    out = np.empty(3) if n == 1 else np.empty((3, n))
    # find whole degrees
    out[0, ...] = np.floor(x)
    # find whole minutes
    out[1, ...] = np.floor(np.mod(x, 1)*ONE_MINUTE)
    # find fractional seconds
    out[2, ...] = (np.mod(x, 1) - out[1, ...]/ONE_MINUTE) * ONE_HOUR
    return out

#%% Functions - dms_2_d
def dms_2_d(x, /):
    r"""Converts an angle from degrees, minutes and seconds to degrees."""
    if x.shape[0] != 3:
        raise ValueError('d_2_dms expects a 3xN array as input.')
    # find fractional degrees by adding parts together
    out = x[0, ...] + x[1, ...]/ONE_MINUTE + x[2, ...]/ONE_HOUR
    return out

#%% Functions - hms_2_r
def hms_2_r(x, /):
    r"""Converts a time from hours, minutes and seconds to radians."""
    if x.shape[0] != 3:
        raise ValueError('hms_2_r expects a 3xN array as input.')
    # find fractional degrees by adding parts together
    hours = x[0, ...] + x[1, ...]/ONE_MINUTE + x[2, ...]/ONE_HOUR
    out = hours/24 * TAU
    return out

#%% Functions - r_2_hms
def r_2_hms(x, /):
    r"""Converts an angle from radians to hours, minutes and seconds."""
    if np.ndim(x) > 1:
        raise ValueError('r_2_hms expects a vector as input.')
    # calculate size of array
    n = np.size(x)
    # initialize output
    out = np.empty(3) if n == 1 else np.empty((3, n))
    # convert from angle to fractional hours
    hours = 24 * (x/TAU)
    # find whole hours
    out[0, ...] = np.floor(hours)
    # find whole minutes
    out[1, ...] = np.floor(np.mod(hours, 1)*ONE_MINUTE)
    # find fractional seconds
    out[2, ...] = (np.mod(hours, 1) - out[1, ...]/ONE_MINUTE) * ONE_HOUR
    return out

#%% Functions - aer_2_rdr
def aer_2_rdr(aer, geo_loc, time):
    r"""Converts Az/El/range to RA/Dec/range."""
    sez = aer_2_sez(aer)
    ijk = sez_2_ijk(sez, geo_loc, time)
    rdr = ijk_2_rdr(ijk)
    return rdr

#%% Functions - aer_2_sez
def aer_2_sez(aer):
    r"""Converts Az/El/range to SEZ cartesian."""
    # pull out az, el and range
    az = aer[0, ...]
    el = aer[1, ...]
    rho = aer[2, ...]
    # transform vector
    sez = np.vstack((-rho*np.cos(el)*np.cos(az), rho*np.cos(el)*np.sin(az), rho*np.sin(el)))
    return sez

#%% Functions - geo_loc_2_ijk
def geo_loc_2_ijk(geo_loc, time_jd):
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
    >>> with np.printoptions(precision=8):
    ...     print(R) # doctest: +NORMALIZE_WHITESPACE
    [[  641795.75242525]
     [-5043096.63066764]
     [ 3838833.11215428]]

    """
    # pull out data from geo_loc
    L = geo_loc[0, ...]
    lambda_ = geo_loc[1, ...]
    H = geo_loc[2, ...]
    # find x & z
    x = (EARTH['a'] / np.sqrt(1. - EARTH['e']**2 * np.sin(L)**2) + H) * np.cos(L)
    z = (EARTH['a'] * (1. - EARTH['e']**2)/np.sqrt(1. - EARTH['e']**2 * np.sin(L)**2) + H) * np.sin(L)
    # find theta
    theta = long_2_sidereal(lambda_, time_jd)
    # create vector in IJK frame
    R = np.vstack((x*np.cos(theta), x*np.sin(theta), z))
    return R

#%% Functions - ijk_2_rdr
def ijk_2_rdr(ijk):
    r"""Converts IJK cartesian to Ra/Dec/range."""
    # pull out i,j,k row vectors
    i = ijk[0, ...]
    j = ijk[1, ...]
    k = ijk[2, ...]
    # calculate magnitude of vectors
    rang = np.sqrt(i**2 + j**2 + k**2)  # TODO: use magnitude
    # find azimuth
    ra = np.atan2(j, i)
    # change from [-pi pi] to [0 2*pi]
    ra = np.mod(ra, TAU)
    # find elevation
    de = np.atan2(k, np.sqrt(i**2 + j**2))
    # output array of values
    rdr = np.vstack((ra, de, rang))
    return rdr

#%% Functions - ijk_2_sez
def ijk_2_sez(ijk, geo_loc, time_jd):
    r"""Converts IJK coordinates to SEZ coordinates."""
    #% Subfunctions
    def _find_D(L ,theta):
        r"""Calculate the IJK to SEZ transformation matrix from L and theta."""
        return np.array([[ np.sin(L)*np.cos(theta), np.sin(L)*np.sin(theta), -np.cos(L)], \
                         [-np.sin(theta)          , np.cos(theta)          ,  0.       ], \
                         [+np.cos(L)*np.cos(theta), np.cos(L)*np.sin(theta),  np.sin(L)]])

    # find vector to observer in IJK frame
    R = geo_loc_2_ijk(geo_loc, time_jd)
    # calculate SEZ vector in IJK frame by subtracting position on Earth in IJK
    sez_in_ijk = ijk - R
    # find the size of the input array
    (m, n) = ijk.shape
    if m != 3:
        raise ValueError('ijk_2_sez expects an array of 3 dimensional vectors for sez')
    (m, length) = geo_loc.shape
    if m != 3:
        raise ValueError('ijk_2_sez expects an array of 3 dimensional vectors for geo_loc')
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
        sez = D*sez_in_ijk
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
        raise ValueError('ijk and geo_loc must be arrays of the same length')
    return sez

#%% Functions - long_2_sidereal
def long_2_sidereal(lambda_, time_jd):
    r"""Converts a geographic longitude to sidereal longitude."""
    # epoch
    to = JULIAN['tg0_2000_time']
    # theta at epoch
    theta_go = JULIAN['tg0_2000']
    # find theta
    theta = np.mod(theta_go + EARTH['omega']*JULIAN['day']*(time_jd - to) + lambda_, TAU)
    return theta

#%% Functions - rdr_2_aer
def rdr_2_aer(rdr, geo_loc, time):
    r"""Converts RA/Dec/range to Az/El/range."""
    # transform rdr to ijk frame
    ijk = rdr_2_ijk(rdr)
    # transform ijk frame to sez frame
    sez = ijk_2_sez(ijk, geo_loc, time)
    # transford sez frame to aer frame
    aer = sez_2_aer(sez)
    return aer

#%% Functions - rdr_2_ijk
def rdr_2_ijk(rdr):
    r"""Converts Ra/Dec/range to IJK cartesian."""
    # pull out components
    ra = rdr[0, ...]
    de = rdr[1, ...]
    rang = rdr[2, ...]
    # calculate vector in IJK frame
    ijk = np.vstack((rang*np.cos(de)*np.cos(ra), rang*np.cos(de)*np.sin(ra), rang*np.sin(de)))
    return ijk

#%% Functions - sez_2_aer
def sez_2_aer(sez):
    r"""Converts SEZ cartesian to Az/El/range."""
    # pull out x,y,z row vectors
    x = sez[0, ...]
    y = sez[1, ...]
    z = sez[2, ...]
    # calculate magnitude of vectors
    rho = np.sqrt(x**2 + y**2 + z**2)  # TODO: use magnitude
    # find azimuth
    az = np.atan2(y, -x)
    # change range from (-pi,pi) to (0,2*pi)
    az = np.mod(az, TAU)
    # find elevation (note Elevation is in range (-pi/2:pi/2)
    el = np.atan(z/np.sqrt(x**2 + y**2))
    # output array of values
    aer = np.vstack((az, el, rho))
    return aer

#%% Functions - sez_2_ijk
def sez_2_ijk(sez, geo_loc, time_jd):
    r"""Converts SEZ coordinates to IJK coordinates."""
    #% Subfunctions
    def _find_D(L ,theta):
        r"""Calculate the SEZ to IJK transformation matrix from L and theta."""
        return np.array([[ np.sin(L)*np.cos(theta), -np.sin(theta), np.cos(L)*np.cos(theta)], \
                         [+np.sin(L)*np.sin(theta),  np.cos(theta), np.cos(L)*np.sin(theta)], \
                         [-np.cos(L)              ,  0.           , np.sin(L)]])

    # find the size of the input array
    (m, n) = sez.shape
    if m != 3:
        raise ValueError('ijk_2_sez expects an array of 3 dimensional vectors for sez')
    (m, length) = geo_loc.shape
    if m != 3:
        raise ValueError('ijk_2_sez expects an array of 3 dimensional vectors for geo_loc')
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
        raise ValueError('sez and geo_loc must be arrays of the same length')
    # find vector to observer's position
    R = geo_loc_2_ijk(geo_loc, time_jd)
    # add position vector to SEZ position
    ijk = sez_in_ijk + R
    return ijk

#%% Functions - rv_aer_2_ijk
def rv_aer_2_ijk(r_aer, v_aer, geo_loc, time_jd):
    # transform to SEZ frame
    (r_sez, v_sez) = rv_aer_2_sez(r_aer, v_aer)
    # transform to IJK frame
    (r_ijk, v_ijk) = rv_sez_2_ijk(r_sez, v_sez, geo_loc, time_jd)
    return (r_ijk, v_ijk)

#%% Functions - rv_aer_2_sez
def rv_aer_2_sez(r_aer, v_aer):
    # tranform position vector
    r_sez = aer_2_sez(r_aer)

    # pull out az, el and range and their derivatives
    az  = r_aer[0, ...]
    el  = r_aer[1, ...]
    rho = r_aer[2, ...]
    az_dot  = v_aer[0, ...]
    el_dot  = v_aer[1, ...]
    rho_dot = v_aer[2, ...]

    # calculate velocity transform
    v_sez = np.array([-rho_dot*np.cos(el)*np.cos(az) + rho*np.sin(el)*el_dot*np.cos(az) + rho*np.cos(el)*np.sin(az)*az_dot, \
        rho_dot*np.cos(el)*np.sin(az) - rho*np.sin(el)*el_dot*np.sin(az) + rho*np.cos(el)*np.cos(az)*az_dot, \
        rho_dot*np.sin(el) + rho*np.cos(el)*el_dot])
    return (r_sez, v_sez)

#%% Functions - rv_ijk_2_aer
def rv_ijk_2_aer(r_ijk, v_ijk, geo_loc, time_jd):
    # transform from IJK frame to SEZ frame
    (r_sez, v_sez) = rv_ijk_2_sez(r_ijk, v_ijk, geo_loc, time_jd)
    # transform from SEZ frame to AER frame
    (r_aer, v_aer) = rv_sez_2_aer(r_sez, v_sez)
    return (r_aer, v_aer)

#%% Functions - rv_ijk_2_sez
def rv_ijk_2_sez(r_ijk, v_ijk, geo_loc, time_jd):
    # transform position from SEZ to IJK frame
    r_sez = sez_2_ijk(r_ijk, geo_loc, time_jd)
    # TODO: is this really true?
    v_sez = v_ijk
    return (r_sez, v_sez)

#%% Functions - rv_sez_2_aer
def rv_sez_2_aer(r_sez, v_sez):
    # transform position from SEZ to AER frame
    r_aer = sez_2_aer(r_sez)
    # TODO: calculate v_aer - don't think this is correct.
    v_aer = sez_2_aer(v_sez)
    return (r_aer, v_aer)

#%% Functions - rv_sez_2_ijk
def rv_sez_2_ijk(r_sez, v_sez, geo_loc, time_jd):
    # transform position from SEZ to IJK frame
    r_ijk = sez_2_ijk(r_sez, geo_loc, time_jd)
    # express SEZ velocity in IJK frame
    v_sez_in_ijk = sez_2_ijk(v_sez, np.array([0., 0., -EARTH['a']]), time_jd)
    # number of vectors
    n = r_sez.shape[1]
    # calculate omega vector (Earth spins about k axis)
    omega_vector = np.repeat(np.array([0., 0., EARTH['omega']]), (1, n))
    # calculate velocity in IJK frame
    v_ijk = v_sez_in_ijk + cross(omega_vector, r_ijk)
    return (r_ijk, v_ijk)

#%% Unit Test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_aerospace_orbit_support', exit=False)
    doctest.testmod(verbose=False)