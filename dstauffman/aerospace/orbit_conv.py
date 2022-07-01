r"""
Functions related to converting orbital elements from one form to another.

Notes
-----
#.  Written by David C. Stauffer in July 2021.
"""

#%% Imports
from __future__ import annotations

import doctest
import logging
from typing import Any, TYPE_CHECKING, Union
import unittest

from slog import LogLevel

from dstauffman import HAVE_NUMPY, HAVE_SCIPY, ONE_DAY, ONE_HOUR
from dstauffman.aerospace.orbit_const import EARTH, JULIAN, PI, TAU
from dstauffman.aerospace.orbit_support import get_sun_radec
from dstauffman.aerospace.quat import qrot

if HAVE_NUMPY:
    import numpy as np
    from numpy import sqrt
else:
    from math import sqrt  # type: ignore[misc]

    from dstauffman.nubs import np_any  # pylint: disable=ungrouped-imports
if HAVE_SCIPY:
    from scipy.optimize import root
if TYPE_CHECKING:
    _N = Union[float, np.ndarray]

#%% Globals
logger = logging.getLogger(__name__)

#%% Functions - _any
def _any(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if HAVE_NUMPY:
        return np.any(x)  # type: ignore[return-value]
    return np_any(x)  # type: ignore[no-any-return]


#%% Functions - anomaly_eccentric_2_mean
def anomaly_eccentric_2_mean(E: _N, e: _N) -> _N:
    r"""
    Finds the mean anomaly from the eccentric anomaly.

    Parameters
    ----------
    E : float or (N, ) ndarray
        Eccentric anomaly [rad]
    e : float or (N, ) ndarray
        Eccentricity

    Returns
    -------
    M : float or (N, ) ndarray
        Mean anomaly [rad] (nu will always be between 0 & 2*pi)

    Notes
    -----
    #.  Written by David C. Stauffer for AA279 on 12 May 2007.
    #.  Translated into Python by David C. Stauffer in October 2021.

    Examples
    --------
    >>> from dstauffman.aerospace import anomaly_eccentric_2_mean
    >>> from math import pi
    >>> E = pi / 4
    >>> e = 0.5
    >>> M = anomaly_eccentric_2_mean(E, e)
    >>> print(f"{M:.8f}")
    0.43184477

    """
    # check if orbit is circular or elliptical
    if np.any(e >= 1):
        raise ValueError("The mean anomaly is not defined when e >= 1")
    # check if E is outside the range of 0 to 2*pi
    if np.any((E > TAU) | (E < 0)):
        logger.log(LogLevel.L6, "The eccentric anomaly was outside the range of 0 to 2*pi")
        E = np.mod(E, TAU)
    # calculate the mean anomaly
    M = E - e * np.sin(E)
    return M


#%% Functions - anomaly_eccentric_2_true
def anomaly_eccentric_2_true(E: _N, e: _N) -> _N:
    r"""
    Finds the true anomaly from the eccentric anomaly.

    Parameters
    ----------
    E : float or (N, ) ndarray
        Eccentric anomaly [rad]
    e : float or (N, ) ndarray
        Eccentricity

    Returns
    -------
    nu : float or (N, ) ndarray
        True anomaly [rad] (nu will always be between 0 & 2*pi)

    Notes
    -----
    #.  Written by David C. Stauffer for AA279 on 12 May 2007.
    #.  Translated into Python by David C. Stauffer in July 2021.

    Examples
    --------
    >>> from dstauffman.aerospace import anomaly_eccentric_2_true
    >>> from math import pi
    >>> E = pi / 4
    >>> e = 0.5
    >>> nu = anomaly_eccentric_2_true(E, e)
    >>> print(f"{nu:.8f}")
    1.24466863

    """
    # check if orbit is circular or elliptical
    if np.any(e >= 1):
        raise ValueError("The true anomaly is not defined when e >= 1")
    # check if E is outside the range of 0 to 2*pi
    if np.any((E > TAU) | (E < 0)):
        logger.log(LogLevel.L6, "The eccentric anomaly was outside the range of 0 to 2*pi")
        E = np.mod(E, TAU)
    # calculate nu
    nu = np.arccos((np.cos(E) - e) / (1.0 - e * np.cos(E)))
    # check half of unit circle for 0 to pi or pi to 2*pi range
    ix = E > PI
    # correct nu if it falls in lower half of unit circle (TODO: functionalize?)
    if np.any(ix):
        if np.size(ix) == 1:
            nu = TAU - nu
        else:
            assert isinstance(nu, np.ndarray)
            nu[ix] = TAU - nu[ix]
    return nu


#%% Functions - anomaly_hyperbolic_2_mean
def anomaly_hyperbolic_2_mean(F: _N, e: _N) -> _N:
    r"""
    Finds the mean anomaly from the hyperbolic anomaly.

    Parameters
    ----------
    F : float or (N, ) ndarray
        Hyperbolic anomaly [rad]
    e : float or (N, ) ndarray
        Eccentricity

    Returns
    -------
    M : float or (N, ) ndarray
        Mean anomaly [rad] (nu will always be between 0 & 2*pi)

    Notes
    -----
    #.  Written by David C. Stauffer for AA279 on 12 May 2007.
    #.  Translated into Python by David C. Stauffer in October 2021.

    Examples
    --------
    >>> from dstauffman.aerospace import anomaly_hyperbolic_2_mean
    >>> from math import pi
    >>> F = pi/4
    >>> e = 1.5
    >>> M = anomaly_hyperbolic_2_mean(F, e)
    >>> print(f"{M:.8f}")
    0.51760828

    """
    # check if orbit is hyperbolic
    if np.any(e < 1):
        raise ValueError("The hyperbolic anomaly is not defined when e < 1")
    # calculate the mean anomaly
    M = e * np.sinh(F) - F
    return M


#%% Functions - anomaly_hyperbolic_2_true
def anomaly_hyperbolic_2_true(F: _N, e: _N) -> _N:

    r"""
    Finds the true anomaly from the hyperbolic anomaly.

    Parameters
    ----------
    F : float or (N, ) ndarray
        Hyperbolic anomaly [rad]
    e : float or (N, ) ndarray
        Eccentricity

    Returns
    -------
    nu : float or (N, ) ndarray
        True anomaly [rad] (nu will always be between 0 & 2*pi)

    Notes
    -----
    #.  Written by David C. Stauffer for AA279 on 12 May 2007.
    #.  Translated into Python by David C. Stauffer in October 2021.

    Examples
    --------
    >>> from dstauffman.aerospace import anomaly_hyperbolic_2_true
    >>> from math import pi
    >>> F = pi/4
    >>> e = 1.5
    >>> nu = anomaly_hyperbolic_2_true(F, e)
    >>> print(f"{nu:.8f}")
    1.39213073

    """
    # check if orbit is hyperbolic
    if np.any(e < 1):
        raise ValueError("The hyperbolic anomaly is not defined when e < 1")
    # calculate nu
    nu = 2 * np.arctan(np.sqrt((e + 1) / (e - 1)) * np.tanh(F / 2))  # TODO: use np.arctan2?
    # check half of unit circle for 0 to pi or pi to 2*pi range
    ix = F < 0
    # correct nu if it falls in lower half of unit circle
    if np.any(ix):
        if np.size(ix) == 1:
            nu += TAU
        else:
            assert isinstance(nu, np.ndarray)
            nu[ix] += TAU
    return nu  # type: ignore[no-any-return]


#%% Functions - anomaly_mean_2_eccentric
def anomaly_mean_2_eccentric(M, e):
    r"""
    Finds the eccentric anomaly from the mean anomaly.

    Parameters
    ----------
    M : float or (N, ) ndarray
        Mean anomaly [rad]
    e : float or (N, ) ndarray
        Eccentricity

    Returns
    -------
    E : float or (N, ) ndarray
        Eccentric anomaly [rad] (E will always be between 0 & 2*pi)

    Notes
    -----
    #.  Written by David C. Stauffer for AA279 on 12 May 2007
    #.  Translated into Python by David C. Stauffer in July 2021.

    Examples
    -------
    >>> from dstauffman.aerospace import anomaly_mean_2_eccentric
    >>> M = 0.
    >>> e = 0.5
    >>> E = anomaly_mean_2_eccentric(M, e)
    >>> print(E)
    0.0

    """

    def _anomalies(E, M, e):
        r"""(Non-linear) Relationship between anomalies to be used in root finder."""
        return M - E + e * np.sin(E)

    # check if orbit is circular or elliptical
    if np.any(e >= 1):
        raise ValueError("The mean anomaly is not defined when e >= 1")
    # check if M is outside the range of 0 to 2*pi
    if np.any((M > TAU) | (M < 0)):
        logger.log(LogLevel.L6, "The mean anomaly was outside the range of 0 to 2*pi")
        M = np.mod(M, TAU)
    # get vector lengths
    l1 = np.size(M)
    l2 = np.size(e)
    # make vectors the same lengths
    if ((l1 > 1) ^ (l2 > 1)) and l1 != l2:
        M = np.repeat(M, l2)
        e = np.repeat(e, l1)
    if l1 == 1 and l2 == 1:
        temp = root(lambda E: _anomalies(E, M, e), PI)
        E = temp.x[0]
    else:
        num = max(l1, l2)
        E = np.zeros(num)
        for i in range(num):
            # calculate the eccentric anomaly
            temp = root(lambda E: _anomalies(E, M[i], e[i]), PI)  # pylint: disable=cell-var-from-loop
            E[i] = temp.x[0]
    # mod with 2*pi in case a different solution was found
    E = np.mod(E, TAU)
    return E


#%% Functions - anomaly_mean_2_true
def anomaly_mean_2_true(M, e):
    r"""Finds the eccentric anomaly from the true anomaly."""
    E = anomaly_mean_2_eccentric(M, e)
    nu = anomaly_eccentric_2_true(E, e)
    return nu


#%% Functions - anomaly_true_2_eccentric
def anomaly_true_2_eccentric(nu, e):
    r"""Finds the true anomaly from the eccentric anomaly."""
    # check if orbit is circular or elliptical
    if np.any(e >= 1.0):
        raise ValueError("The eccentric anomaly is not defined when e >= 1")
    # check if nu is outside the range of 0 to 2*pi
    if np.any((nu > TAU) | (nu < 0.0)):
        logger.log(LogLevel.L6, "The true anomaly was outside the range of 0 to 2*pi")
        nu = np.mod(nu, TAU)
    # calculate E
    E = np.arccos((e + np.cos(nu)) / (1.0 + e * np.cos(nu)))
    # check half of unit circle for 0 to pi or pi to 2*pi range
    ix = nu > PI
    # correct nu if it falls in lower half of unit circle
    if np.any(ix):
        if np.size(ix) == 1:
            E = TAU - E
        else:
            assert isinstance(E, np.ndarray)
            E[ix] = TAU - E[ix]
    return E


#%% Functions - anomaly_true_2_hyperbolic
def anomaly_true_2_hyperbolic(nu, e):
    r"""Finds the hyperbolic anomaly from the true anomaly."""
    # check if orbit is hyperbolic
    if np.any(e < 1.0):
        raise ValueError("The hyperbolic anomaly is not defined when e < 1")
    # check if nu is outside the range of 0 to 2*pi
    if np.any((nu > TAU) | (nu < 0.0)):
        logger.log(LogLevel.L6, "The true anomaly was outside the range of 0 to 2*pi")
        nu = np.mod(nu, TAU)
    # calculate F
    F = np.arccosh((e + np.cos(nu)) / (1.0 + e * np.cos(nu)))
    # check half of unit circle for 0 to pi or pi to 2*pi range
    ix = nu > PI
    # correct nu if it falls in lower half plane
    if np.any(ix):
        if np.size(ix) == 1:
            F = -F
        else:
            F = np.negative(F, where=ix, out=F)
    return F


#%% Functions - anomaly_true_2_mean
def anomaly_true_2_mean(nu, e):
    r"""Finds the mean anomaly from the true anomaly."""
    E = anomaly_true_2_eccentric(nu, e)
    M = anomaly_eccentric_2_mean(E, e)
    return M


#%% Functions - mean_motion_2_semimajor
def mean_motion_2_semimajor(n, mu):
    r"""
    Calculates the semi-major axis from the mean motion.

    Parameters
    ----------
    n : float or (N, ) ndarray
        Mean motion of orbit
    mu : float or (N, ) ndarray
        Gravitional parameter

    Returns
    -------
    a  : float or (N, ) ndarray
        Semi-major axis

    Notes
    -----
    #.  Written by David C. Stauffer for AA279 on 12 May 2007.
    #.  Translated into Python by David C. Stauffer in July 2021.

    Examples
    --------
    >>> from dstauffman.aerospace import mean_motion_2_semimajor
    >>> import numpy as np
    >>> n = np.array([1, 7.2922e-5])
    >>> mu = np.array([1, 3.986e14])
    >>> a = mean_motion_2_semimajor(n, mu)
    >>> with np.printoptions(precision=8):
    ...     print(a)  # doctest: +NORMALIZE_WHITESPACE
    [1.00000000e+00 4.21638297e+07]

    """
    if _any(n <= 0):
        raise ValueError("The orbit is not defined when n <= 0")
    a = (mu / n**2) ** (1 / 3)
    return a


#%% Functions - period_2_semimajor
def period_2_semimajor(p: _N, mu: _N) -> _N:
    r"""
    Calculates the semi-major axis from the period.

    Parameters
    ----------
    p : float or (N, ) ndarray
        period of orbit
    mu : float or (N, ) ndarray
        gravitational parameter

    Returns
    -------
    a : float or (N, ) ndarray
        Semi-major axis

    Notes
    -----
    #.  Written by David C. Stauffer for AA279 on 12 May 2007.
    #.  Translated into Python by David C. Stauffer in October 2021.

    Examples
    --------
    >>> from dstauffman.aerospace import period_2_semimajor
    >>> import numpy as np
    >>> p = np.array([2*np.pi, 86164])
    >>> mu = np.array([1, 3.986e14])
    >>> a = period_2_semimajor(p, mu)
    >>> with np.printoptions(precision=8):
    ...     print(a)  # doctest: +NORMALIZE_WHITESPACE
    [1.00000000e+00 4.21641245e+07]

    """
    if _any(p <= 0):
        raise ValueError("The orbit is not defined when P <= 0")
    a = (mu * p**2 / (TAU**2)) ** (1 / 3)
    return a


#%% Functions - semimajor_2_mean_motion
def semimajor_2_mean_motion(a: _N, mu: _N) -> _N:
    r"""
    Calculates the mean motion from the semi-major axis.

    Parameters
    ----------
    a : float or (N, ) ndarray
        Semi-major axis
    mu : float or (N, ) ndarray
        gravitational parameter

    Returns
    -------
    nu : float or (N, ) ndarray
        Mean motion of orbit [rad]

    Notes
    -----
    #.  If the semi-major axis is not positive, then the mean motion of the orbit is undefined.
    #.  Written by David C. Stauffer for AA279 on 12 May 2007.
    #.  Translated into Python by David C. Stauffer in October 2021.

    Examples
    --------
    >>> from dstauffman.aerospace import semimajor_2_mean_motion
    >>> import numpy as np
    >>> a = np.array([1, 42164e3])
    >>> mu = np.array([1, 3.986e14])
    >>> nu = semimajor_2_mean_motion(a, mu)
    >>> with np.printoptions(precision=8):
    ...     print(nu)  # doctest: +NORMALIZE_WHITESPACE
    [1.00000000e+00 7.29215582e-05]

    """
    if _any(a <= 0):
        raise ValueError("The period is not defined when a <= 0")
    n = sqrt(mu / a**3)
    return n


#%% Functions - semimajor_2_period
def semimajor_2_period(a: _N, mu: _N) -> _N:
    r"""
    Calculates the period from the semi-major axis.

    Parameters
    ----------
    a : float or (N, ) ndarray
        Semi-major axis
    mu : float or (N, ) ndarray
        gravitational parameter

    Returns
    -------
    p : float or (N, ) ndarray
        period of orbit

    Notes
    -----
    #.  If the semi-major axis is not positive, then the mean motion of the orbit is undefined.
    #.  Written by David C. Stauffer for AA279 on 12 May 2007.
    #.  Translated into Python by David C. Stauffer in October 2021.

    Examples
    --------
    >>> from dstauffman.aerospace import semimajor_2_period
    >>> import numpy as np
    >>> a = np.array([1, 42164e3])
    >>> mu = np.array([1, 3.986e14])
    >>> p = semimajor_2_period(a, mu)
    >>> with np.printoptions(precision=8):
    ...     print(p)  # doctest: +NORMALIZE_WHITESPACE
    [6.28318531e+00 8.61636183e+04]

    """
    if _any(a <= 0):
        raise ValueError("The period is not defined when a <= 0")
    p = TAU * sqrt(a**3 / mu)
    return p


#%% Functions - sidereal_2_long
def sidereal_2_long(theta: _N, t: _N) -> _N:
    r"""
    Converts a sidereal longitude to a geographic longitude.

    Parameters
    ----------
    theta : float or (N, ) ndarray
        Sidereal longitude [rad]
    t : float or (N, ) ndarray
        Julian date

    Returns
    -------
    lon : float or (N, ) ndarray
        Geographic longitude [rad]

    Notes
    -----
    #.  Written by David C. Stauffer for AA279 on 12 May 2007.
    #.  Translated into Python by David C. Stauffer in October 2021.

    Examples
    --------
    >>> from dstauffman.aerospace import sidereal_2_long
    >>> theta = 4.839
    >>> jd = 2454587
    >>> lon = sidereal_2_long(theta, jd)
    >>> print(f"{lon:.8f}")
    -2.12997078

    """
    # epoch
    to = JULIAN["tg0_2000_time"]
    # theta at epoch
    theta_go = JULIAN["tg0_2000"]
    # find theta
    earth_rate = EARTH["omega"] * JULIAN["day"]
    lon = np.mod(theta - theta_go - earth_rate * (t - to), TAU)
    # change from (0:2*pi) range to (-pi:pi)
    ix = lon > PI
    if np.any(ix):
        if np.size(ix) == 1:
            lon -= TAU
        else:
            assert isinstance(lon, np.ndarray)
            lon[ix] -= TAU
    return lon


#%% Functions - raan_2_mltan
def raan_2_mltan(raan: _N, time_jd: _N, return_descending: bool = False) -> _N:
    r"""

    Examples
    --------
    >>> from dstauffman.aerospace import numpy_to_jd, raan_2_mltan, r_2_hms
    >>> from dstauffman import convert_datetime_to_np, DEG2RAD
    >>> import datetime
    >>> raan = DEG2RAD * 178.739073
    >>> date = datetime.datetime(2010, 6, 20, 15, 30, 45)
    >>> time_jd = numpy_to_jd(convert_datetime_to_np(date))
    >>> mltan = raan_2_mltan(raan, time_jd)
    >>> mltan_hms = r_2_hms(mltan)
    >>> print(f"{mltan_hms[0]:2.0f}:{mltan_hms[1]:2.0f}:{mltan_hms[2]:.4f}")
    17:56:20.1496

    """
    # right ascension of the sun
    (ra_sun, _) = get_sun_radec(time_jd)
    # mean local time of the ascending node (hours)
    offset = 0.0 if return_descending else PI
    mltan = np.mod(raan - ra_sun + offset, TAU)
    return mltan


#%% Functions - jd_2_sidereal
def jd_2_sidereal(time_jd):
    r"""

    Examples
    --------
    >>> from dstauffman.aerospace import jd_2_sidereal, numpy_to_jd
    >>> from dstauffman import convert_datetime_to_np
    >>> import datetime
    >>> date = datetime.datetime(1992, 8, 20, 12, 14, 0)
    >>> time_jd = numpy_to_jd(convert_datetime_to_np(date))
    >>> lst = jd_2_sidereal(time_jd)

    """

    #  delta time in days since J2000
    delta_time_days_J2000 = time_jd - JULIAN["jd_2000_01_01"]
    # Time in Julian centuries
    T = delta_time_days_J2000 * JULIAN["day"] / JULIAN["century"]
    # Vallado eq 3-45, p191
    gmst_sec = 67310.54841 + (876600 * ONE_HOUR + 8640184.812866) * T + 0.093104 * T**2 - 6.2e-6 * T**3
    # local sidereal time
    lst = TAU * np.mod(gmst_sec / ONE_DAY, 1)
    return lst


#%% Functions - quat_eci_2_ecf
def quat_eci_2_ecf(time_jd):
    r"""

    Examples
    --------
    >>> from dstauffman.aerospace import quat_eci_2_ecf, numpy_to_jd
    >>> from dstauffman import convert_datetime_to_np
    >>> import datetime
    >>> date = datetime.datetime(1992, 8, 20, 12, 14, 0)
    >>> time_jd = numpy_to_jd(convert_datetime_to_np(date))
    >>> quat = quat_eci_2_ecf(time_jd)

    """
    # calculate the local sidereal time for the given epoch
    lst = jd_2_sidereal(time_jd)
    # return the Z-axis rotation for the local time
    return qrot(3, lst)


#%% Unit Test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_aerospace_orbit_conv", exit=False)
    doctest.testmod(verbose=False)
