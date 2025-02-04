r"""
Functions related to converting orbital elements from one form to another.

Notes
-----
#.  Written by David C. Stauffer in July 2021.
"""

# %% Imports
from __future__ import annotations

import doctest
import logging
from typing import Any, TYPE_CHECKING
import unittest

from slog import LogLevel

from dstauffman import ARCSEC2RAD, HAVE_NUMPY, HAVE_SCIPY
from dstauffman.aerospace.orbit_const import EARTH, JULIAN, PI, TAU
from dstauffman.aerospace.orbit_support import d_2_r, get_sun_radec, jd_2_century
from dstauffman.aerospace.quat import qrot, quat_mult

if HAVE_NUMPY:
    import numpy as np
    from numpy import sqrt
else:
    from math import sqrt  # type: ignore[assignment]

    from nubs import np_any
if HAVE_SCIPY:
    from scipy.optimize import root
if TYPE_CHECKING:
    from numpy.typing import NDArray

    _I = NDArray[np.int_]
    _N = NDArray[np.floating]
    _FN = float | _N

# %% Globals
logger = logging.getLogger(__name__)


# %% Functions - _any
def _any(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if HAVE_NUMPY:
        return np.any(x)  # type: ignore[return-value]
    return np_any(x)  # type: ignore[no-any-return]


# %% Functions - anomaly_eccentric_2_mean
def anomaly_eccentric_2_mean(E: _FN, e: _FN) -> _FN:
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


# %% Functions - anomaly_eccentric_2_true
def anomaly_eccentric_2_true(E: _FN, e: _FN) -> _FN:
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


# %% Functions - anomaly_hyperbolic_2_mean
def anomaly_hyperbolic_2_mean(F: _FN, e: _FN) -> _FN:
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


# %% Functions - anomaly_hyperbolic_2_true
def anomaly_hyperbolic_2_true(F: _FN, e: _FN) -> _FN:
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


# %% Functions - anomaly_mean_2_eccentric
def anomaly_mean_2_eccentric(M: _FN, e: _FN) -> _FN:
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
    --------
    >>> from dstauffman.aerospace import anomaly_mean_2_eccentric
    >>> M = 0.
    >>> e = 0.5
    >>> E = anomaly_mean_2_eccentric(M, e)
    >>> print(E)
    0.0

    """

    def _anomalies(E: _FN, M: _FN, e: _FN) -> _FN:
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
            temp = root(lambda E: _anomalies(E, M[i], e[i]), PI)  # type: ignore[index]  # pylint: disable=cell-var-from-loop
            E[i] = temp.x[0]
    # mod with 2*pi in case a different solution was found
    E = np.mod(E, TAU)
    return E  # type: ignore[no-any-return]


# %% Functions - anomaly_mean_2_true
def anomaly_mean_2_true(M: _FN, e: _FN) -> _FN:
    r"""Finds the eccentric anomaly from the true anomaly."""
    E = anomaly_mean_2_eccentric(M, e)
    nu = anomaly_eccentric_2_true(E, e)
    return nu


# %% Functions - anomaly_true_2_eccentric
def anomaly_true_2_eccentric(nu: _FN, e: _FN) -> _FN:
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


# %% Functions - anomaly_true_2_hyperbolic
def anomaly_true_2_hyperbolic(nu: _FN, e: _FN) -> _FN:
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


# %% Functions - anomaly_true_2_mean
def anomaly_true_2_mean(nu: _FN, e: _FN) -> _FN:
    r"""Finds the mean anomaly from the true anomaly."""
    E = anomaly_true_2_eccentric(nu, e)
    M = anomaly_eccentric_2_mean(E, e)
    return M


# %% Functions - mean_motion_2_semimajor
def mean_motion_2_semimajor(n: _FN, mu: _FN) -> _FN:
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


# %% Functions - period_2_semimajor
def period_2_semimajor(p: _FN, mu: _FN) -> _FN:
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


# %% Functions - semimajor_2_mean_motion
def semimajor_2_mean_motion(a: _FN, mu: _FN) -> _FN:
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


# %% Functions - semimajor_2_period
def semimajor_2_period(a: _FN, mu: _FN) -> _FN:
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


# %% Functions - sidereal_2_long
def sidereal_2_long(theta: _FN, t: _FN) -> _FN:
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


# %% Functions - raan_2_mltan
def raan_2_mltan(raan: _FN, time_jd: _FN, return_descending: bool = False) -> _FN:
    r"""
    Convents RAAN to Mean Location Time of the Ascending Node.

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
    17:58:24.9631

    """
    # right ascension of the sun
    (ra_sun, _) = get_sun_radec(time_jd)
    # mean local time of the ascending node (hours)
    offset = 0.0 if return_descending else PI
    mltan = np.mod(raan - ra_sun + offset, TAU)
    return mltan


# %% Functions - jd_2_sidereal
def jd_2_sidereal(time_jd: _FN) -> _FN:
    r"""
    Converts a julian day to the local siderial time of day.

    Examples
    --------
    >>> from dstauffman.aerospace import jd_2_sidereal, numpy_to_jd
    >>> from dstauffman import convert_datetime_to_np
    >>> import datetime
    >>> date = datetime.datetime(1992, 8, 20, 12, 14, 0)
    >>> time_jd = numpy_to_jd(convert_datetime_to_np(date))
    >>> lst = jd_2_sidereal(time_jd)

    """
    # days since J2000 and time in julian centuries
    (Du, T) = jd_2_century(time_jd)
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    T5 = T4 * T
    # Earth rotation angle and Greenwich Mean Sidereal Time (Astronomical Almanac, page B8)
    theta = TAU * (0.7790_5727_32640 + 1.0027_3781_1911_35448 * Du)
    gmstp_asec = 0.014_506 + 4612.156_534 * T + 1.391_5817 * T2 - 0.000_000_44 * T3 - 0.000_029_956 * T4 - 3.68e-8 * T5
    gmst_rad = theta + ARCSEC2RAD * gmstp_asec
    # local sidereal time
    lst = np.mod(gmst_rad, TAU)
    return lst


# %% Functions - quat_eci_2_ecf_approx
def quat_eci_2_ecf_approx(time_jd: _FN) -> _FN:
    r"""
    Calculate the ECI to ECF transformation assuming the Z axis is perfectly aligned.

    Examples
    --------
    >>> from dstauffman.aerospace import quat_eci_2_ecf_approx, numpy_to_jd
    >>> from dstauffman import convert_datetime_to_np
    >>> import datetime
    >>> date = datetime.datetime(1992, 8, 20, 12, 14, 0)
    >>> time_jd = numpy_to_jd(convert_datetime_to_np(date))
    >>> quat = quat_eci_2_ecf_approx(time_jd)

    """
    # calculate the local sidereal time for the given epoch
    lst = jd_2_sidereal(time_jd)
    # return the Z-axis rotation for the local time
    return qrot(3, lst)


# %% Functions - quat_eci_2_ecf
def quat_eci_2_ecf(time_jd: _FN, ignore_nutation: bool = False) -> _FN:
    r"""
    Calculate the ECI to ECF transformation.

    Examples
    --------
    >>> from dstauffman.aerospace import JULIAN, quat_eci_2_ecf
    >>> import numpy as np
    >>> time_jd = JULIAN["jd_2000_01_01"] + 23.5 * 365.25
    >>> I2F = quat_eci_2_ecf(time_jd)
    >>> with np.printoptions(precision=8):
    ...     print(I2F)
    [-1.01100010e-03  5.13279932e-04 -8.85626371e-01  4.64397077e-01]

    """
    # days since J2000 and time in julian centuries
    (Du, T) = jd_2_century(time_jd)
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T
    T5 = T4 * T
    # combined frame bias and precession
    gamma_bar_asec = -0.052_928 + 10.556_378 * T + 0.493_2044 * T2 - 0.000_312_38 * T3 - 2.788e-6 * T4 + 2.60e-8 * T5
    theta_bar_asec = 84381.412_819 - 46.811_016 * T + 0.051_1268 * T2 + 0.000_532_89 * T3 - 0.440e-6 * T4 - 1.76e-8 * T5
    psi_bar_asec = -0.041_775 + 5038.481_484 * T + 1.558_4175 * T2 - 0.000_185_22 * T3 - 26.452e-6 * T4 - 1.48e-8 * T5
    # add nutation
    if ignore_nutation:
        delta_psi_2000A: _FN = 0.0
        delta_epsilon_2000A: _FN = 0.0
    else:
        # Accurate to only about 1 arcsecond
        # fmt: off
        delta_psi_2000A     = d_2_r(-0.0048 * np.sin(d_2_r(125.0 - 0.052_95 * Du)) - 0.0004 * np.sin(d_2_r(200.9 + 1.971_29 * Du)))
        delta_epsilon_2000A = d_2_r(+0.0026 * np.cos(d_2_r(125.0 - 0.052_95 * Du)) + 0.0002 * np.cos(d_2_r(200.9 + 1.971_29 * Du)))
        # fmt: on
    delta_psi = delta_psi_2000A + (0.4697e-6 - 2.7774e-6 * T) * delta_psi_2000A
    psi = ARCSEC2RAD * psi_bar_asec + delta_psi
    epsilon_a_deg = 23.439_279_4444 - 0.013_010_213_61 * T - 5.0861e-8 * T2 + 5.565e-7 * T3 - 1.6e-10 * T4 - 1.2056e-11 * T5
    epsilon_a = d_2_r(epsilon_a_deg)
    delta_epsilon = delta_epsilon_2000A - 2.7774e-6 * T * delta_epsilon_2000A
    true_obliquity_ecliptic = epsilon_a + delta_epsilon

    # combined frame bias, nutation and precession corrections
    R1 = qrot(1, -true_obliquity_ecliptic)
    R2 = qrot(3, -psi)
    R3 = qrot(1, ARCSEC2RAD * theta_bar_asec)
    R4 = qrot(3, ARCSEC2RAD * gamma_bar_asec)
    M = quat_mult(R1, quat_mult(R2, quat_mult(R3, R4)))

    # Earth rotation corrections
    Delta = d_2_r(125.004_555_01) + ARCSEC2RAD * (-6_962_890.5431 * T + 7.4722 * T2 + 0.007_702 * T3 - 0.000_059_39 * T4)
    Ee = 1.0 / 15.0 * (delta_psi * np.cos(epsilon_a) + 0.002_64 * ARCSEC2RAD * np.sin(Delta) + 0.000_06 * ARCSEC2RAD * np.sin(2.0 * Delta))  # fmt: skip
    gmst = jd_2_sidereal(time_jd)
    gast = gmst + Ee
    Q = quat_mult(qrot(3, gast), M)

    return Q


# %% Unit Test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_aerospace_orbit_conv", exit=False)
    doctest.testmod(verbose=False)
