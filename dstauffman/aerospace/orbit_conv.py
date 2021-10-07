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

from dstauffman import HAVE_NUMPY, HAVE_SCIPY, LogLevel

from dstauffman.aerospace.orbit_const import EARTH, JULIAN, PI, TAU

if HAVE_NUMPY:
    import numpy as np
    from numpy import sqrt
else:
    from math import sqrt  # type: ignore[misc]
    from dstauffman.nubs import np_any
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
    elif isinstance(x, (int, float)):
        return bool(x)
    elif HAVE_NUMPY:
        return np.any(x)  # type: ignore[return-value]
    else:
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
    >>> print(f'{M:.8f}')
    0.43184477

    """
    # check if orbit is circular or elliptical
    if np.any(e >= 1):
        raise ValueError('The mean anomaly is not defined when e >= 1')
    # check if E is outside the range of 0 to 2*pi
    if np.any((E > TAU) | (E < 0)):
        logger.log(LogLevel.L6, 'The eccentric anomaly was outside the range of 0 to 2*pi')
        E = np.mod(E, TAU)
    # calculate the mean anomaly
    M = E - e*np.sin(E)
    return M  # type: ignore[no-any-return]

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
    >>> print(f'{nu:.8f}')
    1.24466863

    """
    # check if orbit is circular or elliptical
    if np.any(e >= 1):
        raise ValueError('The true anomaly is not defined when e >= 1')
    # check if E is outside the range of 0 to 2*pi
    if np.any((E > TAU) | (E < 0)):
        logger.log(LogLevel.L6, 'The eccentric anomaly was outside the range of 0 to 2*pi')
        E = np.mod(E, TAU)
    # calculate nu
    nu = np.arccos((np.cos(E) - e) / (1. - e*np.cos(E)))
    # check half of unit circle for 0 to pi or pi to 2*pi range
    ix = E > PI
    # correct nu if it falls in lower half of unit circle (TODO: functionalize?)
    if np.any(ix):
        if np.size(ix) == 1:
            nu = TAU - nu
        else:
            assert isinstance(nu, np.ndarray)
            nu[ix] = TAU - nu[ix]
    return nu  # type: ignore[no-any-return]

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
    >>> print(f'{M:.8f}')
    0.51760828

    """
    # check if orbit is hyperbolic
    if np.any(e < 1):
        raise ValueError('The hyperbolic anomaly is not defined when e < 1')
    # calculate the mean anomaly
    M = e*np.sinh(F) - F
    return M  # type: ignore[no-any-return]

#%% Functions - anomaly_hyperbolic_2_true
def anomaly_hyperbolic_2_true(F:_N, e: _N) -> _N:

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
    >>> print(f'{nu:.8f}')
    1.39213073

    """
    # check if orbit is hyperbolic
    if np.any(e < 1):
        raise ValueError('The hyperbolic anomaly is not defined when e < 1')
    # calculate nu
    nu = 2*np.arctan(np.sqrt((e + 1)/(e - 1)) * np.tanh(F/2))  # TODO: use np.arctan2?
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
    # check if orbit is circular or elliptical
    if np.any(e >= 1):
        raise ValueError('The mean anomaly is not defined when e >= 1')
    # check if M is outside the range of 0 to 2*pi
    if np.any((M > TAU) | (M < 0)):
        logger.log(LogLevel.L6, 'The mean anomaly was outside the range of 0 to 2*pi')
        M = np.mod(M, TAU)
    # get vector lengths
    l1 = np.size(M)
    l2 = np.size(e)
    # make vectors the same lengths
    if ((l1 > 1) ^ (l2 > 1)) and l1 != l2:
        M = np.repeat(M, l2)
        e = np.repeat(e, l1)
    if l1 == 1 and l2 == 1:
        temp = root(lambda E: M - E + e * np.sin(E), PI)
        E = temp.x[0]
    else:
        num = max(l1, l2)
        E = np.zeros(num)
        for i in range(num):
            # calculate the eccentric anomalyS
            temp = root(lambda E: M[i] - E + e[i] * np.sin(E), PI)
            E[i] = temp.x[0]
    # mod with 2*pi in case a different solution was found
    E = np.mod(E, TAU)
    return E

#%% Functions - anomaly_mean_2_true

#%% Functions - anomaly_true_2_eccentric

#%% Functions - anomaly_true_2_hyperbolic

#%% Functions - anomaly_true_2_mean

#%% Functions - long_2_sidereal
def long_2_sidereal(lon: _N, jd: _N) -> _N:
    r"""
    Converts a geographic longitude to sidereal longitude.

    Parameters
    ----------
    lon : float or (N, ) ndarray
        Geographic longitude [rad]
    t : float or (N, ) ndarray
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
    >>> jd = 2454587
    >>> theta = long_2_sidereal(lon, jd)
    >>> print(f'{theta:.8f}')
    4.83897078

    """
    # epoch
    to = JULIAN['tg0_2000_time']
    # theta at epoch
    theta_go = JULIAN['tg0_2000']
    # earth rate per day
    earth_rate = EARTH['omega']*JULIAN['day']
    # find theta
    theta = np.mod(theta_go + earth_rate*(jd - to) + lon, TAU)
    return theta  # type: ignore[no-any-return]

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
        raise ValueError('The orbit is not defined when n <= 0')
    a = (mu/n**2)**(1/3)
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
        raise ValueError('The orbit is not defined when P <= 0')
    a = (mu*p**2/(TAU**2))**(1/3)
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
        raise ValueError('The period is not defined when a <= 0')
    n = sqrt(mu/a**3)
    return n  # type: ignore[no-any-return]

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
        raise ValueError('The period is not defined when a <= 0')
    p = TAU*sqrt(a**3/mu)
    return p  # type: ignore[no-any-return]

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
    >>> print(f'{lon:.8f}')
    -2.12997078

    """
    # epoch
    to = JULIAN['tg0_2000_time']
    # theta at epoch
    theta_go = JULIAN['tg0_2000']
    # find theta
    earth_rate = EARTH['omega']*JULIAN['day']
    lon = np.mod(theta - theta_go - earth_rate*(t - to), TAU)
    # change from (0:2*pi) range to (-pi:pi)
    ix = lon > PI
    if np.any(ix):
        if np.size(ix) == 1:
            lon -= TAU
        else:
            assert isinstance(lon, np.ndarray)
            lon[ix] -= TAU
    return lon  # type: ignore[no-any-return]

#%% Unit Test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_aerospace_orbit_conv', exit=False)
    doctest.testmod(verbose=False)
