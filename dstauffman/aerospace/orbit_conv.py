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
from typing import TYPE_CHECKING, Union
import unittest

from dstauffman import HAVE_NUMPY, HAVE_SCIPY, LogLevel

#from dstauffman.aerospace.vectors import vec_cross

if HAVE_NUMPY:
    import numpy as np
    from numpy import pi as PI
else:
    from math import pi as PI
if HAVE_SCIPY:
    from scipy.optimize import root
if TYPE_CHECKING:
    _N = Union[float, np.ndarray]

#%% Constants
TWO_PI = 2 * PI

#%% Globals
logger = logging.getLogger(__name__)

#%% Functions - function nu = anomaly_eccentric_2_true(E, e)
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
    #.  Written by David C. Stauffer for AA279 on 12 May 2007
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
    if np.any((E > TWO_PI) | (E < 0)):
        logger.log(LogLevel.L6, 'The eccentricity anomaly was outside the range of 0 to 2*pi')
        E = np.mod(E, TWO_PI)
    # calculate nu
    nu = np.arccos((np.cos(E) - e) / (1. - e*np.cos(E)))
    # check half of unit circle for 0 to pi or pi to 2*pi range
    index = E > PI
    # correct nu if it falls in lower half of unit circle (TODO: functionalize?)
    if np.size(index) == 1:
        if index:
            nu = TWO_PI - nu
    else:
        assert isinstance(nu, np.ndarray)
        nu[index] = TWO_PI - nu[index]
    return nu  # type: ignore[no-any-return]

#%% Functions - anomaly_mean_2_eccentric
def anomaly_mean_2_eccentric(M, e):
    r"""
    Finds the eccentric anomaly from the mean anomaly.

    Parameters
    M : float or (N, ) ndarray
        Mean anomaly [rad]
    e : float or (N, ) ndarray
        Eccentricity

    Parameters
    ----------
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
    if np.any((M > TWO_PI) | (M < 0)):
        logger.log(LogLevel.L6, 'The mean anomaly was outside the range of 0 to 2*pi')
        M = np.mod(M, TWO_PI)
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
    E = np.mod(E, TWO_PI)
    return E

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
    if np.any(n <= 0):
        raise ValueError('The orbit is not defined when n <= 0')

    a = (mu/n**2)**(1/3)
    return a

#%% Unit Test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_aerospace_orbit_conv', exit=False)
    doctest.testmod(verbose=False)
