r"""
Contains statistics related routines that can be independently defined and used by other modules.

Notes
-----
#.  Written by David C. Stauffer in December 2015.
"""

# %% Imports
from __future__ import annotations

import doctest
from typing import overload, TYPE_CHECKING
import unittest

from dstauffman.constants import HAVE_NUMPY
from dstauffman.units import MONTHS_PER_YEAR

if HAVE_NUMPY:
    import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    _B = NDArray[np.bool_]
    _N = NDArray[np.floating]


# %% Functions - convert_annual_to_monthly_probability
def convert_annual_to_monthly_probability(annual: _N) -> _N:
    r"""
    Convert a given annual probabily into the equivalent monthly one.

    Parameters
    ----------
    annual : numpy.ndarray
        annual probabilities, 0 <= annual <= 1

    Returns
    -------
    monthly : numpy.ndarray
        equivalent monthly probabilities, 0 <= monthly <= 1

    Raises
    ------
    ValueError
        Any probabilities outside of the [0, 1] range

    Notes
    -----
    #.  Checks for boundary cases to avoid a divide by zero warning

    Examples
    --------
    >>> from dstauffman import convert_annual_to_monthly_probability
    >>> import numpy as np
    >>> annual  = np.array([0, 0.1, 1])
    >>> monthly = convert_annual_to_monthly_probability(annual)
    >>> with np.printoptions(precision=8):
    ...     print(monthly)  # doctest: +NORMALIZE_WHITESPACE
    [0. 0.00874161  1. ]

    """
    # check ranges
    if np.any(annual < 0.0):
        raise ValueError("annual must be >= 0")
    if np.any(annual > 1.0):
        raise ValueError("annual must be <= 1")
    # convert to equivalent probability and return result
    out = np.full(annual.shape, -np.inf)
    monthly = 1.0 - np.exp(np.log(1.0 - annual, out=out, where=annual != 1.0) / MONTHS_PER_YEAR)
    return monthly


# %% Functions - convert_monthly_to_annual_probability
def convert_monthly_to_annual_probability(monthly: _N) -> _N:
    r"""
    Convert a given monthly probability into the equivalent annual one.

    Parameters
    ----------
    monthly : numpy.ndarray
        equivalent monthly probabilities, 0 <= monthly <= 1

    Returns
    -------
    annual : numpy.ndarray
        annual probabilities, 0 <= annual <= 1

    Examples
    --------
    >>> from dstauffman import convert_monthly_to_annual_probability
    >>> import numpy as np
    >>> monthly = np.array([0, 0.1, 1])
    >>> annual = convert_monthly_to_annual_probability(monthly)
    >>> with np.printoptions(precision=8):
    ...     print(annual)  # doctest: +NORMALIZE_WHITESPACE
    [0. 0.71757046 1. ]

    """
    # check ranges
    if np.any(monthly < 0.0):
        raise ValueError("monthly must be >= 0")
    if np.any(monthly > 1.0):
        raise ValueError("annual must be <= 1")
    # convert to equivalent probability and return result
    annual = 1.0 - (1.0 - monthly) ** MONTHS_PER_YEAR
    return annual


# %% Functions - ca2mp & cm2ap aliases
ca2mp = convert_annual_to_monthly_probability
cm2ap = convert_monthly_to_annual_probability


# %% Functions - prob_to_rate
@overload
def prob_to_rate(prob: float) -> float: ...
@overload
def prob_to_rate(prob: _N) -> _N: ...
@overload
def prob_to_rate(prob: float, time: int | float) -> float: ...
@overload
def prob_to_rate(prob: _N, time: int | float) -> _N: ...
def prob_to_rate(prob: float | _N, time: int | float = 1) -> float | _N:
    r"""
    Convert a given probability and time to a rate.

    Parameters
    ----------
    prob : numpy.ndarray
        Probability of event happening over the given time
    time : float
        Time for the given probability in years

    Returns
    -------
    rate : numpy.ndarray
        Equivalent annual rate for the given probability and time

    Notes
    -----
    #.  Written by David C. Stauffer in January 2016.

    Examples
    --------
    >>> from dstauffman import prob_to_rate
    >>> import numpy as np
    >>> prob = np.array([0, 0.1, 1])
    >>> time = 3
    >>> rate = prob_to_rate(prob, time)
    >>> with np.printoptions(precision=8):
    ...     print(rate)  # doctest: +NORMALIZE_WHITESPACE
    [0. 0.03512017 inf]

    """
    # check for scalar case
    was_numpy = hasattr(prob, "ndim")
    prob = np.asanyarray(prob)
    # check ranges
    if np.any(prob < 0.0):
        raise ValueError("Probability must be >= 0")
    if np.any(prob > 1.0):
        raise ValueError("Probability must be <= 1")
    # calculate rate
    rate: _N = -np.log(1.0 - prob, out=np.full(prob.shape, -np.inf), where=prob != 1) / time
    # prevent code from returning a bunch of negative zeros when prob is exactly 0
    if rate.size == 1:
        if rate == 0.0:
            rate = abs(rate)
    else:
        rate = np.abs(rate, out=rate, where=rate == 0.0)
    if not was_numpy and rate.size == 1:
        return float(rate)
    return rate


# %% Functions - rate_to_prob
@overload
def rate_to_prob(rate: float) -> float: ...
@overload
def rate_to_prob(rate: _N) -> _N: ...
@overload
def rate_to_prob(rate: float, time: int | float) -> float: ...
@overload
def rate_to_prob(rate: _N, time: int | float) -> _N: ...
def rate_to_prob(rate: float | _N, time: int | float = 1) -> float | _N:
    r"""
    Convert a given rate and time to a probability.

    Parameters
    ----------
    rate : numpy.ndarray
        Annual rate for the given time
    time : float
        Time period for the desired probability to be calculated from, in years

    Returns
    -------
    prob : numpy.ndarray
        Equivalent probability of event happening over the given time

    Notes
    -----
    #.  Written by David C. Stauffer in January 2016.

    Examples
    --------
    >>> from dstauffman import rate_to_prob
    >>> import numpy as np
    >>> rate = np.array([0, 0.1, 1, 100, np.inf])
    >>> time = 1./12
    >>> prob = rate_to_prob(rate, time)
    >>> with np.printoptions(precision=8):
    ...     print(prob)  # doctest: +NORMALIZE_WHITESPACE
    [0. 0.00829871 0.07995559 0.99975963 1. ]

    """
    # check ranges
    if np.any(rate < 0.0):
        raise ValueError("Rate must be >= 0")
    # calculate probability
    prob = 1.0 - np.exp(-rate * time)
    return prob


# %% Functions - annual_rate_to_monthly_probability
@overload
def annual_rate_to_monthly_probability(rate: float) -> float: ...
@overload
def annual_rate_to_monthly_probability(rate: _N) -> _N: ...
def annual_rate_to_monthly_probability(rate: float | _N) -> float | _N:
    r"""
    Convert a given annual rate to a monthly probability.

    Parameters
    ----------
    rate : numpy.ndarray
        Annual rate

    Returns
    -------
    prob : numpy.ndarray
        Equivalent monthly probability

    Notes
    -----
    #.  Written by David C. Stauffer in January 2016.

    See Also
    --------
    rate_to_prob

    Examples
    --------
    >>> from dstauffman import annual_rate_to_monthly_probability
    >>> import numpy as np
    >>> rate = np.array([0, 0.5, 1, 5, np.inf])
    >>> prob = annual_rate_to_monthly_probability(rate)
    >>> with np.printoptions(precision=8):
    ...     print(prob)  # doctest: +NORMALIZE_WHITESPACE
    [0. 0.04081054 0.07995559 0.34075937 1. ]

    """
    # divide rate and calculate probability
    prob = rate_to_prob(rate / MONTHS_PER_YEAR)
    return prob


# %% Functions - monthly_probability_to_annual_rate
@overload
def monthly_probability_to_annual_rate(prob: float) -> float: ...
@overload
def monthly_probability_to_annual_rate(prob: _N) -> _N: ...
def monthly_probability_to_annual_rate(prob: float | _N) -> float | _N:
    r"""
    Convert a given monthly probability to an annual rate.

    Parameters
    ----------
    prob : numpy.ndarray
        Monthly probability

    Returns
    -------
    rate : numpy.ndarray
        Equivalent annual rate

    Notes
    -----
    #.  Written by David C. Stauffer in April 2016.

    See Also
    --------
    prob_to_rate

    Examples
    --------
    >>> from dstauffman import monthly_probability_to_annual_rate
    >>> import numpy as np
    >>> prob = np.array([0, 0.04081054, 0.07995559, 0.34075937, 1])
    >>> rate = monthly_probability_to_annual_rate(prob)
    >>> print(" ".join(("{:.2f}".format(x) for x in rate)))  # doctest: +NORMALIZE_WHITESPACE
    0.00 0.50 1.00 5.00 inf

    """
    # divide rate and calculate probability
    rate = prob_to_rate(prob, time=1 / MONTHS_PER_YEAR)
    return rate


# %% Functions - ar2mp
ar2mp = annual_rate_to_monthly_probability
mp2ar = monthly_probability_to_annual_rate


# %% Functions - rand_draw
def rand_draw(chances: _N, prng: np.random.RandomState, *, check_bounds: bool = True) -> _B:
    r"""
    Draws psuedo-random numbers from the given generator to compare to given factors.

    Has optimizations to ignore factors less than or equal to zero.

    Parameters
    ----------
    chances : ndarray of float
        Probability that someone should be chosen
    prng : class numpy.random.RandomState
        Pseudo-random number generator
    check_bounds : bool
        Whether this function should check for known outcomes and not generate random numbers for
        them, default is True

    Returns
    -------
    is_set : ndarray of bool
        True/False for whether the chance held out

    Notes
    -----
    #.  Written by David C. Stauffer in April 2018.

    See Also
    --------
        numpy.random.rand

    Examples
    --------
    >>> from dstauffman import rand_draw
    >>> import numpy as np
    >>> chances = np.array([-0.5, 0., 0.5, 1., 5, np.inf])
    >>> prng = np.random.RandomState()
    >>> is_set = rand_draw(chances, prng)
    >>> print(is_set[0])
    False

    >>> print(is_set[5])
    True

    """
    # simple version
    if not check_bounds:
        is_set = prng.rand(*chances.shape) < chances
        return is_set

    # find those who need a random number draw
    eligible = (chances > 0) & (chances <= 1)
    # initialize output assuming no one is selected
    is_set = np.zeros(chances.shape, dtype=bool)
    # determine who got picked
    is_set[eligible] = prng.rand(np.count_nonzero(eligible)) < chances[eligible]
    # set those who were always going to be chosen
    is_set[chances >= 1] = True
    return is_set


# %% Functions - apply_prob_to_mask
def apply_prob_to_mask(mask: _B, prob: float, prng: np.random.RandomState, inplace: bool = False) -> _B:
    r"""
    Applies a one-time probability to a logical mask while minimizing the random number calls.

    Parameters
    ----------
    mask : ndarray of bool
        Input mask
    prob : float
        Probability to apply to mask

    Returns
    -------
    out : ndarray of bool
        Output mask

    Notes
    -----
    #.  Written by David C. Stauffer in August 2022.

    Examples
    --------
    >>> from dstauffman import apply_prob_to_mask
    >>> import numpy as np
    >>> prng = np.random.RandomState()
    >>> mask = prng.rand(50000) < 0.5
    >>> prob = 0.3
    >>> out = apply_prob_to_mask(mask, prob, prng)
    >>> assert np.count_nonzero(mask) < 30000, "Too many trues in mask."
    >>> assert np.count_nonzero(out) < 0.4 * np.count_nonzero(mask), "Too many trues in out."

    """
    out = mask if inplace else mask.copy()

    keep = prng.rand(np.count_nonzero(mask)) < prob
    out[mask] &= keep
    return out


# %% Unit test
if __name__ == "__main__":
    unittest.main(module="dstauffman.tests.test_stats", exit=False)
    doctest.testmod(verbose=False)
