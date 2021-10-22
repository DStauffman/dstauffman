r"""
Contains statistics related routines that can be independently defined and used by other modules.

Notes
-----
#.  Written by David C. Stauffer in December 2015.
"""

#%% Imports
from __future__ import annotations
import doctest
from typing import Dict, Tuple
import unittest

from dstauffman import HAVE_NUMPY, MONTHS_PER_YEAR

if HAVE_NUMPY:
    import numpy as np
    sqrt = np.sqrt
else:
    from math import sqrt  # type: ignore[misc]

#%% Functions - convert_annual_to_monthly_probability
def convert_annual_to_monthly_probability(annual):
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
    >>> from dstauffman.health import convert_annual_to_monthly_probability
    >>> import numpy as np
    >>> annual  = np.array([0, 0.1, 1])
    >>> monthly = convert_annual_to_monthly_probability(annual)
    >>> with np.printoptions(precision=8):
    ...     print(monthly) # doctest: +NORMALIZE_WHITESPACE
    [0. 0.00874161  1. ]

    """
    # check ranges
    if np.any(annual < 0):
        raise ValueError('annual must be >= 0')
    if np.any(annual > 1):
        raise ValueError('annual must be <= 1')
    # convert to equivalent probability and return result
    monthly = 1-np.exp(np.log(1-annual, out=np.full(annual.shape, -np.inf), where=annual!=1)/MONTHS_PER_YEAR)
    return monthly

#%% Functions - convert_monthly_to_annual_probability
def convert_monthly_to_annual_probability(monthly):
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
    >>> from dstauffman.health import convert_monthly_to_annual_probability
    >>> import numpy as np
    >>> monthly = np.array([0, 0.1, 1])
    >>> annual = convert_monthly_to_annual_probability(monthly)
    >>> with np.printoptions(precision=8):
    ...     print(annual) # doctest: +NORMALIZE_WHITESPACE
    [0. 0.71757046 1. ]

    """
    # check ranges
    if np.any(monthly < 0):
        raise ValueError('monthly must be >= 0')
    if np.any(monthly > 1):
        raise ValueError('annual must be <= 1')
    # convert to equivalent probability and return result
    annual = 1 - (1 - monthly)**MONTHS_PER_YEAR
    return annual

#%% Functions - ca2mp & cm2ap aliases
ca2mp = convert_annual_to_monthly_probability
cm2ap = convert_monthly_to_annual_probability

#%% Functions - prob_to_rate
def prob_to_rate(prob, time=1):
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
    >>> from dstauffman.health import prob_to_rate
    >>> import numpy as np
    >>> prob = np.array([0, 0.1, 1])
    >>> time = 3
    >>> rate = prob_to_rate(prob, time)
    >>> with np.printoptions(precision=8):
    ...     print(rate) # doctest: +NORMALIZE_WHITESPACE
    [0. 0.03512017 inf]

    """
    # check for scalar case
    was_numpy = hasattr(prob, 'ndim')
    prob = np.asanyarray(prob)
    # check ranges
    if np.any(prob < 0):
        raise ValueError('Probability must be >= 0')
    if np.any(prob > 1):
        raise ValueError('Probability must be <= 1')
    # calculate rate
    rate = -np.log(1 - prob, out=np.full(prob.shape, -np.inf), where=prob!=1) / time
    # prevent code from returning a bunch of negative zeros when prob is exactly 0
    if rate.size == 1:
        if rate == 0.:
            rate = abs(rate)
    else:
        rate = np.abs(rate, out=rate, where=rate == 0.)
    if not was_numpy and rate.size == 1:
        return float(rate)
    return rate

#%% Functions - rate_to_prob
def rate_to_prob(rate, time=1):
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
    >>> from dstauffman.health import rate_to_prob
    >>> import numpy as np
    >>> rate = np.array([0, 0.1, 1, 100, np.inf])
    >>> time = 1./12
    >>> prob = rate_to_prob(rate, time)
    >>> with np.printoptions(precision=8):
    ...     print(prob) # doctest: +NORMALIZE_WHITESPACE
    [0. 0.00829871 0.07995559 0.99975963 1. ]

    """
    # check ranges
    if np.any(rate < 0):
        raise ValueError('Rate must be >= 0')
    # calculate probability
    prob = 1 - np.exp(-rate * time)
    return prob

#%% Functions - annual_rate_to_monthly_probability
def annual_rate_to_monthly_probability(rate):
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
    >>> from dstauffman.health import annual_rate_to_monthly_probability
    >>> import numpy as np
    >>> rate = np.array([0, 0.5, 1, 5, np.inf])
    >>> prob = annual_rate_to_monthly_probability(rate)
    >>> with np.printoptions(precision=8):
    ...     print(prob) # doctest: +NORMALIZE_WHITESPACE
    [0. 0.04081054 0.07995559 0.34075937 1. ]

    """
    # divide rate and calculate probability
    prob = rate_to_prob(rate/MONTHS_PER_YEAR)
    return prob

#%% Functions - monthly_probability_to_annual_rate
def monthly_probability_to_annual_rate(prob):
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
    >>> from dstauffman.health import monthly_probability_to_annual_rate
    >>> import numpy as np
    >>> prob = np.array([0, 0.04081054, 0.07995559, 0.34075937, 1])
    >>> rate = monthly_probability_to_annual_rate(prob)
    >>> print(' '.join(('{:.2f}'.format(x) for x in rate))) # doctest: +NORMALIZE_WHITESPACE
    0.00 0.50 1.00 5.00 inf

    """
    # divide rate and calculate probability
    rate = prob_to_rate(prob, time=1/MONTHS_PER_YEAR)
    return rate

#%% Functions - ar2mp
ar2mp = annual_rate_to_monthly_probability
mp2ar = monthly_probability_to_annual_rate

#%% Functions - combine_sets
def combine_sets(n1: int, u1: float, s1: float, n2: int, u2: float, s2: float) -> Tuple[int, float, float]:
    r"""
    Combine the mean and standard deviations for two non-overlapping sets of data.

    This function combines two non-overlapping data sets, given a number of samples, mean
    and standard deviation for the two data sets.  It first calculates the total number of samples
    then calculates the total mean using a weighted average, and then calculates the combined
    standard deviation using an equation found on wikipedia.  It also checks for special cases
    where either data set is empty or if only one total point is in the combined set.

    Parameters
    ----------
    n1 : int
        number of points in data set 1
    u1 : float
        mean of data set 1
    s1 : float
        standard deviation of data set 1
    n2 : int
        number of points in data set 2
    u2 : float
        mean of data set 2
    s2 : float
        standard deviation of data set 2

    Returns
    -------
    n  : int
        number of points in the combined data set
    u  : float
        mean of the combined data set
    s  : float
        standard deviation of the combined data set

    See Also
    --------
    np.mean, np.std

    References
    ----------
    #.  http://en.wikipedia.org/wiki/Standard_deviation#Sample-based_statistics, on 8/7/12

    Notes
    -----
    #.  Written in Matlab by David C. Stauffer in Jul 2012.
    #.  Ported to Python by David C. Stauffer in May 2015.
    #.  Could be expanded to broadcast and handle array inputs.

    Examples
    --------
    >>> from dstauffman.health import combine_sets
    >>> n1 = 5
    >>> u1 = 1
    >>> s1 = 0.5
    >>> n2 = 10
    >>> u2 = 2
    >>> s2 = 0.25
    >>> (n, u, s) = combine_sets(n1, u1, s1, n2, u2, s2)
    >>> print(n)
    15
    >>> print(u) # doctest: +ELLIPSIS
    1.666666...67
    >>> print(s)
    0.591356390810466

    """
    # assertions
    assert n1 >= 0
    assert n2 >= 0
    assert s1 >= 0.
    assert s2 >= 0.
    # combine total number of samples
    n = n1 + n2
    # check for zero case
    if n == 0:
        u = 0.
        s = 0.
        return (n, u, s)
    # calculate the combined mean
    u = 1/n * (n1*u1 + n2*u2)
    # calculate the combined standard deviation
    if n != 1:
        s = sqrt(1/(n-1) * ( (n1-1)*s1**2 + n1*u1**2 + (n2-1)*s2**2 + n2*u2**2 - n*u**2))
    else:
        # special case where one of the data sets is empty
        if n1 == 1:
            s = s1
        elif n2 == 1:
            s = s2
        else:
            # shouldn't be able to ever reach this line with assertions on
            raise ValueError('Total samples are 1, but neither data set has only one item.')  # pragma: no cover
    return (n, u, s)

#%% Functions - bounded_normal_draw
def bounded_normal_draw(num: int, values: Dict[str, float], field: str, prng: np.random.RandomState) -> np.ndarray:
    r"""
    Create a normalized distribution with the given mean and standard deviations.

    Includes options for min and max bounds, all taken from a dictionary with the specified `field`
    name.

    Parameters
    ----------
    num : int
        Number of random draws to make
    values : dict
        Dictionary of mean, std, min and max values
    field : str
        Name of field that is prepended to the values
    prng : class numpy.random.RandomState
        Pseudo-random number generator

    Returns
    -------
    out : ndarray (N,)
        Normalized random numbers

    Notes
    -----
    #.  Written by David C. Stauffer in March 2017.

    Examples
    --------
    >>> from dstauffman.health import bounded_normal_draw
    >>> import numpy as np
    >>> num   = 10
    >>> values = {'last_mean': 2, 'last_std': 0.5, 'last_min': 1, 'last_max': 3}
    >>> field  = 'last'
    >>> prng   = np.random.RandomState()
    >>> out    = bounded_normal_draw(num, values, field, prng)

    """
    # get values from the dictionary
    try:
        this_mean = values[field + '_mean']
    except KeyError:
        this_mean = 0
    try:
        this_std  = values[field + '_std']
    except KeyError:
        this_std  = 1
    try:
        this_min  = values[field + '_min']
    except KeyError:
        this_min  = -np.inf
    try:
        this_max  = values[field + '_max']
    except KeyError:
        this_max  = np.inf
    # calculate the normal distribution
    if this_std == 0:
        out = np.full(num, this_mean)
    else:
        out = prng.normal(this_mean, this_std, size=num)
    # enforce the min and maxes
    np.minimum(out, this_max, out)
    np.maximum(out, this_min, out)
    return out

#%% Functions - rand_draw
def rand_draw(chances, prng, *, check_bounds=True):
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
    >>> from dstauffman.health import rand_draw
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

#%% Functions - ecdf
def ecdf(y, /):
    r"""
    Calculate the empirical cumulative distribution function, as in Matlab's ecdf function.

    Parameters
    ----------
    array_like of float
        Input samples

    Returns
    -------
    x : ndarray of float
        cumulative probability
    f : ndarray of float
        function values evaluated at the points returned in x

    Notes
    -----
    #.  Written by David C. Stauffer in February 2021.

    Examples
    --------
    >>> from dstauffman.health import ecdf
    >>> import numpy as np
    >>> y = np.random.rand(1000)
    >>> (x, f) = ecdf(y)
    >>> exp = np.arange(0.001, 1.001, 0.001)
    >>> print(np.max(np.abs(f - exp)) < 0.05)
    True

    """
    f, counts = np.unique(y, return_counts=True)
    x = np.cumsum(counts) / np.size(y)
    return (x, f)

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='dstauffman.tests.test_health_stats', exit=False)
    doctest.testmod(verbose=False)
