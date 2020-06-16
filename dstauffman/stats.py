# -*- coding: utf-8 -*-
r"""
Contains statistics related routines that can be independently defined and used by other modules.

Notes
-----
#.  Written by David C. Stauffer in December 2015.

"""

#%% Imports
import doctest
from functools import reduce
import unittest
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

from dstauffman.plot_support import is_datetime
from dstauffman.units import MONTHS_PER_YEAR
from dstauffman.utils import is_np_int

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
    >>> from dstauffman import convert_annual_to_monthly_probability
    >>> import numpy as np
    >>> annual  = np.array([0, 0.1, 1])
    >>> monthly = convert_annual_to_monthly_probability(annual)
    >>> print(monthly) # doctest: +NORMALIZE_WHITESPACE
    [0. 0.00874161  1. ]

    """
    # check ranges
    if np.any(annual < 0):
        raise ValueError('annual must be >= 0')
    if np.any(annual > 1):
        raise ValueError('annual must be <= 1')
    # convert to equivalent probability and return result
    monthly = 1-np.exp(np.log(1-annual)/MONTHS_PER_YEAR)
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
    >>> from dstauffman import convert_monthly_to_annual_probability
    >>> import numpy as np
    >>> monthly = np.array([0, 0.1, 1])
    >>> annual = convert_monthly_to_annual_probability(monthly)
    >>> print(annual) # doctest: +NORMALIZE_WHITESPACE
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
    >>> from dstauffman import prob_to_rate
    >>> import numpy as np
    >>> prob = np.array([0, 0.1, 1])
    >>> time = 3
    >>> rate = prob_to_rate(prob, time)
    >>> print(rate) # doctest: +NORMALIZE_WHITESPACE
    [0. 0.03512017 inf]

    """
    # check ranges
    if np.any(prob < 0):
        raise ValueError('Probability must be >= 0')
    if np.any(prob > 1):
        raise ValueError('Probability must be <= 1')
    # calculate rate
    rate = -np.log(1 - prob) / time
    # prevent code from returning a bunch of negative zeros when prob is exactly 0
    rate += 0.
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
    >>> from dstauffman import rate_to_prob
    >>> import numpy as np
    >>> rate = np.array([0, 0.1, 1, 100, np.inf])
    >>> time = 1./12
    >>> prob = rate_to_prob(rate, time)
    >>> print(prob) # doctest: +NORMALIZE_WHITESPACE
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
    >>> from dstauffman import annual_rate_to_monthly_probability
    >>> import numpy as np
    >>> rate = np.array([0, 0.5, 1, 5, np.inf])
    >>> prob = annual_rate_to_monthly_probability(rate)
    >>> print(prob) # doctest: +NORMALIZE_WHITESPACE
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
    >>> from dstauffman import monthly_probability_to_annual_rate
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
def combine_sets(n1, u1, s1, n2, u2, s2):
    r"""
    Combine the mean and standard deviations for two non-overlapping sets of data.

    This function combines two non-overlapping data sets, given a number of samples, mean
    and standard deviation for the two data sets.  It first calculates the total number of samples
    then calculates the total mean using a weighted average, and then calculates the combined
    standard deviation using an equation found on wikipedia.  It also checks for special cases
    where either data set is empty or if only one total point is in the combined set.

    Parameters
    ----------
    n1 : float
        number of points in data set 1
    u1 : float
        mean of data set 1
    s1 : float
        standard deviation of data set 1
    n2 : float
        number of points in data set 2
    u2 : float
        mean of data set 2
    s2 : float
        standard deviation of data set 2

    Returns
    -------
    n  : float,
        number of points in the combined data set
    u  : float,
        mean of the combined data set
    s  : float,
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
    >>> from dstauffman import combine_sets
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
    assert s1 >= 0
    assert s2 >= 0
    # combine total number of samples
    n = n1 + n2
    # check for zero case
    if n == 0:
        u = 0
        s = 0
        return (n, u, s)
    # calculate the combined mean
    u = 1/n * (n1*u1 + n2*u2)
    # calculate the combined standard deviation
    if n != 1:
        s = np.sqrt(1/(n-1) * ( (n1-1)*s1**2 + n1*u1**2 + (n2-1)*s2**2 + n2*u2**2 - n*u**2))
    else:
        # special case where one of the data sets is empty
        if n1 == 1:
            s = s1
        elif n2 == 1:
            s = s2
        else:
            # shouldn't be able to ever reach this line with assertions on
            raise ValueError('Total samples are 1, but neither data set has only one item.') # pragma: no cover
    return (n, u, s)

#%% Functions - bounded_normal_draw
def bounded_normal_draw(num, values, field, prng):
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
    >>> from dstauffman import bounded_normal_draw
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

#%% Functions - z_from_ci
def z_from_ci(ci):
    r"""
    Calculates the Z score that matches the desired confidence interval.

    Parameters
    ----------
    ci : float
        Desired confidence interval

    Returns
    -------
    z : float
        Desired z value

    Notes
    -----
    #.  Written by David C. Stauffer in October 2017 based on:
        https://stackoverflow.com/questions/20864847/probability-to-z-score-and-vice-versa-in-python

    Examples
    --------
    >>> from dstauffman import z_from_ci
    >>> ci = 0.95
    >>> z = z_from_ci(ci)
    >>> print('{:.2f}'.format(z))
    1.96

    """
    z = st.norm.ppf(1-(1-ci)/2)
    return z

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

#%% Functions - intersect
def intersect(a, b, *, tolerance=0, assume_unique=False, return_indices=False):
    r"""
    Finds the intersect of two arrays given a numerical tolerance.

    Return the sorted, unique values that are in both of the input arrays.

    Parameters
    ----------
    a, b : array_like
        Input arrays. Will be flattened if not already 1D.
    tolerance : float or int
        Tolerance for which something is considered unique
    assume_unique : bool
        If True, the input arrays are both assumed to be unique, which
        can speed up the calculation.  Default is False.
    return_indices : bool
        If True, the indices which correspond to the intersection of the two
        arrays are returned. The first instance of a value is used if there are
        multiple. Default is False.

    Returns
    -------
    c : ndarray
        Sorted 1D array of common and unique elements.
    ia : ndarray
        The indices of the first occurrences of the common values in `ar1`.
        Only provided if `return_indices` is True.
    ib : ndarray
        The indices of the first occurrences of the common values in `ar2`.
        Only provided if `return_indices` is True.

    See Also
    --------
    numpy.intersect1d : Function used to do comparsion with sets of quantized inputs.

    Notes
    -----
    #.  Written by David C. Stauffer in March 2019.
    #.  Updated by David C. Stauffer in June 2020 to allow for a numeric tolerance.

    Examples
    --------
    >>> from dstauffman import intersect
    >>> import numpy as np
    >>> a = np.array([1, 2, 4, 4, 6], dtype=int)
    >>> b = np.array([0, 8, 2, 2, 5, 8, 6, 8, 8], dtype=int)
    >>> (c, ia, ib) = intersect(a, b, return_indices=True)
    >>> print(c)
    [2 6]

    >>> print(ia)
    [1 4]

    >>> print(ib)
    [2 6]

    """
    # allow a zero tolerance to be passed in and behave like the normal intersect command
    if tolerance == 0:
        return np.intersect1d(a, b, assume_unique=assume_unique, return_indices=return_indices)

    # allow list and other array_like inputs (or just scalar floats)
    a = np.atleast_1d(np.asanyarray(a))
    b = np.atleast_1d(np.asanyarray(b))
    tolerance = np.asanyarray(tolerance)

    # check for datetimes and convert to integers
    is_dates = np.array([is_datetime(a), is_datetime(b)], dtype=bool)
    assert np.count_nonzero(is_dates) != 1, 'Both arrays must be datetimes if either is.'
    if np.any(is_dates):
        orig_datetime = a.dtype
        a = a.astype(np.int64)
        b = b.astype(np.int64)
        tolerance = tolerance.astype(np.int64)

    # check if largest component of a and b is too close to the tolerance floor (for floats)
    all_int = is_np_int(a) and is_np_int(b) and is_np_int(tolerance)
    max_a_or_b = np.max((np.max(np.abs(a), initial=0), np.max(np.abs(b), initial=0)))
    if not all_int and ((max_a_or_b / tolerance) > (0.01/ np.finfo(float).eps)):
        warnings.warn('This function may have problems if tolerance gets too small.')

    # due to the splitting of the quanta, two very close numbers could still fail the quantized intersect
    # fix this by repeating the comparison when shifted by half a quanta in either direction
    half_tolerance = tolerance / 2
    if all_int:
        # allow for integer versions of half a quanta in either direction
        lo_tol = np.floor(half_tolerance).astype(tolerance.dtype)
        hi_tol = np.ceil(half_tolerance).astype(tolerance.dtype)
    else:
        lo_tol = half_tolerance
        hi_tol = half_tolerance

    # create quantized version of a & b, plus each one shifted by half a quanta
    a1 = np.floor_divide(a, tolerance)
    b1 = np.floor_divide(b, tolerance)
    a2 = np.floor_divide(a - lo_tol, tolerance)
    b2 = np.floor_divide(b - lo_tol, tolerance)
    a3 = np.floor_divide(a + hi_tol, tolerance)
    b3 = np.floor_divide(b + hi_tol, tolerance)

    # do a normal intersect on the quantized data for different comparisons
    (_, ia1, ib1) = np.intersect1d(a1, b1, assume_unique=assume_unique, return_indices=True)
    (_, ia2, ib2) = np.intersect1d(a1, b2, assume_unique=assume_unique, return_indices=True)
    (_, ia3, ib3) = np.intersect1d(a1, b3, assume_unique=assume_unique, return_indices=True)
    (_, ia4, ib4) = np.intersect1d(a2, b1, assume_unique=assume_unique, return_indices=True)
    (_, ia5, ib5) = np.intersect1d(a3, b1, assume_unique=assume_unique, return_indices=True)

    # combine the results
    ia = reduce(np.union1d, [ia1, ia2, ia3, ia4, ia5])
    ib = reduce(np.union1d, [ib1, ib2, ib3, ib4, ib5])

    # calculate output
    # Note that a[ia] and b[ib] should be the same with a tolerance of 0, but not necessarily otherwise
    # This function returns the values from the first vector a
    c = np.sort(a[ia])
    if np.any(is_dates):
        c = c.astype(orig_datetime)
    if return_indices:
        return (c, ia, ib)
    return c

#%% Unit test
if __name__ == '__main__':
    plt.ioff()
    unittest.main(module='dstauffman.tests.test_stats', exit=False)
    doctest.testmod(verbose=False)
