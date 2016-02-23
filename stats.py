# -*- coding: utf-8 -*-
r"""
Stats module file for the "dstauffman" library.  It contains generic statistics related routines
that can be independently defined and used by other modules.

Notes
-----
#.  Written by David C. Stauffer in December 2015.
"""
# pylint: disable=E1101, C0301, C0103

#%% Imports
import doctest
import numpy as np
import unittest
from dstauffman.constants import MONTHS_PER_YEAR

#%% Functions - convert_annual_to_monthly_probability
def convert_annual_to_monthly_probability(annual):
    r"""
    Converts a given annual probabily into the equivalent monthly one.

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
    [ 0. 0.00874161  1. ]

    """
    # check ranges
    if np.any(annual < 0):
        raise ValueError('annual must be >= 0')
    if np.any(annual > 1):
        raise ValueError('annual must be <= 1')
    # ignore divide by zero errors when annual == 1
    with np.errstate(divide='ignore'):
        # convert to equivalent probability and return result
        monthly = 1-np.exp(np.log(1-annual)/MONTHS_PER_YEAR)
    return monthly

#%% Functions - convert_monthly_to_annual_probability
def convert_monthly_to_annual_probability(monthly):
    r"""
    Converts a given monthly probability into the equivalent annual one.

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
    [ 0. 0.71757046 1. ]

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
    Converts a given probability and time to a rate.

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
    [-0. 0.03512017 inf]

    """
    # check ranges
    if np.any(prob < 0):
        raise ValueError('Probability must be >= 0')
    if np.any(prob > 1):
        raise ValueError('Probability must be <= 1')
    # ignore log of zero errors when prob == 1
    with np.errstate(divide='ignore'):
        # calculate rate
        rate = -np.log(1 - prob) / time
    return rate

#%% Functions - rate_to_prob
def rate_to_prob(rate, time=1):
    r"""
    Converts a given rate and time to a probability.

    Parameters
    ----------
    rate : numpy.ndarray
        Equivalent annual rate for the given time
    time : float
        Time period for the desired probability to be calculated from, in years

    Returns
    -------
    prob : numpy.ndarray
        Probability of event happening over the given time

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
    [ 0. 0.00829871 0.07995559 0.99975963 1. ]

    """
    # check ranges
    if np.any(rate < 0):
        raise ValueError('Rate must be >= 0')
    # calculate probability
    prob = 1 - np.exp(-rate * time)
    return prob

#%% Functions - month_prob_mult_ratio
def month_prob_mult_ratio(prob, ratio):
    r"""
    Multiplies a monthly probability by a given risk or hazard ratio.

    Parameters
    ----------
    prob : numpy.ndarray
        Probability of event happening over one month
    ratio : float
        Multiplication ratio to apply to probability

    Returns
    -------
    mult_prob : numpy.ndarray
        Equivalent multiplicative monthly probability

    Notes
    -----
    #.  Written by David C. Staufer in January 2016.

    Examples
    --------

    >>> from dstauffman import month_prob_mult_ratio
    >>> import numpy as np
    >>> prob = np.array([0, 0.1, 1])
    >>> ratio = 2
    >>> mult_prob = month_prob_mult_ratio(prob, ratio)
    >>> print(mult_prob) # doctest: +NORMALIZE_WHITESPACE
    [ 0. 0.19 1. ]

    >>> ratio = 0.5
    >>> mult_prob = month_prob_mult_ratio(prob, ratio)
    >>> print(mult_prob) # doctest: +NORMALIZE_WHITESPACE
    [ 0. 0.0513167 1. ]

    """
    # convert the probability to a rate
    rate = prob_to_rate(prob, time=1./MONTHS_PER_YEAR)
    # scale the rate
    mult_rate = rate * ratio
    # convert back to a probability
    mult_prob = rate_to_prob(mult_rate, time=1./MONTHS_PER_YEAR)
    return mult_prob

#%% Functions - annual_rate_to_monthly_probability
def annual_rate_to_monthly_probability(rate):
    r"""
    Converts a given annual rate to a monthly probability.

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
    [ 0. 0.04081054 0.07995559 0.34075937 1. ]

    """
    # divide rate and calculate probability
    prob = rate_to_prob(rate/MONTHS_PER_YEAR)
    return prob

#%% Functions - ar2mp
ar2mp = annual_rate_to_monthly_probability

#%% Functions - combine_sets
def combine_sets(n1, u1, s1, n2, u2, s2):
    r"""
    Combines the mean and standard deviations for two non-overlapping sets of data.

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
    0.59135639081

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

#%% Functions - icer
def icer(cost, qaly):
    r"""
    Calculates the incremental cost effectiveness ratios with steps to throw out dominated strategies.

    Summary
    -------
        In a loop, the code sorts by cost, throws out strongly dominated strategies (qaly doesn't
        improve despite higher costs), calculates an incremental cost, qaly and cost effectiveness
        ratio, then throws out weakly dominated strategies (icer doesn't improve over cheaper options)
        and finally returns the incremental cost, qaly and ratios for the remaining "frontier" options
        along with an order variable to map them back to the inputs.

    Parameters
    ----------
    cost : (N) array_like
        Cost of each strategy
    qaly : (N) array_like
        Quality adjusted life years (QALY) gained by each strategy

    Results
    -------
    inc_cost : (M) ndarray
        incremental costs - see note 1
    inc_qaly : (M) ndarray
        incremental QALYs gained
    icer_out : (M) ndarray
        incremental cost effectiveness ratios
    order    : (N) ndarray
        order mapping to the original inputs, with NaNs for dominated strategies

    Notes
    -----
    #.  N may be smaller than M due to dominated strategies being removed.  The order variable
            will have (M - N) values set to NaN.

    Examples
    --------

    >>> from dstauffman import icer
    >>> cost = [250e3, 750e3, 2.25e6, 3.75e6]
    >>> qaly = [20., 30, 40, 80]
    >>> (inc_cost, inc_qaly, icer_out, order) = icer(cost, qaly)
    >>> print(inc_cost) # doctest: +NORMALIZE_WHITESPACE
    [ 250000. 500000. 3000000.]

    >>> print(inc_qaly) # doctest: +NORMALIZE_WHITESPACE
    [ 20. 10. 50.]

    >>> print(icer_out) # doctest: +NORMALIZE_WHITESPACE
    [ 12500. 50000. 60000.]

    >>> print(order) # doctest: +NORMALIZE_WHITESPACE
    [ 0. 1. nan 2.]

    """
    # force inputs to be ndarrays
    cost = np.asarray(cost)
    qaly = np.asarray(qaly)

    # check inputs
    assert np.all(cost > 0), 'Costs must be positive.'
    assert np.all(qaly > 0), 'Qalys must be positive.'
    assert cost.shape == qaly.shape, 'Cost and Qalys must have same size.'
    assert cost.size > 0, 'Costs and Qalys cannot be empty.'

    # build an index order variable to keep track of strategies
    keep = list(range(cost.size))

    # deal with garbage 0D arrays so that they can be indexed by keep
    if cost.ndim == 0:
        cost = cost[np.newaxis]
    if qaly.ndim == 0:
        qaly = qaly[np.newaxis]

    # enter processing loop
    while True:
        # pull out current values based on evolving order mask
        this_cost = cost[keep]
        this_qaly = qaly[keep]

        # sort by cost
        ix_sort     = np.argsort(this_cost)
        sorted_cost = this_cost[ix_sort]
        sorted_qaly = this_qaly[ix_sort]

        # check for strongly dominated strategies
        if not np.all(np.diff(sorted_qaly) >= 0):
            # find the first occurence (increment by one to find the one less effective than the last)
            bad = np.nonzero(np.diff(sorted_qaly) < 0)[0] + 1
            if len(bad) == 0:
                raise ValueError('Index should never be empty, something unexpected happended.')
            # update the mask and continue to next pass of while loop
            keep.pop(ix_sort[bad[0]])
            continue

        # calculate incremental costs
        inc_cost = np.hstack((sorted_cost[0], np.diff(sorted_cost)))
        inc_qaly = np.hstack((sorted_qaly[0], np.diff(sorted_qaly)))
        icer_out = inc_cost / inc_qaly

        # check for weakly dominated strategies
        if not np.all(np.diff(icer_out) >= 0):
            # find the first bad occurence
            bad = np.nonzero(np.diff(icer_out) < 0)[0]
            if len(bad) == 0:
                raise ValueError('Index should never be empty, something unexpected happended.')
            # update mask and continue to next pass
            keep.pop(ix_sort[bad[0]])
            continue

        # if no continue statements were reached, then another iteration is not necessary, so break out
        break

    # save the final ordering
    order = np.nan * np.ones(cost.shape)
    order[keep] = ix_sort

    return (inc_cost, inc_qaly, icer_out, order)

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='tests.test_stats', exit=False)
    doctest.testmod(verbose=False)
