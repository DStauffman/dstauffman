# -*- coding: utf-8 -*-
r"""
Contains statistics related routines that can be independently defined and used by other modules.

Notes
-----
#.  Written by David C. Stauffer in December 2015.

"""

#%% Imports
import doctest
import unittest
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

from dstauffman.constants import MONTHS_PER_YEAR
from dstauffman.plotting import Opts, setup_plots

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
    [ 0. 0.00874161  1. ]

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
    [ 0. 0.03512017 inf]

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
    Multiply a monthly probability by a given risk or hazard ratio.

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
    [ 0. 0.04081054 0.07995559 0.34075937 1. ]

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
def icer(cost, qaly, names=None, baseline=None, make_plot=False, opts=None):
    r"""
    Calculate the incremental cost effectiveness ratios with steps to throw out dominated strategies.

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
    names : (N) array_like, optional
        Names of the different strategies
    baseline : int, optional
        Index of baseline strategy to use for cost comparisons, if not nan
    make_plot : bool, optional
        True/false flag for whether to plot the data
    opts : class Opts, optional
        Plotting options

    Results
    -------
    inc_cost  : (M) ndarray
        incremental costs - see note 1
    inc_qaly  : (M) ndarray
        incremental QALYs gained
    icer_out  : (M) ndarray
        incremental cost effectiveness ratios
    order     : (N) ndarray
        order mapping to the original inputs, with NaNs for dominated strategies
    icer_data : (N) pandas dataframe
        ICER data as a pandas dataframe
    fig       : (object) figure handle or None
        Figure handle for any figure that was produced

    Notes
    -----
    #.  N may be smaller than M due to dominated strategies being removed.  The order variable
            will have (M - N) values set to NaN.

    Examples
    --------
    >>> from dstauffman import icer
    >>> cost = [250e3, 750e3, 2.25e6, 3.75e6]
    >>> qaly = [20., 30, 40, 80]
    >>> (inc_cost, inc_qaly, icer_out, order, icer_data, fig) = icer(cost, qaly)
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
    cost = np.atleast_1d(np.asarray(cost))
    qaly = np.atleast_1d(np.asarray(qaly))
    fig  = None

    # check inputs
    assert np.all(cost > 0), 'Costs must be positive.'
    assert np.all(qaly > 0), 'Qalys must be positive.'
    assert cost.shape == qaly.shape, 'Cost and Qalys must have same size.'
    assert cost.size > 0, 'Costs and Qalys cannot be empty.'

    # alias the number of strategies
    num = cost.size

    # build an index order variable to keep track of strategies
    keep = list(range(num))

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
            bad = np.flatnonzero(np.diff(sorted_qaly) < 0) + 1
            if len(bad) == 0:
                raise ValueError('Index should never be empty, something unexpected happended.') # pragma: no cover
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
            bad = np.flatnonzero(np.diff(icer_out) < 0)
            if len(bad) == 0:
                raise ValueError('Index should never be empty, something unexpected happended.') # pragma: no cover
            # update mask and continue to next pass
            keep.pop(ix_sort[bad[0]])
            continue

        # if no continue statements were reached, then another iteration is not necessary, so break out
        break

    # save the final ordering
    order = np.full(cost.shape, np.nan, dtype=float)
    order[keep] = ix_sort

    # build an index to pull data out
    temp = np.flatnonzero(~np.isnan(order))
    ix   = temp[order[~np.isnan(order)].astype(int)]

    # recalculate based on given baseline
    if baseline is not None:
        inc_cost = np.diff(np.hstack((cost[baseline], cost[ix])))
        inc_qaly = np.diff(np.hstack((qaly[baseline], qaly[ix])))
        icer_out = inc_cost / inc_qaly

    # output as dataframe
    # build a name list if not given
    if names is None:
        names = ['Strategy {}'.format(i+1) for i in range(num)]
    # preallocate some variables
    full_inc_costs     = np.full((num), np.nan, dtype=float)
    full_inc_qalys     = np.full((num), np.nan, dtype=float)
    full_icers         = np.full((num), np.nan, dtype=float)
    # fill the calculations in where applicable
    full_inc_costs[ix] = inc_cost
    full_inc_qalys[ix] = inc_qaly
    full_icers[ix]     = icer_out
    # make into dictionary with more explicit column names
    data = OrderedDict()
    data['Strategy'] = names
    data['Cost'] = cost
    data['QALYs'] = qaly
    data['Increment_Costs'] = full_inc_costs
    data['Incremental_QALYs'] = full_inc_qalys
    data['ICER'] = full_icers
    data['Order'] = order

    # make the whole data set into a dataframe
    icer_data = pd.DataFrame.from_dict(data)
    icer_data.set_index('Strategy', inplace=True)

    # Make a plot
    if make_plot:
        # check optional inputs
        if opts is None:
            opts = Opts()
        # create a figure and axis
        fig = plt.figure()
        fig.canvas.set_window_title('Cost Benefit Frontier')
        ax = fig.add_subplot(111)
        # plot the data
        ax.plot(qaly, cost, 'ko', label='strategies')
        ax.plot(qaly[ix], cost[ix], 'r.', markersize=20, label='frontier')
        # get axis limits before (0,0) point is added
        lim = ax.axis()
        # add ICER lines
        if baseline is None:
            ax.plot(np.hstack((0, qaly[ix])), np.hstack((0, cost[ix])), 'r-', label='ICERs')
        else:
            ax.plot(np.hstack((0, qaly[ix[0]])), np.hstack((0, cost[ix[0]])), 'r:')
            ax.plot(np.hstack((qaly[baseline], qaly[ix])), np.hstack((cost[baseline], cost[ix])), 'r-', label='ICERs')
        # Label each point
        dy = (lim[3] - lim[2]) / 100
        for i in range(num):
            ax.annotate(names[i], xy=(qaly[i], cost[i]+dy), xycoords='data', horizontalalignment='center', \
                verticalalignment='bottom', fontsize=12)
        # add some labels and such
        ax.set_title(fig.canvas.get_window_title())
        ax.set_xlabel('Benefits')
        ax.set_ylabel('Costs')
        ax.legend(loc='upper left')
        ax.grid(True)
        # reset limits with including (0,0) point in case it skews everything too much
        ax.axis(lim)
        # add standard plotting features
        setup_plots(fig, opts, 'dist_no_yscale')

    return (inc_cost, inc_qaly, icer_out, order, icer_data, fig)

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

#%% Unit test
if __name__ == '__main__':
    plt.ioff()
    unittest.main(module='tests.test_stats', exit=False)
    doctest.testmod(verbose=False)
