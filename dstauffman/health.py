# -*- coding: utf-8 -*-
r"""
Contains more complex analysis related routines mostly specific to health care modeling.

Notes
-----
#.  Written by David C. Stauffer in October 2017.

"""

#%% Imports
import doctest
import unittest
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dstauffman.latex import bins_to_str_ranges
from dstauffman.plotting import Opts, setup_plots

#%% Functions - dist_enum_and_mons
def dist_enum_and_mons(num, distribution, prng, *, max_months=None, start_num=1, alpha=1, beta=1):
    r"""
    Create a distribution for an enumerated state with a duration (such as a disease status).

    Parameters
    ----------
    num : int
        Number of people in the population
    distribution : array_like
        Likelihood of being in each state (should cumsum to 100%)
    prng : class numpy.random.RandomState
        Pseudo-random number generator
    max_months : scalar or array_like, optional
        Maximum number of months for being in each state
    start_num : int, optional
        Number to start counting from, default is 1
    alpha : int, optional
        The alpha parameter for the beta distribution
    beta : int, optional
        The beta parameter for the beta distribution

    Returns
    -------
    state : ndarray
        Enumerated status for this month for everyone in the population
    mons : ndarray
        Number of months in this state for anyone with an infection

    Notes
    -----
    #.  Written by David C. Stauffer in April 2015.
    #.  Updated by David C. Stauffer in June 2015 to use a beta curve to distribute the number of
        months spent in each state.
    #.  Made into a generic function for the dstauffman library by David C. Stauffer in July 2015.
    #.  Updated by David C. Stauffer in November 2015 to change the inputs to allow max_months and
        mons output to be optional.
    #.  Updated by David C. Stauffer in April 2017 to only return state if desired, and to allow
        distribution to be a 2D matrix, so you can have age based distributions.

    Examples
    --------
    >>> from dstauffman import dist_enum_and_mons
    >>> import numpy as np
    >>> num = 100
    >>> distribution = np.array([0.10, 0.20, 0.30, 0.40])
    >>> max_months = np.array([5, 100, 20, 1])
    >>> start_num = 0
    >>> prng = np.random.RandomState()
    >>> (state, mons) = dist_enum_and_mons(num, distribution, prng, max_months=max_months, start_num=start_num)

    """
    # hard-coded values
    precision = 1e-12
    # create the cumulative distribution (allows different distribution per person if desired)
    cum_dist = np.cumsum(np.atleast_2d(distribution), axis=1)
    assert np.all(np.abs(cum_dist[:,-1] - 1) < precision), "Given distribution doesn't sum to 1."
    # do a random draw based on the cumulative distribution
    state = np.sum(prng.rand(num, 1) >= cum_dist, axis=1, dtype=int) + start_num
    # set the number of months in this state based on a beta distribution with the given
    # maximum number of months in each state
    if max_months is None:
        return state
    else:
        if np.isscalar(max_months):
            max_months = np.full(len(distribution), max_months)
        mons = np.ceil(max_months[state-start_num] * prng.beta(alpha, beta, num)).astype(int)
        return (state, mons)

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
    [20. 10. 50.]

    >>> print(icer_out) # doctest: +NORMALIZE_WHITESPACE
    [12500. 50000. 60000.]

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
        fig = plot_icer(qaly, cost, ix, baseline=baseline, names=names, opts=opts)

    return (inc_cost, inc_qaly, icer_out, order, icer_data, fig)

#%% Functions - plot_icer
def plot_icer(qaly, cost, ix_front, baseline=None, names=None, opts=None):
    # check optional inputs
    if opts is None:
        opts = Opts()
    # create a figure and axis
    fig = plt.figure()
    fig.canvas.set_window_title('Cost Benefit Frontier')
    ax = fig.add_subplot(111)
    # plot the data
    ax.plot(qaly, cost, 'ko', label='strategies')
    ax.plot(qaly[ix_front], cost[ix_front], 'r.', markersize=20, label='frontier')
    # get axis limits before (0,0) point is added
    lim = ax.axis()
    # add ICER lines
    if baseline is None:
        ax.plot(np.hstack((0, qaly[ix_front])), np.hstack((0, cost[ix_front])), 'r-', label='ICERs')
    else:
        ax.plot(np.hstack((0, qaly[ix_front[0]])), np.hstack((0, cost[ix_front[0]])), 'r:')
        ax.plot(np.hstack((qaly[baseline], qaly[ix_front])), np.hstack((cost[baseline], cost[ix_front])), 'r-', label='ICERs')
    # Label each point
    dy = (lim[3] - lim[2]) / 100
    for i in range(cost.size):
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
    setup_plots(fig, opts)
    return fig

#%% Functions - plot_population_pyramid
def plot_population_pyramid(age_bins, male_per, fmal_per, title='Population Pyramid', *, opts=None, \
        name1='Male', name2='Female', color1='xkcd:blue', color2='xkcd:red'):
    r"""
    Plot the standard population pyramid.

    Parameters
    ----------
    age_bins : (N+1,) array_like of float/ints
        Age boundaries to plot
    male_per : (N,) array_like of int
        Male population percentage in each bin
    fmal_per : (N,) array_like of int
        Female population percentage in each bin
    title : str, optional, default is 'Population Pyramid'
        Title for the plot
    opts : class Opts, optional
        Plotting options
    name1 : str, optional
        Name for data source 1
    name2 : str, optional
        Name for data source 2
    color1 : str or valid color tuple, optional
        Color for data source 1
    color2 : str or valid color tuple, optional
        Color for data source 2

    Returns
    -------
    fig : object
        figure handle

    Notes
    -----
    #.  Written by David C. Stauffer in April 2017.

    References
    ----------
    .. [1]  https://en.wikipedia.org/wiki/Population_pyramid

    Examples
    --------
    >>> from dstauffman import plot_population_pyramid
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> age_bins = np.array([  0,   5,  10,  15,  20, 1000], dtype=int)
    >>> male_per = np.array([500, 400, 300, 200, 100]) / 3000
    >>> fmal_per = np.array([450, 375, 325, 225, 125]) / 3000
    >>> fig      = plot_population_pyramid(age_bins, male_per, fmal_per)

    Close figure
    >>> plt.close(fig)

    """
    # hard-coded values
    scale = 100

    # check optional inputs
    if opts is None:
        opts = Opts()
    legend_loc = opts.leg_spot

    # convert data to percentages
    num_pts   = age_bins.size - 1
    y_values  = np.arange(num_pts)
    y_labels  = bins_to_str_ranges(age_bins, dt=1, cutoff=200)

    # create the figure and axis and set the title
    fig = plt.figure()
    fig.canvas.set_window_title(title)
    ax = fig.add_subplot(111)

    # plot bars
    ax.barh(y_values, -scale*male_per, 0.95, color=color1, label=name1)
    ax.barh(y_values,  scale*fmal_per, 0.95, color=color2, label=name2)

    # make sure plot is symmetric about zero
    xlim = max(abs(x) for x in ax.get_xlim())
    ax.set_xlim(-xlim, xlim)

    # add labels
    ax.set_xlabel('Population [%]')
    ax.set_ylabel('Age [years]')
    ax.set_title(title)
    ax.set_yticks(y_values)
    ax.set_yticklabels(y_labels)
    ax.set_xticklabels(np.abs(ax.get_xticks()))
    ax.legend(loc=legend_loc)

    # Setup plots
    setup_plots(fig, opts)

    return fig

#%% Unit test
if __name__ == '__main__':
    plt.ioff()
    unittest.main(module='dstauffman.tests.test_health', exit=False)
    doctest.testmod(verbose=False)