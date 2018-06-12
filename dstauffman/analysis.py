# -*- coding: utf-8 -*-
r"""
Contains more complex analysis related routines.

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

from dstauffman.plotting import Opts, setup_plots

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
    setup_plots(fig, opts, 'dist_no_yscale')
    return fig

#%% Unit test
if __name__ == '__main__':
    plt.ioff()
    unittest.main(module='tests.test_analysis', exit=False)
    doctest.testmod(verbose=False)
