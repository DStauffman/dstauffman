r"""
Defines useful plotting utilities related to health policy.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
#.  Separated into plot_health.py from plotting.py by David C. Stauffer in May 2020.
#.  Moved to consolidated plotting submodule by David C. Stauffer in July 2020.
"""

#%% Imports
import doctest
import unittest
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter

from dstauffman import get_factors, rms
from dstauffman.health import bins_to_str_ranges

from dstauffman.plotting.plotting import Opts
from dstauffman.plotting.support  import ColorMap, DEFAULT_COLORMAP, ignore_plot_data, \
    plot_second_units_wrapper, disp_xlimits, setup_plots, show_zero_ylim, whitten, z_from_ci

#%% Functions - plot_health_time_history
def plot_health_time_history(time, data, label, units='', opts=None, *, legend=None, \
        second_yscale=None, ignore_empties=False, data_lo=None, data_hi=None, colormap=None):
    r"""
    Plot multiple metrics over time.

    Parameters
    ----------
    time : 1D ndarray
        time history
    data : 1D, 2D or 3D ndarray
        data for corresponding time history, time is first dimension, last dimension is bin
        middle dimension if 3D is the cycle
    label : str
        Name to label on the plots
    units : str, optional
        units of the data to be displayed on the plot
    opts : class Opts, optional
        plotting options
    legend : list of str, optional
        Names to use for each channel of data
    second_yscale : float or dict, optional
        Multiplication scale factor to use to display on a secondary Y axis
    ignore_empties : bool, optional
        Removes any entries from the plot and legend that contain only zeros or only NaNs
    data_lo : same as data
        Lower confidence bound on data, plot if not None
    data_hi : same as data
        Upper confidence bound on data, plot if not None
    colormap : str or matplotlib.colors.Colormap, optional
        Name of colormap to use, if specified, overrides the opts.colormap

    Returns
    -------
    fig : object
        figure handle, if None, no figure was created

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.
    #.  Updated by David C. Stauffer in October 2017 to do comparsions of multiple runs.

    Examples
    --------
    >>> from dstauffman.plotting import plot_health_time_history
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> description = 'Random Data'
    >>> time  = np.arange(0, 5, 1./12) + 2000
    >>> data  = np.random.rand(len(time), 5).cumsum(axis=1)
    >>> data  = 10 * data / np.expand_dims(data[:, -1], axis=1)
    >>> fig   = plot_health_time_history(time, data, description)

    Close plot
    >>> plt.close(fig)

    """
    # force inputs to be ndarrays
    time = np.atleast_1d(np.asanyarray(time))
    data = np.asanyarray(data)

    # check optional inputs
    if opts is None:
        opts = Opts()
    if colormap is None:
        if opts.colormap is None:
            colormap = DEFAULT_COLORMAP
        else:
            colormap = opts.colormap
    legend_loc = opts.leg_spot
    show_zero  = opts.show_zero
    time_units = opts.time_base
    unit_text  = ' [' + units + ']' if units else ''
    (scale, prefix) = get_factors(opts.vert_fact)

    # check for valid data
    if ignore_plot_data(data, ignore_empties):
        print(' ' + label + ' plot skipped due to missing data.')
        return None
    assert time.ndim == 1, 'Time must be a 1D array.'

    # ensure that data is at least 2D
    if data.ndim == 0:
        data = np.atleast_2d(data)
    elif data.ndim == 1:
        data = data[:, np.newaxis] # forces to grow in second dimension, instead of first

    # get shape information
    if data.ndim == 2:
        normal    = True
        num_loops = 1
        num_bins  = data.shape[1]
        names     = ['']
    elif data.ndim == 3:
        assert len(opts.names) == data.shape[1], 'Names must match the number of channels is the 3rd axis of data.'
        normal    = False
        num_loops = data.shape[1]
        num_bins  = data.shape[2]
        names     = opts.names
    else:
        assert False, 'Data must be 0D to 3D array.'
    assert time.shape[0] == data.shape[0], 'Time and data must be the same length. Current time.shape={} and data.shape={}'.format(time.shape, data.shape)
    if legend is not None:
        assert len(legend) == num_bins, 'Number of data channels does not match the legend.'
    else:
        legend = ['Channel {}'.format(i+1) for i in range(num_bins)]

    # process other inputs
    this_title = label + ' vs. Time'

    # get colormap based on high and low limits
    cm = ColorMap(colormap, num_colors=num_bins)

    # plot data
    fig = plt.figure()
    fig.canvas.set_window_title(this_title)
    ax = fig.add_subplot(111)
    cm.set_colors(ax)
    for i in range(num_bins):
        for j in range(num_loops):
            this_name = names[j] + ' - ' if names[j] else ''
            if normal:
                this_data    = data[:, i]
                this_data_lo = data_lo[:, i] if data_lo is not None else None
                this_data_hi = data_hi[:, i] if data_hi is not None else None
            else:
                this_data    = data[:, j, i]
                this_data_lo = data_lo[:, j, i] if data_lo is not None else None
                this_data_hi = data_hi[:, j, i] if data_hi is not None else None
            if not ignore_plot_data(this_data, ignore_empties):
                color_dt = j * 0.5 / num_loops
                if j // 2:
                    this_color = whitten(cm.get_color(i), white=(0, 0, 0, 1), dt=color_dt)
                else:
                    this_color = whitten(cm.get_color(i), white=(1, 1, 1, 1), dt=color_dt)
                ax.plot(time, scale*this_data, '.-', label=this_name + legend[i], color=this_color, zorder=10)
                if this_data_lo is not None:
                    ax.plot(time, scale*this_data_lo, 'o:', markersize=2, label='', color=whitten(cm.get_color(i)), zorder=6)
                if this_data_hi is not None:
                    ax.plot(time, scale*this_data_hi, 'o:', markersize=2, label='', color=whitten(cm.get_color(i)), zorder=6)

    # add labels and legends
    ax.set_xlabel('Time [' + time_units + ']')
    ax.set_ylabel(label + unit_text)
    ax.set_title(this_title)
    ax.legend(loc=legend_loc)
    ax.grid(True)
    # set the display period
    disp_xlimits(ax, xmin=opts.disp_xmin, xmax=opts.disp_xmax)
    # optionally force zero to be plotted
    if show_zero:
        show_zero_ylim(ax)
    # set years to always be whole numbers on the ticks
    if time_units == 'year' and (np.max(time) - np.min(time)) >= 4:
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))

    # optionally add second Y axis
    plot_second_units_wrapper(ax, second_yscale)

    # setup plots
    setup_plots(fig, opts)
    return fig

#%% Functions - plot_health_monte_carlo
def plot_health_monte_carlo(time, data, label, units='', opts=None, *, plot_indiv=True, \
    truth=None, plot_as_diffs=False, second_yscale=None, plot_sigmas=1, \
    plot_confidence=0, colormap=None):
    r"""
    Plot the given data channel versus time, with a generic label argument.

    Parameters
    ----------
    time : array_like
        time history
    data : array_like
        data for corresponding time history
    label : str
        generic text to put on the plot title and figure name
    units : str, optional
        units of the data to be displayed on the plot
    opts : class Opts, optional
        plotting options
    plot_indiv : bool, optional
        Plot the individual cycles, default is true
    truth : TruthPlotter, optional
        Truth instance for adding to the plot
    plot_as_diffs : bool, optional, default is False
        Plot each entry in results against the other ones, default is False
    second_yscale : float or dict, optional
        Multiplication scale factor to use to display on a secondary Y axis
    plot_sigmas : numeric, optional
        If value converts to true as bool, then plot the sigma values of the given value
    plot_confidence : numeric, optional
        If value converts to true as bool, then plot the confidence intervals of the given value
    colormap : str or matplotlib.colors.Colormap, optional
        Name of colormap to use, if specified, overrides the opts.colormap

    Returns
    -------
    fig : object
        figure handle

    Notes
    -----
    #.  Written by David C. Stauffer in March 2015.
    #.  Updated by David C. Stauffer in December 2015 to include an optional secondary Y axis.
    #.  Updated by David C. Stauffer in October 2016 to use the new TruthPlotter class.
    #.  Updated by David C. Stauffer in June 2017 to put some basic stuff in Opts instead of kwargs.
    #.  Updated by David C. Stauffer in October 2017 to include one sigma and 95% confidence intervals.
    #.  If ndim == 2, then dimension 0 is time and dimension 1 is the number of runs.

    Examples
    --------
    >>> from dstauffman.plotting import plot_health_monte_carlo
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> time  = np.arange(0, 10, 0.1)
    >>> data  = np.sin(time)
    >>> label = 'Sin'
    >>> units = 'population'
    >>> fig   = plot_health_monte_carlo(time, data, label, units)

    Close plot
    >>> plt.close(fig)

    """
    # force inputs to be ndarrays
    time = np.asanyarray(time)
    data = np.asanyarray(data)

    # check optional inputs
    if opts is None:
        opts = Opts()
    if colormap is None:
        if opts.colormap is None:
            colormap = DEFAULT_COLORMAP
        else:
            colormap = opts.colormap
    rms_in_legend = opts.show_rms
    legend_loc    = opts.leg_spot
    show_zero     = opts.show_zero
    time_units    = opts.time_base
    show_legend   = rms_in_legend or plot_as_diffs or (truth is not None and not truth.is_null)
    unit_text     = ' [' + units + ']' if units else ''
    (scale, prefix) = get_factors(opts.vert_fact)

    # ensure that data is at least 2D
    if data.ndim == 0:
        data = np.atleast_2d(data)
    elif data.ndim == 1:
        data = data[:, np.newaxis] # forces to grow in second dimension, instead of first
    elif data.ndim == 2:
        pass
    else:
        raise ValueError('Unexpected number of dimensions in data. Monte carlo can currently only ' +
                         'handle single channels with multiple runs. Split the channels apart into' +
                         'separate plots if desired.')

    # get number of different series
    num_series = data.shape[1]

    if plot_as_diffs:
        # build colormap
        cm = ColorMap(colormap, num_colors=data.shape[1])
        # calculate RMS
        if rms_in_legend:
            rms_data = np.atleast_1d(rms(scale*data, axis=0, ignore_nans=True))
    else:
        # calculate the mean and std of data, while disabling warnings for time points that are all NaNs
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            warnings.filterwarnings('ignore', message='Degrees of freedom <= 0 for slice.')
            mean = np.nanmean(data, axis=1)
            std  = np.nanstd(data, axis=1)

        # calculate an RMS
        if rms_in_legend:
            rms_data = rms(scale*mean, axis=0, ignore_nans=True)

    # alias the title
    this_title = label + ' vs. Time'
    # create the figure and set the title
    fig = plt.figure()
    fig.canvas.set_window_title(this_title)
    # add an axis and plot the data
    ax = fig.add_subplot(111)
    if plot_as_diffs:
        cm.set_colors(ax)
        for ix in range(data.shape[1]):
            this_label = opts.get_names(ix)
            if not this_label:
                this_label = 'Series {}'.format(ix+1)
            if rms_in_legend:
                this_label += ' (RMS: {:.2f})'.format(rms_data[ix])
            ax.plot(time, scale*data[:, ix], '.-', linewidth=2, zorder=10, label=this_label)
    else:
        this_label = opts.get_names(0) + label
        if rms_in_legend:
            this_label += ' (RMS: {:.2f})'.format(rms_data)
        ax.plot(time, scale*mean, '.-', linewidth=2, color='#0000cd', zorder=10, label=this_label)
        if plot_sigmas and num_series > 1:
            sigma_label = r'$\pm {}\sigma$'.format(plot_sigmas)
            ax.plot(time, scale*mean + plot_sigmas*scale*std, '.-', markersize=2, color='#20b2aa', zorder=6, label=sigma_label)
            ax.plot(time, scale*mean - plot_sigmas*scale*std, '.-', markersize=2, color='#20b2aa', zorder=6)
        if plot_confidence and num_series > 1:
            conf_label = '{}% C.I.'.format(100*plot_confidence)
            conf_z = z_from_ci(plot_confidence)
            conf_std = conf_z * std / np.sqrt(num_series)
            ax.plot(time, scale*mean + scale*conf_std, '.-', markersize=2, color='#2e8b57', zorder=7, label=conf_label)
            ax.plot(time, scale*mean - scale*conf_std, '.-', markersize=2, color='#2e8b57', zorder=7)
        # inidividual line plots
        if plot_indiv and data.ndim > 1:
            for ix in range(num_series):
                ax.plot(time, scale*data[:, ix], color='0.75', zorder=1)
    # optionally plot truth (without changing the axis limits)
    if truth is not None:
        truth.plot_truth(ax, scale)
    # add labels and legends
    ax.set_xlabel('Time [' + time_units + ']')
    ax.set_ylabel(label + unit_text)
    ax.set_title(this_title)
    if show_legend:
        ax.legend(loc=legend_loc)
    # show a grid
    ax.grid(True)
    # optionally force zero to be plotted
    if show_zero and min(ax.get_ylim()) > 0:
        ax.set_ylim(bottom=0)
    # set years to always be whole numbers on the ticks
    if time_units == 'year' and np.any(time) and (np.max(time) - np.min(time)) >= 4:
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    # optionally add second Y axis
    plot_second_units_wrapper(ax, second_yscale)
    # Setup plots
    setup_plots(fig, opts)
    return fig

#%% Functions - plot_icer
def plot_icer(qaly, cost, ix_front, baseline=None, names=None, opts=None):
    r"""Plot the icer results."""
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
    >>> from dstauffman.plotting import plot_population_pyramid
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
    unittest.main(module='dstauffman.tests.test_plotting_health', exit=False)
    doctest.testmod(verbose=False)