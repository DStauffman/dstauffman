# -*- coding: utf-8 -*-
r"""
Defines useful plotting utilities.

Notes
-----
#.  Written by David C. Stauffer in March 2015.

"""

#%% Imports
# normal imports
import doctest
import os
import unittest
import warnings

# plotting/numpy imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import StrMethodFormatter

# model imports
from dstauffman.classes import Frozen
from dstauffman.constants import DEFAULT_COLORMAP
from dstauffman.latex import bins_to_str_ranges
from dstauffman.plot_support import ColorMap, get_color_lists, ignore_plot_data, \
                                        setup_plots, TruthPlotter, whitten
from dstauffman.quat import quat_angle_diff
from dstauffman.stats import z_from_ci
from dstauffman.units import get_factors
from dstauffman.utils import pprint_dict, rms

#%% Classes - Opts
class Opts(Frozen):
    r"""Optional plotting configurations."""
    def __init__(self):
        self.case_name  = ''
        self.save_path  = os.getcwd()
        self.save_plot  = False
        self.plot_type  = 'png'
        self.sub_plots  = True
        self.show_plot  = True
        self.show_link  = False
        self.disp_xmin  = -np.inf
        self.disp_xmax  =  np.inf
        self.rms_xmin   = -np.inf
        self.rms_xmax   =  np.inf
        self.vert_fact  = 'unity'
        self.colormap   = None
        self.show_rms   = True
        self.show_zero  = False
        self.base_time  = 'year'
        self.legend_loc = 'best'
        self.names      = list()

    def get_names(self, ix):
        r"""Get the specified name from the list."""
        if hasattr(self, 'names') and len(self.names) >= ix+1:
            name = self.names[ix]
        else:
            name = ''
        return name

    def pprint(self, indent=1, align=True):
        r"""Display a pretty print version of the class."""
        pprint_dict(self.__dict__, name=self.__class__.__name__, indent=indent, align=align)

#%% Functions - plot_time_history
def plot_time_history(time, data, label, units='unity', opts=None, *, legend=None, \
        second_y_scale=None, ignore_empties=False, data_lo=None, data_hi=None, colormap=None):
    r"""
    Plot multiple metrics over time.

    Parameters
    ----------
    time : 1D ndarray
        time history
    data : 2D or 3D ndarray
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
    second_y_scale : float or dict, optional
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
    >>> from dstauffman import plot_time_history
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> time  = np.arange(0, 5, 1./12) + 2000
    >>> data  = np.random.rand(len(time), 5)
    >>> mag   = data.cumsum(axis=1)[:,-1]
    >>> data  = 10 * data / np.expand_dims(mag, axis=1)
    >>> label = 'Random Data'
    >>> fig   = plot_time_history(time, data, label)

    Close plot

    >>> plt.close(fig)

    """
    # check optional inputs
    if opts is None:
        opts = Opts()
    if colormap is None:
        if opts.colormap is None:
            colormap = DEFAULT_COLORMAP
        else:
            colormap = opts.colormap
    legend_loc = opts.legend_loc
    show_zero  = opts.show_zero
    time_units = opts.base_time
    unit_text = ' [' + units + ']' if units else ''
    (scale, prefix) = get_factors(opts.vert_fact)

    # check for valid data
    if ignore_plot_data(data, ignore_empties):
        print(' ' + label + ' plot skipped due to missing data.')
        return None
    assert time.ndim == 1, 'Time must be a 1D array.'
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
        assert False, 'Data must be a 2D or 3D array.'
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
    # optionally force zero to be plotted
    if show_zero and min(ax.get_ylim()) > 0:
        ax.set_ylim(bottom=0)
    # set years to always be whole numbers on the ticks
    if time_units == 'year' and (np.max(time) - np.min(time)) >= 4:
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))

    # optionally add second Y axis
    if second_y_scale is not None:
        ax2 = ax.twinx()
        if isinstance(second_y_scale, (int, float)):
            ax2.set_ylim(np.multiply(second_y_scale, ax.get_ylim()))
        else:
            for (key, value) in second_y_scale.items():
                ax2.set_ylim(np.multiply(value, ax.get_ylim()))
                ax2.set_ylabel(key)

    # setup plots
    setup_plots(fig, opts, 'time')
    return fig

#%% Functions - plot_monte_carlo
def plot_monte_carlo(time, data, label, units='unity', opts=None, *, plot_indiv=True, \
    truth=None, plot_as_diffs=False, second_y_scale=None, truth_time=None, \
    truth_data=None, plot_sigmas=1, plot_confidence=0, colormap=None):
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
    second_y_scale : float or dict, optional
        Multiplication scale factor to use to display on a secondary Y axis
    truth_time : array_like, optional
        Time for truth data
    truth_data : array_line, optional
        Date for corresponding truth history
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
    >>> from dstauffman import plot_monte_carlo
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> time  = np.arange(0, 10, 0.1)
    >>> data  = np.sin(time)
    >>> label = 'Sin'
    >>> units = 'population'
    >>> fig   = plot_monte_carlo(time, data, label, units)

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
    legend_loc    = opts.legend_loc
    show_zero     = opts.show_zero
    time_units    = opts.base_time
    unit_text     = ' [' + units + ']' if units else ''
    (scale, prefix) = get_factors(opts.vert_fact)

    # maintain older API
    if truth_data is not None: # pragma: no cover
        if truth is not None:
            raise ValueError('Attempting to use both APIs, please only use new truth input.')
        else:
            warnings.warn('This API will be removed in the future, please use the new truth input.', DeprecationWarning)
            truth = TruthPlotter(truth_time, truth_data)
    show_legend = rms_in_legend or plot_as_diffs or (truth is not None and not truth.is_null)

    # ensure that data is at least 2D
    if data.ndim == 0:
        data = np.atleast_2d(data)
    elif data.ndim == 1:
        data = data[:, np.newaxis] # forces to grow in second dimension, instead of first

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
            sigma_label = '$\pm {}\sigma$'.format(plot_sigmas)
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
    if second_y_scale is not None:
        ax2 = ax.twinx()
        if isinstance(second_y_scale, (int, float)):
            ax2.set_ylim(np.multiply(second_y_scale, ax.get_ylim()))
        else:
            for (key, value) in second_y_scale.items():
                ax2.set_ylim(np.multiply(value, ax.get_ylim()))
                ax2.set_ylabel(key)
    # Setup plots
    setup_plots(fig, opts, 'time')
    return fig

#%% Functions - plot_correlation_matrix
def plot_correlation_matrix(data, labels=None, units='unity', opts=None, *, matrix_name='Correlation Matrix', \
        cmin=0, cmax=1, xlabel='', ylabel='', plot_lower_only=True, label_values=False, x_lab_rot=90, \
        colormap=None, plot_border=None):
    r"""
    Visually plot a correlation matrix.

    Parameters
    ----------
    data : array_like
        data for corresponding time history
    labels : list of str, optional
        Names to put on row and column headers
    units : str, optional
        units of the data to be displayed on the plot
    opts : class Opts, optional
        plotting options
    matrix_name : str, optional
        Name to put on figure and plot title
    cmin : float, optional
        Minimum value for color range, default is zero
    cmax : float, optional
        Maximum value for color range, default is one
    xlabel : str, optional
        X label to put on plot
    ylabel : str, optional
        Y label to put on plot
    plot_lower_only : bool, optional
        Plots only the lower half of a symmetric matrix, default is True
    label_values : bool, optional
        Annotate the numerical values of each square in addition to the color code, default is False
    x_lab_rot : float, optional
        Amount in degrees to rotate the X labels, default is 90
    colormap : str or matplotlib.colors.Colormap, optional
        Name of colormap to use, if specified, overrides the opts.colormap
    plot_border : str, optional
        Color of the border to plot

    Returns
    -------
    fig : object
        figure handle

    Notes
    -----
    #.  Written by David C. Stauffer in July 2015.

    Examples
    --------
    >>> from dstauffman import plot_correlation_matrix, unit
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> data = np.random.rand(10, 10)
    >>> labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    >>> data = unit(data, axis=0)
    >>> fig = plot_correlation_matrix(data, labels)

    Close plots
    >>> plt.close(fig)

    """
    # check optional inputs
    if opts is None:
        opts = Opts()
    if colormap is None:
        if opts.colormap is None:
            colormap = 'cool'
        else:
            colormap = opts.colormap
    (scale, prefix) = get_factors(opts.vert_fact)

    # Hard-coded values
    box_size        = 1
    precision       = 1e-12

    # get sizes
    (n, m) = data.shape

    # check labels
    if labels is None:
        xlab = [str(i) for i in range(m)]
        ylab = [str(i) for i in range(n)]
    else:
        if isinstance(labels[0], list):
            xlab = labels[0]
            ylab = labels[1]
        else:
            xlab = labels
            ylab = labels
    # check lengths
    if len(xlab) != m or len(ylab) != n:
        raise ValueError('Incorrectly sized labels.')

    # Determine if symmetric
    if m == n and np.all((np.abs(data - np.transpose(data)) < precision) | np.isnan(data)):
        is_symmetric = True
    else:
        is_symmetric = False
    plot_lower_only  = plot_lower_only and is_symmetric

    # Override color ranges based on data
    # test if in -1 to 1 range instead of 0 to 1
    if np.all(data >= -1 + precision) and np.any(data <= -precision) and cmin == 0 and cmax == 1:
        cmin = -1
    # test if outside the cmin to cmax range, and if so, adjust range.
    temp = np.min(data)
    if temp < cmin:
        cmin = temp
    temp = np.max(data)
    if temp > cmax:
        cmax = temp

    # determine which type of data to plot
    this_title = matrix_name # + (' [' + units + ']' if units else '')

    # Create plots
    # create figure
    fig = plt.figure()
    # set figure title
    fig.canvas.set_window_title(this_title)
    # get handle to axes for use later
    ax = fig.add_subplot(111)
    # set axis color to none
    ax.patch.set_facecolor('none')
    # set title
    ax.set_title(this_title)
    # get colormap based on high and low limits
    cm = ColorMap(colormap, low=scale*cmin, high=scale*cmax)
    # loop through and plot each element with a corresponding color
    for i in range(m):
        for j in range(n):
            if not plot_lower_only or (i <= j):
                if not np.isnan(data[j, i]):
                    ax.add_patch(Rectangle((box_size*i,box_size*j),box_size, box_size, \
                        facecolor=cm.get_color(scale*data[j, i]), edgecolor=plot_border))
                if label_values:
                    ax.annotate('{:.2g}'.format(scale*data[j,i]), xy=(box_size*i + box_size/2, box_size*j + box_size/2), \
                        xycoords='data', horizontalalignment='center', \
                        verticalalignment='center', fontsize=15)
    # show colorbar
    fig.colorbar(cm.get_smap())
    # make square
    ax.set_aspect('equal')
    # set limits and tick labels
    ax.set_xlim(0, m)
    ax.set_ylim(0, n)
    ax.set_xticks(np.arange(0, m)+box_size/2)
    ax.set_xticklabels(xlab, rotation=x_lab_rot)
    ax.set_yticks(np.arange(0, n)+box_size/2)
    ax.set_yticklabels(ylab)
    # label axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # reverse the y axis
    ax.invert_yaxis()

    # Setup plots
    setup_plots(fig, opts, 'dist')
    return fig

#%% Functions - plot_bar_breakdown
def plot_bar_breakdown(time, data, label, opts=None, *, legend=None, ignore_empties=False, colormap=None):
    r"""
    Plot the pie chart like breakdown by percentage in each category over time.

    Parameters
    ----------
    time : array_like
        time history
    data : array_like
        data for corresponding time history, 2D: time by ratio in each category
    label : str
        Name to label on the plots
    opts : class Opts, optional
        plotting options
    legend : list of str, optional
        Names to use for each channel of data
    ignore_empties : bool, optional
        Removes any entries from the plot and legend that contain only zeros or only NaNs
    colormap : str or matplotlib.colors.Colormap, optional
        Name of colormap to use, if specified, overrides the opts.colormap

    Returns
    -------
    fig : object
        figure handle

    Notes
    -----
    #.  Written by David C. Stauffer in June 2015.

    Examples
    --------
    >>> from dstauffman import plot_bar_breakdown
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> time  = np.arange(0, 5, 1./12) + 2000
    >>> data  = np.random.rand(len(time), 5)
    >>> mag   = np.sum(data, axis=1)
    >>> data  = data / np.expand_dims(mag, axis=1)
    >>> label = 'Test'
    >>> fig   = plot_bar_breakdown(time, data, label)

    Close plots
    >>> plt.close(fig)

    """
    # check optional inputs
    if opts is None:
        opts = Opts()
    if colormap is None:
        if opts.colormap is None:
            colormap = DEFAULT_COLORMAP
        else:
            colormap = opts.colormap
    legend_loc = opts.legend_loc
    time_units = opts.base_time

    # check for valid data
    if ignore_plot_data(data, ignore_empties):
        print(' ' + label + ' plot skipped due to missing data.')
        return

    # hard-coded values
    this_title = label + ' vs. Time'
    scale      = 100
    units      = '%'
    unit_text  = ' [' + units + ']'

    # data checks
    num_bins   = data.shape[1]
    if legend is not None:
        assert len(legend) == num_bins, 'Number of data channels does not match the legend.'
    else:
        legend = ['Series {}'.format(i+1) for i in range(num_bins)]

    # get colormap based on high and low limits
    cm = ColorMap(colormap, 0, num_bins-1)

    # figure out where the bottoms should be to stack the data
    bottoms = np.concatenate((np.zeros((len(time),1)), np.cumsum(data, axis=1)), axis=1)

    # plot breakdown
    fig = plt.figure()
    fig.canvas.set_window_title(this_title)
    ax = fig.add_subplot(111)
    for i in range(num_bins):
        if not ignore_plot_data(data, ignore_empties, col=i):
            # Note: The performance of ax.bar is really slow with large numbers of bars (>20), so
            # fill_between is a better alternative
            ax.fill_between(time, scale*bottoms[:, i], scale*bottoms[:, i+1], step='mid', \
                label=legend[i], color=cm.get_color(i), edgecolor='none')
    ax.set_xlabel('Time [' + time_units + ']')
    ax.set_ylabel(label + unit_text)
    ax.set_ylim(0, 100)
    ax.grid(True)
    # set years to always be whole numbers on the ticks
    if (time[-1] - time[0]) >= 4:
        ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
    ax.legend(loc=legend_loc)
    ax.set_title(this_title)

    # Setup plots
    setup_plots(fig, opts, 'time')
    return fig

#%% Functions - plot_bpe_convergence
def plot_bpe_convergence(costs, opts=None):
    r"""
    Plot the BPE convergence rate by iteration on a log scale.

    Parameters
    ----------
    costs : array_like
        Costs for the beginning run, each iteration, and final run
    opts : class Opts, optional
        Plotting options

    Returns
    -------
    fig : object
        figure handle

    Notes
    -----
    #.  Written by David C. Stauffer in July 2016.

    Examples
    --------
    >>> from dstauffman import plot_bpe_convergence
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> costs = np.array([1, 0.1, 0.05, 0.01])
    >>> fig = plot_bpe_convergence(costs)

    Close plots
    >>> plt.close(fig)

    """
    # check optional inputs
    if opts is None:
        opts = Opts()

    # get number of iterations
    num_iters = len(costs) - 2
    time      = np.arange(len(costs))
    labels    = ['Begin'] + [str(x+1) for x in range(num_iters)] + ['Final']

    # alias the title
    this_title = 'Convergence by Iteration'
    # create the figure and set the title
    fig = plt.figure()
    fig.canvas.set_window_title(this_title)
    # add an axis and plot the data
    ax = fig.add_subplot(111)
    ax.semilogy(time, costs, 'b.-', linewidth=2)
    # add labels
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.set_title(this_title)
    ax.set_xticks(time)
    ax.set_xticklabels(labels)
    # show a grid
    ax.grid(True)
    # Setup plots
    setup_plots(fig, opts, 'time')
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
    legend_loc = opts.legend_loc

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
    setup_plots(fig, opts, 'dist_no_yscale')

    return fig

#%% Functions - general_quaternion_plot
def general_quaternion_plot(description, time, quat_one, quat_two, name_one, name_two, *,
        ix_rms_xmin, ix_rms_xmax, start_date='', fig_visible=True, make_subplots=True,
        plot_components=True, legend_loc='best'):
    r"""
    Generic quaternion comparison plot for use in other wrapper functions.  This function plots two
    quaternion histories over time, along with a difference from one another.

    Input
    -----
    description : str
        name of the data being plotted, used as title
    time : array_like
        time history [sec]
    quat_one : (4, N) ndarray
        quaternion one
    quat_two : (4, N) ndarray
        quaternion two
    name_one : str
        name of data source 1
    name_two : str
        name of data source 2
    ix_rms_xmin : int
        index to first point of RMS calculation
    ix_rms_xmax : int
        index to last point of RMS calculation
    start_date : str
        date of t(0), may be an empty string
    fig_visible : bool
        whether figure is visible
    make_subplots : bool
        flag to use subplots
    plot_components : bool
        flag to plot components as opposed to angular difference

    Returns
    -------
    fig_hand : list of class matplotlib.Figure
        list of figure handles
    err : (3,N) ndarray
        Quaternion differences expressed in Q1 frame

    See Also
    --------
    TBD_wrapper

    Notes
    -----
    #.  Written by David C. Stauffer in MATLAB in October 2011, updated in 2018.
    #.  Ported to Python by David C. Stauffer in December 2018.

    Examples
    --------
    >>> from dstauffman import general_quaternion_plot, quat_norm
    >>> import numpy as np
    >>> from datetime import datetime
    >>> description     = 'example'
    >>> time            = np.arange(11)
    >>> quat_one        = quat_norm(np.random.rand(4, 11))
    >>> quat_two        = quat_norm(np.random.rand(4, 11))
    >>> name_one        = 'test1'
    >>> name_two        = 'test2'
    >>> start_date      = str(datetime.now())
    >>> ix_rms_xmin     = 0
    >>> ix_rms_xmax     = 10
    >>> fig_visible     = True
    >>> make_subplots   = True
    >>> plot_components = True
    >>> (fig_hand, err) = general_quaternion_plot(description, time, quat_one, quat_two, name_one, \
    ...     name_two, ix_rms_xmin=ix_rms_xmin, ix_rms_xmax=ix_rms_xmax, start_date=start_date, \
    ...     fig_visible=fig_visible, make_subplots=make_subplots, plot_components=plot_components)

    """
#    # force inputs to be ndarrays
#    time = np.asanyarray(time)
#    quat_one = np.asanyarray(quat_one)
#    quat_two = np.asanyarray(quat_two)

    # determine if you have the quaternions
    have_quat_one = quat_one is not None and np.any(~np.isnan(quat_one))
    have_quat_two = quat_two is not None and np.any(~np.isnan(quat_two))
    #% calculations
    if have_quat_one:
        q1_rms = rms(quat_one[:,ix_rms_xmin:ix_rms_xmax], axis=1, ignore_nans=True)
    if have_quat_two:
        q2_rms = rms(quat_two[:,ix_rms_xmin:ix_rms_xmax], axis=1, ignore_nans=True)
    # output errors
    if have_quat_one and have_quat_two:
        (nondeg_angle, nondeg_error) = quat_angle_diff(quat_one, quat_two)
        nondeg_rms = rms(nondeg_error[:, ix_rms_xmin:ix_rms_xmax], axis=1, ignore_nans=True)
        if plot_components:
            err = nondeg_rms
        else:
            err = np.array([rms(nondeg_angle[ix_rms_xmin:ix_rms_xmax], ignore_nans=True), np.nan, np.nan])
    else:
        err = np.full(3, np.nan, dtype=float)
    # get default plotting colors
    color_lists = get_color_lists()
    colororder3 = ColorMap(color_lists['vec'], num_colors=3)
    colororder8 = ColorMap(color_lists['quat_diff'], num_colors=8)
    # quaternion component names
    names = ['X', 'Y', 'Z', 'S']
    # names = ['qx', 'qy', 'qz', 'qs']
    # TODO: make non-harded coded
    # unit conversion value
    rad2urad = 1e6

    #% Overlay plots
    f1 = plt.figure()
    # create axis
    if make_subplots:
        f1.canvas.set_window_title(description) # TODO: fig_visible (in wrapper)?
        if have_quat_one and have_quat_two:
            ax1 = f1.add_subplot(2,1,1)
        else:
            ax1 = f1.add_subplot(111)
        colororder8.set_colors(ax1)
    else:
        f1.canvas.set_window_title(description + ' Quaternion Components')
        ax1 = f1.add_subplot(111)
    # plot data
    if have_quat_one:
        for i in range(4):
            this_label = name_one + ' ' + names[i] + ' (RMS: {:1.3f})'.format(q1_rms[i])
            ax1.plot(time, quat_one[i,:], '^-', markersize=4, label=this_label)
    if have_quat_two:
        for i in range(4):
            this_label = name_two + ' ' + names[i] + ' (RMS: {:1.3f})'.format(q2_rms[i])
            ax1.plot(time, quat_two[i,:], 'v:', markersize=4, label=this_label)
    # format display of plot
    ax1.legend(loc=legend_loc)
    #plot_rms_lines(time([ix_rms_xmin,ix_rms_xmax]),ylim)
    ax1.set_title(description + ' Quaternion Components')
    ax1.set_xlabel('Time [sec]' + start_date)
    ax1.set_ylabel(description + ' Quaternion Components [dimensionless]')
    ax1.grid(True)

    #% Difference plot
    if have_quat_one and have_quat_two:
        # make axis
        if make_subplots:
            ax2 = f1.add_subplot(2,1,2)
            f2 = None
        else:
            f2 = plt.figure()
            f2.canvas.set_window_title(description + ' Difference')
            ax2 = f2.add_subplot(111)
        colororder3.set_colors(ax2)
        # plot data
        if plot_components:
            ax2.plot(time, nondeg_error.T, '^-', markersize=4)
            ax2.legend([ \
                'X (RMS: {:1.3f} \murad)'.format(rad2urad*nondeg_rms[0]), \
                'Y (RMS: {:1.3f} \murad)'.format(rad2urad*nondeg_rms[1]), \
                'Z (RMS: {:1.3f} \murad)'.format(rad2urad*nondeg_rms[2])], loc=legend_loc)
            #set z-axis data to be plotted underneath everything else
            # TODO: use zorder, but need separate calls
        else:
            ax2.plot(time, nondeg_angle, '^-', markersize=4)
            ax2.legend('Angle (RMS: {:1.3f} \murad)'.format(rad2urad*nondeg_rms[0]), loc=legend_loc)
        # format display of plot
        #plot_rms_lines(time([ix_rms_xmin,ix_rms_xmax]),ylim)
        ax2.set_title(description + '  Difference')
        ax2.set_xlabel('Time [sec]' + start_date)
        ax2.set_ylabel(description + ' Difference [rad]')
        plt.grid(True)
        # link axes to zoom together
        #linkaxes([ax1, ax2],'x')
    else:
        f2 = None
    fig_hand = [x for x in (f1, f2) if x is not None]
    return (fig_hand, err)

#%% Unit test
if __name__ == '__main__':
    plt.ioff()
    unittest.main(module='dstauffman.tests.test_plotting', exit=False)
    doctest.testmod(verbose=False)
