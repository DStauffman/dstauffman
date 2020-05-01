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
from dstauffman.plot_support import ColorMap, get_color_lists, ignore_plot_data, \
                                        plot_rms_lines, plot_second_yunits, setup_plots, \
                                        show_zero_ylim, whitten
from dstauffman.quat import quat_angle_diff
from dstauffman.stats import intersect, z_from_ci
from dstauffman.units import get_factors
from dstauffman.utils import pprint_dict, rms

#%% Classes - Opts
class Opts(Frozen):
    r"""Optional plotting configurations."""
    def __init__(self):
        r"""
        Default configuration with:
            .case_name : str
                Name of the case to be plotted
            .date_zero : (1x6) datevec (or datetime)
                Date of t = 0 time [year month day hour minute second]
            .save_plot : bool
                Flag for whether to save the plots
            .save_path : str
                Location for the plots to be saved
            .show_plot : bool
                Flag to show the plots or only save to disk
            .show_link : bool
                Flag to show a link to the folder where the plots were saved
            .plot_type : str
                Type of plot to save to disk, from {'png','jpg','fig','emf'}
            .sub_plots : bool
                Flag specifying whether to plot as subplots or separate figures
            .sing_line : bool
                Flag specifying whether to plot only one line per axes, using subplots as necessary
            .disp_xmin : float
                Minimum time to display on plot [sec]
            .disp_xmax : float
                Maximum time to display on plot [sec]
            .rms_xmin  : float
                Minimum time from which to begin RMS calculations [sec]
            .rms_xmax  : float
                Maximum time from which to end RMS calculations [sec]
            .show_rms  : bool
                Flag for whether to show the RMS in the legend
            .use_mean  : bool
                Flag for using mean instead of RMS for legend calculations
            .show_zero : bool
                Flag for whether to show Y=0 on the plot axis
            .quat_comp : bool
                Flag to plot quaternion component differences or just the angle
            .show_xtra : bool
                Flag to show extra points in one vector or the other when plotting differences
            .time_base : str
                Base units of time, typically from {'sec', 'months'}
            .time_unit : str
                Time unit for the x axis, from {'', 'sec', 'min', 'hr', 'day', 'month', 'year'}
            .vert_fact : str
                Vertical factor to apply to the Y axis,
                from: {'yotta','zetta','exa','peta','tera','giga','mega','kilo','hecto','deca',
                'unity','deci','centi','milli', 'micro','nano','pico','femto','atto','zepto','yocto'}
            .colormap  : str
                Name of the colormap to use
            .leg_spot  : str
                Location to place the legend, from {'best', 'upper right', 'upper left',
                'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center',
                'upper center', 'center' or tuple of position}
            .classify  : str
                Classification level to put on plots
            .names     : list of str
                Names of the data structures to be plotted
        """
        self.case_name = ''
        self.date_zero = None
        self.save_plot = False
        self.save_path = os.getcwd()
        self.show_plot = True
        self.show_link = False # TODO: is this used?
        self.plot_type = 'png'
        self.sub_plots = True
        self.sing_line = False
        self.disp_xmin = -np.inf
        self.disp_xmax =  np.inf
        self.rms_xmin  = -np.inf
        self.rms_xmax  =  np.inf
        self.show_rms  = True
        self.use_mean  = False
        self.show_zero = False
        self.quat_comp = True
        self.show_xtra = True
        self.time_base = 'sec'
        self.time_unit = 'sec'
        self.vert_fact = 'unity'
        self.colormap  = None
        self.leg_spot  = 'best'
        self.classify  = ''
        self.names     = list()

    def get_names(self, ix):
        r"""Get the specified name from the list."""
        if hasattr(self, 'names') and len(self.names) >= ix+1:
            name = self.names[ix]
        else:
            name = ''
        return name

    def get_date_zero_str(self):
        r"""
        Gets a string representation of date_zero, typically used to print on an X axis.

        Returns
        -------
        start_date : str
            String representing the date of time zero.

        Examples
        --------
        >>> from dstauffman import Opts
        >>> from datetime import datetime
        >>> opts = Opts()
        >>> opts.date_zero = datetime(2019, 4, 1, 18, 0, 0)
        >>> print(opts.get_date_zero_str())
          t(0) = 01-Apr-2019 18:00:00 Z

        """
        TIMESTR_FORMAT = '%d-%b-%Y %H:%M:%S'
        start_date = '  t(0) = ' + self.date_zero.strftime(TIMESTR_FORMAT) + ' Z' if self.date_zero is not None else ''
        return start_date

    def pprint(self, indent=1, align=True):
        r"""Display a pretty print version of the class."""
        pprint_dict(self.__dict__, name=self.__class__.__name__, indent=indent, align=align)

#%% Functions - plot_time_history
def plot_time_history(time, data, label, units='', opts=None, *, legend=None, \
        second_y_scale=None, ignore_empties=False, data_lo=None, data_hi=None, colormap=None):
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
    # optionally force zero to be plotted
    if show_zero and min(ax.get_ylim()) > 0:
        ax.set_ylim(bottom=0) # TODO: use this to functionalize
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
def plot_monte_carlo(time, data, label, units='', opts=None, *, plot_indiv=True, \
    truth=None, plot_as_diffs=False, second_y_scale=None, plot_sigmas=1, \
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
    second_y_scale : float or dict, optional
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
def plot_correlation_matrix(data, labels=None, units='', opts=None, *, matrix_name='Correlation Matrix', \
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

    Close plot
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
    this_title = matrix_name + (' [' + units + ']' if units else '')

    # Create plots
    # create figure
    fig = plt.figure()
    # set figure title
    fig.canvas.set_window_title(matrix_name)
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
                    ax.add_patch(Rectangle((box_size*i, box_size*j), box_size, box_size, \
                        facecolor=cm.get_color(scale*data[j, i]), edgecolor=plot_border))
                if label_values:
                    ax.annotate('{:.2g}'.format(scale*data[j, i]), xy=(box_size*i + box_size/2, box_size*j + box_size/2), \
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
    legend_loc = opts.leg_spot
    time_units = opts.time_base

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
    bottoms = np.concatenate((np.zeros((len(time), 1)), np.cumsum(data, axis=1)), axis=1)

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

#%% Functions - general_quaternion_plot
def general_quaternion_plot(description, time_one, time_two, quat_one, quat_two, *,
        name_one='', name_two='', time_units='sec', start_date='', plot_components=True,
        rms_xmin=-np.inf, rms_xmax=np.inf, disp_xmin=-np.inf, disp_xmax=np.inf, fig_visible=True,
        make_subplots=True, single_lines=False, use_mean=False, plot_zero=False, show_rms=True,
        legend_loc='best', show_extra=True, truth_name='Truth', truth_time=None, truth_data=None):
    r"""
    Generic quaternion comparison plot for use in other wrapper functions.  This function plots two
    quaternion histories over time, along with a difference from one another.

    Input
    -----
    description : str
        name of the data being plotted, used as title
    time_one : array_like
        time history one [sec]
    time_two : array_like
        time history two [sec]
    quat_one : (4, N) ndarray
        quaternion one
    quat_two : (4, N) ndarray
        quaternion two
    name_one : str, optional
        name of data source 1
    name_two : str, optional
        name of data source 2
    time_units : str, optional
        time units, defaults to 'sec'
    start_date : str, optional
        date of t(0), may be an empty string
    plot_components : bool, optional
        Whether to plot the quaternion components, or just the angular difference
    rms_xmin : float, optional
        time of first point of RMS calculation
    rms_xmax : float, optional
        time of last point of RMS calculation
    disp_xmin : float, optional
        lower time to limit the display of the plot
    disp_xmax : float, optional
        higher time to limit the display of the plot
    fig_visible : bool, optional
        whether figure is visible
    make_subplots : bool, optional
        flag to use subplots for differences
    single_lines : bool, optional
        flag meaning to plot subplots by channel instead of together
    use_mean : bool, optional
        whether to use mean instead of RMS in legend calculations
    plot_zero : bool, optional
        whether to force zero to always be plotted on the Y axis
    show_rms : bool, optional
        whether to show the RMS calculation in the legend
    legend_loc : str, optional
        location to put the legend, default is 'best'
    show_extra : bool, optional
        whether to show missing data on difference plots
    truth_name : str, optional
        name to associate with truth data, default is 'Truth'
    truth_time : ndarray, optional
        truth time history
    truth_data : ndarray, optional
        truth quaternion history

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
    #.  Made fully functional by David C. Stauffer in March 2019.

    Examples
    --------
    >>> from dstauffman import general_quaternion_plot, quat_norm
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from datetime import datetime
    >>> description     = 'example'
    >>> time_one        = np.arange(11)
    >>> time_two        = np.arange(2, 13)
    >>> quat_one        = quat_norm(np.random.rand(4, 11))
    >>> quat_two        = quat_norm(quat_one[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1]] + 1e-5 * np.random.rand(4, 11))
    >>> name_one        = 'test1'
    >>> name_two        = 'test2'
    >>> time_units      = 'sec'
    >>> start_date      = str(datetime.now())
    >>> plot_components = True
    >>> rms_xmin        = 1
    >>> rms_xmax        = 10
    >>> disp_xmin       = -2
    >>> disp_xmax       = np.inf
    >>> fig_visible     = True
    >>> make_subplots   = True
    >>> single_lines    = False
    >>> use_mean        = False
    >>> plot_zero       = False
    >>> show_rms        = True
    >>> legend_loc      = 'best'
    >>> show_extra      = True
    >>> truth_name      = 'Truth'
    >>> truth_time      = None
    >>> truth_data      = None
    >>> (fig_hand, err) = general_quaternion_plot(description, time_one, time_two, quat_one, quat_two,
    ...     name_one=name_one, name_two=name_two, time_units=time_units, start_date=start_date, \
    ...     plot_components=plot_components, rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, \
    ...     disp_xmax=disp_xmax, fig_visible=fig_visible, make_subplots=make_subplots, single_lines=single_lines, \
    ...     use_mean=use_mean, plot_zero=plot_zero, show_rms=show_rms, legend_loc=legend_loc, \
    ...     show_extra=show_extra, truth_name=truth_name, truth_time=truth_time, truth_data=truth_data)

    Close plots
    >>> for fig in fig_hand:
    ...     plt.close(fig)

    """
    # hard-coded values
    leg_format = '{:1.3f}'
    truth_color = 'k'

#    # force inputs to be ndarrays
#    time_one = np.asanyarray(time_one)
#    time_two = np.asanyarray(time_two)
#    quat_one = np.asanyarray(quat_one)
#    quat_two = np.asanyarray(quat_two)

    # determine if you have the quaternions
    have_quat_one = quat_one is not None and np.any(~np.isnan(quat_one))
    have_quat_two = quat_two is not None and np.any(~np.isnan(quat_two))
    have_both     = have_quat_one and have_quat_two
    # determine if using datetimes
    use_datetime = False # TODO: test with these

    #% calculations
    (time_overlap, q1_diff_ix, q2_diff_ix) = intersect(time_one, time_two) # TODO: add a tolerance?
    # find differences
    q1_miss_ix = np.setxor1d(np.arange(len(time_one)), q1_diff_ix)
    q2_miss_ix = np.setxor1d(np.arange(len(time_two)), q2_diff_ix)
    # build RMS indices
    rms_ix1  = (time_one >= rms_xmin) & (time_one <= rms_xmax)
    rms_ix2  = (time_two >= rms_xmin) & (time_two <= rms_xmax)
    rms_ix3  = (time_overlap >= rms_xmin) & (time_overlap <= rms_xmax)
    rms_pts1 = np.maximum(rms_xmin, np.minimum(np.min(time_one), np.min(time_two)))
    rms_pts2 = np.minimum(rms_xmax, np.maximum(np.max(time_one), np.max(time_two)))
    # get default plotting colors
    color_lists = get_color_lists()
    colororder3 = ColorMap(color_lists['vec'], num_colors=3)
    colororder8 = ColorMap(color_lists['quat_diff'], num_colors=8)
    # quaternion component names
    elements = ['X', 'Y', 'Z', 'S']
    num_channels = len(elements)
    # calculate the difference
    if have_both:
        (nondeg_angle, nondeg_error) = quat_angle_diff(quat_one[:, q1_diff_ix], quat_two[:, q2_diff_ix])
    # calculate the rms (or mean) values
    nans = np.full(3, np.nan, dtype=float)
    if not use_mean:
        func_name   = 'RMS'
        q1_func     = rms(quat_one[:, rms_ix1], axis=1, ignore_nans=True) if have_quat_one else nans
        q2_func     = rms(quat_two[:, rms_ix2], axis=1, ignore_nans=True) if have_quat_two else nans
        nondeg_func = rms(nondeg_error[:, rms_ix3], axis=1, ignore_nans=True) if have_both else nans
        mag_func    = rms(nondeg_angle[rms_ix3], axis=0, ignore_nans=True) if have_both else nans[0:1]
    else:
        func_name   = 'Mean'
        q1_func     = np.nanmean(quat_one[:, rms_ix1], axis=1) if have_quat_one else nans
        q2_func     = np.nanmean(quat_two[:, rms_ix2], axis=1) if have_quat_two else nans
        nondeg_func = np.nanmean(nondeg_error[:, rms_ix3], axis=1) if have_both else nans
        mag_func    = np.nanmean(nondeg_angle[rms_ix3], axis=0) if have_both else nans[0:1]
    # output errors
    err = {'one': q1_func, 'two': q2_func, 'diff': nondeg_func, 'mag': mag_func}
    # unit conversion value
    (temp, prefix) = get_factors('micro')
    leg_conv = 1/temp
    # determine which symbols to plot with
    if have_both:
        symbol_one = '^-'
        symbol_two = 'v:'
    elif have_quat_one:
        symbol_one = '.-'
        symbol_two = '' # not-used
    elif have_quat_two:
        symbol_one = '' # not-used
        symbol_two = '.-'
    else:
        symbol_one = '' # invalid case
        symbol_two = '' # invalid case
    # pre-plan plot layout
    if have_both:
        if make_subplots:
            num_figs = 1
            if single_lines:
                num_rows = num_channels
                num_cols = 2
            else:
                num_rows = 2
                num_cols = 1
        else:
            num_figs = 2
            num_cols = 1
            if single_lines:
                num_rows = num_channels
            else:
                num_rows = 1
    else:
        num_figs = 1
        if single_lines:
            num_rows = num_channels
            num_cols = 1
        else:
            num_rows = 1
            num_cols = 1
    num_axes = num_figs*num_rows*num_cols

    #% Create plots
    # create figures
    f1 = plt.figure()
    if make_subplots:
        f1.canvas.set_window_title(description) # TODO: fig_visible (in wrapper)?
    else:
        f1.canvas.set_window_title(description + ' Quaternion Components')
    if have_both and not make_subplots:
        f2 = plt.figure()
        f2.canvas.set_window_title(description + 'Difference')
        fig_hand = [f1, f2]
    else:
        fig_hand = [f1]
    # create axes
    ax = []
    ax_prim = None
    for i in range(num_figs):
        for j in range(num_cols):
            for k in range(num_rows):
                temp_axes = fig_hand[i].add_subplot(num_rows, num_cols, k*num_cols + j + 1, sharex=ax_prim)
                if ax_prim is None:
                    ax_prim = temp_axes
                ax.append(temp_axes)
    # plot data
    for i in range(num_axes):
        this_axes = ax[i]
        is_diff_plot = i > num_rows-1 or (not single_lines and make_subplots and i == 1)
        if single_lines:
            if is_diff_plot:
                loop_counter = [i - num_rows]
            else:
                loop_counter = [i]
        else:
            loop_counter = range(num_channels)
        if not is_diff_plot:
            # standard plot
            if have_quat_one:
                for j in loop_counter:
                    if show_rms:
                        value = leg_format.format(q1_func[j])
                        this_label = '{} {} ({}: {})'.format(name_one, elements[j], func_name, value)
                    else:
                        this_label = name_one + ' ' + elements[j]
                    this_axes.plot(time_one, quat_one[j, :], symbol_one, markersize=4, label=this_label, \
                        color=colororder8.get_color(j+(0 if have_quat_two else num_channels)), zorder=3)
            if have_quat_two:
                for j in loop_counter:
                    if show_rms:
                        value = leg_format.format(q2_func[j])
                        this_label = '{} {} ({}: {})'.format(name_two, elements[j], func_name, value)
                    else:
                        this_label = name_two + ' ' + elements[j]
                    this_axes.plot(time_two, quat_two[j, :], symbol_two, markersize=4, label=this_label, \
                        color=colororder8.get_color(j+num_channels), zorder=5)
        else:
            #% Difference plot
            zorders = [8, 6, 5]
            for j in range(3):
                if not plot_components or (single_lines and i % num_channels != j):
                    continue
                if show_rms:
                    value = leg_format.format(leg_conv*nondeg_func[j])
                    this_label = '{} ({}: {}) {}rad)'.format(elements[j], func_name, value, prefix)
                else:
                    this_label = elements[j]
                this_axes.plot(time_overlap, nondeg_error[j, :], '.-', markersize=4, label=this_label, zorder=zorders[j], \
                    color=colororder3.get_color(j))
            if not plot_components or (single_lines and (i + 1) % num_channels == 0):
                if show_rms:
                    value = leg_format.format(leg_conv*mag_func)
                    this_label = 'Angle ({}: {} {}rad)'.format(func_name, value, prefix)
                else:
                    this_label = 'Angle'
                this_axes.plot(time_overlap, nondeg_angle, '.-', markersize=4, label=this_label, color=colororder3.get_color(0))
            if show_extra:
                this_axes.plot(time_one[q1_miss_ix], np.zeros(len(q1_miss_ix)), 'kx', markersize=8, markeredgewidth=2, markerfacecolor='None', label=name_one + ' Extra')
                this_axes.plot(time_one[q2_miss_ix], np.zeros(len(q2_miss_ix)), 'go', markersize=8, markeredgewidth=2, markerfacecolor='None', label=name_two + ' Extra')

        # set X display limits
        if i == 0:
            xlim = list(this_axes.get_xlim())
            if np.isfinite(disp_xmin):
                xlim[0] = max([xlim[0], disp_xmin])
            if np.isfinite(disp_xmax):
                xlim[1] = min([xlim[1], disp_xmax])
        this_axes.set_xlim(xlim)
        # set Y display limits
        if plot_zero:
            show_zero_ylim(this_axes)
        # optionally plot truth (after having set axes limits)
        if i < num_rows and truth_time is not None and truth_data is not None and not np.all(np.isnan(truth_data)):
            if single_lines:
                this_axes.plot(truth_time, truth_data[i, :], '.-', color=truth_color, markerfacecolor=truth_color, \
                    linewidth=2, label=truth_name + ' ' + elements[i])
            else:
                if i == 0:
                    # TODO: add RMS to Truth data?
                    this_axes.plot(truth_time, truth_data[i, :], '.-', color=truth_color, markerfacecolor=truth_color, \
                        linewidth=2, label=truth_name)
        # format display of plot
        this_axes.legend(loc=legend_loc)
        if i == 0:
            this_axes.set_title(description + ' Quaternion Components')
        elif (single_lines and i == num_rows) or (not single_lines and i == 1):
            this_axes.set_title(description + ' Difference')
        if use_datetime:
            this_axes.set_xlabel('Date')
        else:
            this_axes.set_xlabel('Time [' + time_units + ']' + start_date)
        if is_diff_plot:
            this_axes.set_ylabel(description + ' Difference [rad]')
        else:
            this_axes.set_ylabel(description + ' Quaternion Components [dimensionless]')
        this_axes.grid(True)
        # plot RMS lines
        if show_rms:
            plot_rms_lines(this_axes, [rms_pts1, rms_pts2], this_axes.get_ylim())

    return (fig_hand, err)

#%% Functions - general_difference_plot
def general_difference_plot(description, time_one, time_two, data_one, data_two, *,
        name_one='', name_two='', elements=None, units=None, time_units='sec', leg_scale='unity',
        start_date='', rms_xmin=-np.inf, rms_xmax=np.inf, disp_xmin=-np.inf, disp_xmax=np.inf,
        fig_visible=True, make_subplots=True, single_lines=False, colormap=None, use_mean=False,
        plot_zero=False, show_rms=True, legend_loc='best', show_extra=True, second_y_scale=None,
        y_label=None, truth_name='Truth', truth_time=None, truth_data=None):
    r"""
    Generic difference comparison plot for use in other wrapper functions.  This function plots two
    vector histories over time, along with a difference from one another.

    Input
    -----
    description : str
        name of the data being plotted, used as title
    time_one : array_like
        time history one [sec]
    time_two : array_like
        time history two [sec]
    data_one : (A, N) ndarray
        vector one history
    data_two : (A, N) ndarray
        vector two history
    name_one : str, optional
        name of data source 1
    name_two : str, optional
        name of data source 2
    elements : list
        name of each element to plot within the vector
    units : list
        name of units for plot
    time_units : str, optional
        time units, defaults to 'sec'
    leg_scale : str, optional
        factor to use when scaling the value in the legend, default is 'unity'
    start_date : str, optional
        date of t(0), may be an empty string
    rms_xmin : float, optional
        time of first point of RMS calculation
    rms_xmax : float, optional
        time of last point of RMS calculation
    disp_xmin : float, optional
        lower time to limit the display of the plot
    disp_xmax : float, optional
        higher time to limit the display of the plot
    fig_visible : bool, optional
        whether figure is visible
    make_subplots : bool, optional
        flag to use subplots for differences
    single_lines : bool, optional
        flag meaning to plot subplots by channel instead of together
    colormap : list or colormap
        colors to use on the plot
    use_mean : bool, optional
        whether to use mean instead of RMS in legend calculations
    plot_zero : bool, optional
        whether to force zero to always be plotted on the Y axis
    show_rms : bool, optional
        whether to show the RMS calculation in the legend
    legend_loc : str, optional
        location to put the legend, default is 'best'
    show_extra : bool, optional
        whether to show missing data on difference plots
    second_y_scale : dict, optional
        single key and value pair to use for scaling data to a second Y axis
    y_label : str, optional
        Labels to put on the Y axes, potentially by element
    truth_name : str, optional
        name to associate with truth data, default is 'Truth'
    truth_time : ndarray, optional
        truth time history
    truth_data : ndarray, optional
        truth quaternion history

    Returns
    -------
    fig_hand : list of class matplotlib.Figure
        list of figure handles
    err : (A,N) ndarray
        Differences

    See Also
    --------
    TBD_wrapper

    Notes
    -----
    #.  Written by David C. Stauffer in MATLAB in October 2011, updated in 2018.
    #.  Ported to Python by David C. Stauffer in March 2019.
    #.  Made fully function by David C. Stauffer in April 2020.

    Examples
    --------
    >>> from dstauffman import general_difference_plot, get_color_lists
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from matplotlib.colors import ListedColormap
    >>> from datetime import datetime
    >>> description     = 'example'
    >>> time_one        = np.arange(11)
    >>> time_two        = np.arange(2, 13)
    >>> data_one        = 50e-6 * np.random.rand(2, 11)
    >>> data_two        = data_one[:, [2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1]] - 1e-6 * np.random.rand(2, 11)
    >>> name_one        = 'test1'
    >>> name_two        = 'test2'
    >>> elements        = ['x', 'y']
    >>> units           = 'rad'
    >>> time_units      = 'sec'
    >>> leg_scale       = 'micro'
    >>> start_date      = str(datetime.now())
    >>> rms_xmin        = 1
    >>> rms_xmax        = 10
    >>> disp_xmin       = -2
    >>> disp_xmax       = np.inf
    >>> fig_visible     = True
    >>> make_subplots   = True
    >>> single_lines    = False
    >>> color_lists     = get_color_lists()
    >>> colormap        = ListedColormap(color_lists['dbl_diff'].colors + color_lists['double'].colors)
    >>> use_mean        = False
    >>> plot_zero       = False
    >>> show_rms        = True
    >>> legend_loc      = 'best'
    >>> show_extra      = True
    >>> second_y_scale  = {u'µrad': 1e6}
    >>> y_label         = None
    >>> truth_name      = 'Truth'
    >>> truth_time      = None
    >>> truth_data      = None
    >>> (fig_hand, err) = general_difference_plot(description, time_one, time_two, data_one, data_two,
    ...     name_one=name_one, name_two=name_two, elements=elements, units=units, time_units=time_units, \
    ...     leg_scale=leg_scale, start_date=start_date, rms_xmin=rms_xmin, rms_xmax=rms_xmax, disp_xmin=disp_xmin, \
    ...     disp_xmax=disp_xmax, fig_visible=fig_visible, make_subplots=make_subplots, single_lines=single_lines, \
    ...     colormap=colormap, use_mean=use_mean, plot_zero=plot_zero, show_rms=show_rms, legend_loc=legend_loc, \
    ...     show_extra=show_extra, second_y_scale=second_y_scale, y_label=y_label, truth_name=truth_name, \
    ...     truth_time=truth_time, truth_data=truth_data)

    Close plots
    >>> for fig in fig_hand:
    ...     plt.close(fig)

    """
    # Hard-coded values
    leg_format  = '{:1.3f}'
    truth_color = 'k'

#    # force inputs to be ndarrays
#    time_one = np.asanyarray(time_one)
#    time_two = np.asanyarray(time_two)
#    data_one = np.asanyarray(data_one)
#    data_two = np.asanyarray(data_two)

    # determine if you have the histories
    have_data_one = data_one is not None and np.any(~np.isnan(data_one))
    have_data_two = data_two is not None and np.any(~np.isnan(data_two))
    have_both     = have_data_one and have_data_two
    # determine if using datetimes
    use_datetime = False # TODO: test with these

    #% Calculations
    # find overlapping times
    (time_overlap, d1_diff_ix, d2_diff_ix) = intersect(time_one, time_two) # TODO: add a tolerance?
    # find differences
    d1_miss_ix = np.setxor1d(np.arange(len(time_one)), d1_diff_ix)
    d2_miss_ix = np.setxor1d(np.arange(len(time_two)), d2_diff_ix)
    # build RMS indices
    rms_ix1  = (time_one >= rms_xmin) & (time_one <= rms_xmax)
    rms_ix2  = (time_two >= rms_xmin) & (time_two <= rms_xmax)
    rms_ix3  = (time_overlap >= rms_xmin) & (time_overlap <= rms_xmax)
    rms_pts1 = np.maximum(rms_xmin, np.minimum(np.min(time_one), np.min(time_two)))
    rms_pts2 = np.minimum(rms_xmax, np.maximum(np.max(time_one), np.max(time_two)))
    # find number of elements being differenced
    num_channels = len(elements)
    cm = ColorMap(colormap=colormap, num_colors=3*num_channels)
    # calculate the differences
    if have_both:
        nondeg_error = data_two[:, d2_diff_ix] - data_one[:, d1_diff_ix]
    # calculate the rms (or mean) values
    nans = np.full(3, np.nan, dtype=float)
    if not use_mean:
        func_name   = 'RMS'
        data1_func  = rms(data_one[:, rms_ix1], axis=1, ignore_nans=True) if have_data_one else nans
        data2_func  = rms(data_two[:, rms_ix2], axis=1, ignore_nans=True) if have_data_two else nans
        nondeg_func = rms(nondeg_error[:, rms_ix3], axis=1, ignore_nans=True) if have_both else nans
    else:
        func_name   = 'Mean'
        data1_func  = np.nanmean(data_one[:, rms_ix1], axis=1) if have_data_one else nans
        data2_func  = np.nanmean(data_two[:, rms_ix2], axis=1) if have_data_two else nans
        nondeg_func = np.nanmean(nondeg_error[:, rms_ix3], axis=1) if have_both else nans
    # output errors
    err = {'one': data1_func, 'two': data2_func, 'diff': nondeg_func}
    # unit conversion value
    (temp, prefix) = get_factors(leg_scale)
    leg_conv = 1/temp
    # determine which symbols to plot with
    if have_both:
        symbol_one = '^-'
        symbol_two = 'v:'
    elif have_data_one:
        symbol_one = '.-'
        symbol_two = '' # not-used
    elif have_data_two:
        symbol_one = '' # not-used
        symbol_two = '.-'
    else:
        symbol_one = '' # invalid case
        symbol_two = '' # invalid case
    # pre-plan plot layout
    if have_both:
        if make_subplots:
            num_figs = 1
            if single_lines:
                num_rows = num_channels
                num_cols = 2
            else:
                num_rows = 2
                num_cols = 1
        else:
            num_figs = 2
            num_cols = 1
            if single_lines:
                num_rows = num_channels
            else:
                num_rows = 1
    else:
        num_figs = 1
        if single_lines:
            num_rows = num_channels
            num_cols = 1
        else:
            num_rows = 1
            num_cols = 1
    num_axes = num_figs*num_rows*num_cols

    #% Create plots
    # create figures
    f1 = plt.figure()
    f1.canvas.set_window_title(description) # TODO: fig_visible (in wrapper)?
    if have_both and not make_subplots:
        f2 = plt.figure()
        f2.canvas.set_window_title(description + 'Difference')
        fig_hand = [f1, f2]
    else:
        fig_hand = [f1]
    # create axes
    ax = []
    ax_prim = None
    for i in range(num_figs):
        for j in range(num_cols):
            for k in range(num_rows):
                temp_axes = fig_hand[i].add_subplot(num_rows, num_cols, k*num_cols + j + 1, sharex=ax_prim)
                if ax_prim is None:
                    ax_prim = temp_axes
                ax.append(temp_axes)
    assert num_axes == len(ax), 'There is a mismatch in the number of axes.'
    # plot data
    for (i, this_axes) in enumerate(ax):
        is_diff_plot = i > num_rows-1 or (not single_lines and make_subplots and i == 1)
        if single_lines:
            if is_diff_plot:
                loop_counter = [i - num_rows]
            else:
                loop_counter = [i]
        else:
            loop_counter = range(num_channels)
        if not is_diff_plot:
            # standard plot
            if have_data_one:
                for j in loop_counter:
                    if show_rms:
                        value = leg_format.format(leg_conv*data1_func[j])
                        this_label = '{} {} ({}: {} {}{})'.format(name_one, elements[j], func_name, value, prefix, units)
                    else:
                        this_label = name_one + ' ' + elements[j]
                    this_axes.plot(time_one, data_one[j, :], symbol_one, markersize=4, label=this_label, \
                        color=cm.get_color(j), zorder=3)
            if have_data_two:
                for j in loop_counter:
                    if show_rms:
                        value = leg_format.format(leg_conv*data2_func[j])
                        this_label = '{} {} ({}: {} {}{})'.format(name_two, elements[j], func_name, value, prefix, units)
                    else:
                        this_label = name_two + ' ' + elements[j]
                    this_axes.plot(time_two, data_two[j, :], symbol_two, markersize=4, label=this_label, \
                        color=cm.get_color(j+num_channels), zorder=5)
        else:
            #% Difference plot
            for j in loop_counter:
                if single_lines and i % num_channels != j:
                    continue
                if show_rms:
                    value = leg_format.format(leg_conv*nondeg_func[j])
                    this_label = '{} ({}: {}) {}{})'.format(elements[j], func_name, value, prefix, units)
                else:
                    this_label = elements[j]
                this_axes.plot(time_overlap, nondeg_error[j, :], '.-', markersize=4, label=this_label, \
                    color=cm.get_color(j+2*num_channels))
            if show_extra:
                this_axes.plot(time_one[d1_miss_ix], np.zeros(len(d1_miss_ix)), 'kx', markersize=8, markeredgewidth=2, markerfacecolor='None', label=name_one + ' Extra')
                this_axes.plot(time_one[d2_miss_ix], np.zeros(len(d2_miss_ix)), 'go', markersize=8, markeredgewidth=2, markerfacecolor='None', label=name_two + ' Extra')

        # set X display limits
        if i == 0:
            xlim = list(this_axes.get_xlim())
            if np.isfinite(disp_xmin):
                xlim[0] = max([xlim[0], disp_xmin])
            if np.isfinite(disp_xmax):
                xlim[1] = min([xlim[1], disp_xmax])
        this_axes.set_xlim(xlim)
        # set Y display limits
        if plot_zero:
            show_zero_ylim(this_axes)
        # optionally plot truth (after having set axes limits)
        if i < num_rows and truth_time is not None and truth_data is not None and not np.all(np.isnan(truth_data)):
            if single_lines:
                this_axes.plot(truth_time, truth_data[i, :], '.-', color=truth_color, markerfacecolor=truth_color, \
                    linewidth=2, label=truth_name + ' ' + elements[i])
            else:
                if i == 0:
                    # TODO: add RMS to Truth data?
                    this_axes.plot(truth_time, truth_data[i, :], '.-', color=truth_color, markerfacecolor=truth_color, \
                        linewidth=2, label=truth_name)
        # format display of plot
        this_axes.legend(loc=legend_loc)
        if i == 0:
            this_axes.set_title(description)
        elif (single_lines and i == num_rows) or (not single_lines and i == 1):
            this_axes.set_title(description + ' Difference')
        if use_datetime:
            this_axes.set_xlabel('Date')
        else:
            this_axes.set_xlabel('Time [' + time_units + ']' + start_date)
        if y_label is None:
            if is_diff_plot:
                this_axes.set_ylabel(description + ' Difference [' + units + ']')
            else:
                this_axes.set_ylabel(description + ' [' + units + ']')
        else:
            # TODO: handle single_lines case by allowing list for y_label
            ix = y_label.find('[')
            if is_diff_plot and ix > 0:
                this_axes.set_ylabel(y_label[:ix-1] + 'Difference ' + y_label[ix:])
            else:
                this_axes.set_ylabel(y_label)
        this_axes.grid(True)
        # optionally add second Y axis
        if second_y_scale is not None:
            if isinstance(second_y_scale, (int, float)):
                if not np.isnan(second_y_scale) and second_y_scale != 0:
                    plot_second_yunits(this_axes, '', second_y_scale)
            else:
                for (key, value) in second_y_scale.items():
                    if not np.isnan(value) and value != 0:
                        plot_second_yunits(this_axes, description + ' Difference [' + key + ']', value)
        # plot RMS lines
        if show_rms:
            plot_rms_lines(this_axes, [rms_pts1, rms_pts2], this_axes.get_ylim())

    return (fig_hand, err)

#%% plot_phases
def plot_phases(ax, times, colormap='tab10', labels=None):
    r"""
    Plots some labeled phases as semi-transparent patchs on the given axis.

    Parameters
    ----------
    ax : (Axes)
        Figure axes
    times : (1xN) or (2xN) list of times, if it has two rows, then the second are the end points
         otherwise assume the sections are continuous.

    Examples
    --------
    >>> from dstauffman import plot_phases, get_color_lists
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig = plt.figure()
    >>> fig.canvas.set_window_title('Sine Wave')
    >>> ax = fig.add_subplot(111)
    >>> time = np.arange(101)
    >>> data = np.cos(time / 10)
    >>> _ = ax.plot(time, data, '.-')
    >>> times = np.array([5, 20, 60, 90])
    >>> # times = np.array([[5, 20, 60, 90], [10, 60, 90, 95]])
    >>> labels = ['Part 1', 'Phase 2', 'Watch Out', 'Final']
    >>> colorlists = get_color_lists()
    >>> colors = colorlists['quat']
    >>> plot_phases(ax, times, colors, labels)
    >>> plt.show(block=False) # doctest: +SKIP

    Close plot
    >>> plt.close(fig)

    """
    # hard-coded values
    transparency = 0.2 # 1 = opaque

    # get number of segments
    if times.ndim == 1:
        num_segments = times.size
    else:
        num_segments = times.shape[1]

    # check for optional arguments
    cm = ColorMap(colormap=colormap, num_colors=num_segments)

    # get the limits of the plot
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    # create second row of times if not specified (assumes each phase goes all the way to the next one)
    if times.ndim == 1:
        times = np.vstack((times, np.hstack((times[1:], max(times[-1], xlims[1])))))

    # loop through all the phases
    for i in range(num_segments):
        # get the label and color for this phase
        this_color = cm.get_color(i)
        # get the locations for this phase
        x1 = times[0, i]
        x2 = times[1, i]
        y1 = ylims[0]
        y2 = ylims[1]
        # create the shaded box
        ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, facecolor=this_color, edgecolor=this_color, \
            alpha=transparency))
        # create the label
        if labels is not None:
            ax.annotate(labels[i], xy=(x1, y2), \
                xycoords='data', horizontalalignment='left', verticalalignment='top', \
                fontsize=15, rotation=-90)

    # reset any limits that might have changed due to the patches
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

#%% Unit test
if __name__ == '__main__':
    plt.ioff()
    unittest.main(module='dstauffman.tests.test_plotting', exit=False)
    doctest.testmod(verbose=False)
