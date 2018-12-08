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
import gc
import os
import platform
import sys
import unittest
import warnings

# plotting/numpy imports
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.ticker import StrMethodFormatter

# Qt imports
try:
    from PyQt5.QtWidgets import QApplication, QPushButton
    from PyQt5.QtGui import QIcon
    from PyQt5.QtCore import QSize
except ImportError: # pragma: no cover
    warnings.warn('PyQt5 was not found. Some funtionality will be limited.')
    QPushButton = object

# model imports
from dstauffman.classes import Frozen
from dstauffman.constants import DEFAULT_COLORMAP
from dstauffman.latex import bins_to_str_ranges
from dstauffman.stats import z_from_ci
from dstauffman.utils import get_images_dir, pprint_dict, rms

#%% Classes - _HoverButton
class _HoverButton(QPushButton):
    r"""Custom button that allows hovering and icons."""
    def __init__(self, *args, **kwargs):
        # initialize
        super().__init__(*args, **kwargs)
        # Enable mouse hover event tracking
        self.setMouseTracking(True)
        self.setStyleSheet('border: 0px;')
        # set icon
        for this_arg in args:
            if isinstance(this_arg, QIcon):
                self.setIcon(this_arg)
                self.setIconSize(QSize(24, 24))

    def enterEvent(self, event):
        # Draw border on hover
        self.setStyleSheet('border: 1px; border-style: solid;') # pragma: no cover

    def leaveEvent(self, event):
        # Delete border after hover
        self.setStyleSheet('border: 0px;') # pragma: no cover

#%% Classes - Plotter
class Plotter(Frozen):
    r"""
    Class that allows customization of when to show or not show plots.

    For use with testing plotting functions.
    """
    # class attribute for plotting flag
    show_plot = True

    def __init__(self, show=None):
        r"""Create options instance with ability to override defaults."""
        if show is not None:
            type(self).show_plot = bool(show)

    def __str__(self):
        r"""Print the current plotting flag."""
        return '{}({})'.format(type(self).__name__, self.show_plot)

    @classmethod
    def get_plotter(cls):
        r"""Get the plotting flag."""
        return cls.show_plot

    @classmethod
    def set_plotter(cls, show):
        r"""Set the plotting flag."""
        cls.show_plot = bool(show)

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

#%% Classes - TruthPlotter
class TruthPlotter(Frozen):
    r"""
    Class wrapper for the different types of truth data to include on plots.

    Examples
    --------
    >>> from dstauffman import TruthPlotter
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig = plt.figure()
    >>> fig.canvas.set_window_title('Figure Title')
    >>> x = np.arange(0, 10, 0.1)
    >>> y = np.sin(x)
    >>> truth = TruthPlotter(x, y+0.01, lo=y, hi=y+0.03)
    >>> ax = fig.add_subplot(111)
    >>> _ = ax.plot(x, y, label='data')
    >>> truth.plot_truth(ax)
    >>> _ = ax.legend(loc='best')

    >>> plt.show(block=False) # doctest: +SKIP

    Close plot
    >>> plt.close(fig)

    """
    def __init__(self, time=None, data=None, lo=None, hi=None, type_='normal', name='Observed'):
        self.time    = time
        self.data    = None
        self.type_   = type_ # from {'normal', 'errorbar'}
        self.data_lo = lo
        self.data_hi = hi
        self.name    = name
        # TODO: keep this old API? (otherwise just: self.data = data)
        if data is not None:
            if data.ndim == 1:
                self.data = data
            elif data.shape[1] == 1:
                self.data = data[:, 0]
            elif data.shape[1] == 3:
                self.data    = data[:, 1]
                self.data_lo = data[:, 0]
                self.data_hi = data[:, 2]
            else:
                raise ValueError('Bad shape for data of {}.'.format(data.shape))

    def pprint(self, indent=1, align=True):
        r"""Display a pretty print version of the class."""
        pprint_dict(self.__dict__, name=self.__class__.__name__, indent=indent, align=align)

    @property
    def is_null(self):
        r"""Determine if there is no truth to plot, and thus nothing is done."""
        return self.data is None and self.data_lo is None and self.data_hi is None

    @staticmethod
    def get_data(data, scale=1, ix=None):
        r"""Scale and index the data, returning None if it is not there."""
        if data is None:
            return data
        if ix is None:
            return scale * data
        else:
            return scale * data[:, ix]

    def plot_truth(self, ax, scale=1, ix=None, *, hold_xlim=True, hold_ylim=False):
        r"""Add the information in the TruthPlotter instance to the given axis, with the optional scale."""
        # check for null case
        if self.is_null:
            return
        # get original limits
        x_lim = ax.get_xbound()
        y_lim = ax.get_ybound()
        # plot the new data
        this_data = self.get_data(self.data, scale, ix)
        if this_data is not None and not np.all(np.isnan(this_data)):
            ax.plot(self.time, this_data, 'k.-', linewidth=2, zorder=8, label=self.name)
        if self.type_ == 'normal':
            this_data = self.get_data(self.data_lo, scale, ix)
            if this_data is not None and not np.all(np.isnan(this_data)):
                ax.plot(self.time, this_data, '.-', color='0.5', linewidth=2, zorder=6)
            this_data = self.get_data(self.data_hi, scale, ix)
            if self.data_hi is not None and not np.all(np.isnan(this_data)):
                ax.plot(self.time, this_data, '.-', color='0.5', linewidth=2, zorder=6)
        elif self.type_ == 'errorbar':
            if self.data_lo is not None and self.data_hi is not None:
                if ix is None:
                    yerr = np.vstack((self.data-self.data_lo, self.data_hi-self.data))
                    ax.errorbar(self.time, scale*self.data, scale*yerr, linestyle='None', \
                        marker='None', ecolor='c', zorder=6)
                else:
                    yerr = np.vstack((self.data[:, ix]-self.data_lo[:, ix], self.data_hi[:, ix]-self.data[:, ix])).T
                    ax.errorbar(self.time, scale*self.data[:, ix], scale*yerr[:, ix], linestyle='None', \
                        marker='None', ecolor='c', zorder=6)
        else:
            raise ValueError('Unexpected value for type_ of "{}".'.format(self.type_))
        # potentially restore the original limits, since they might have been changed by the truth data
        if hold_xlim and x_lim != ax.get_xbound():
            ax.set_xbound(*x_lim)
        if hold_ylim and y_lim != ax.get_ybound():
            ax.set_ybound(*y_lim)

#%% Classes - MyCustomToolbar
class MyCustomToolbar():
    r"""
    Define a custom toolbar to use in any matplotlib plots.

    Examples
    --------
    >>> from dstauffman import MyCustomToolbar
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig = plt.figure()
    >>> fig.canvas.set_window_title('Figure Title')
    >>> ax = fig.add_subplot(111)
    >>> x = np.arange(0, 10, 0.1)
    >>> y = np.sin(x)
    >>> _ = ax.plot(x, y)
    >>> fig.toolbar_custom_ = MyCustomToolbar(fig)

    Close plot
    >>> plt.close(fig)

    """
    def __init__(self, fig):
        r"""Initialize the custom toolbar."""
        # check to see if a QApplication exists, and if not, make one
        if QApplication.instance() is None:
            self.qapp = QApplication(sys.argv) # pragma: no cover
        else:
            self.qapp = QApplication.instance()
        # Store the figure number for use later (Note this works better than relying on plt.gcf()
        # to determine which figure actually triggered the button events.)
        self.fig_number = fig.number
        # create buttons - Prev Plot
        icon = QIcon(os.path.join(get_images_dir(), 'prev_plot.png'))
        self.btn_prev_plot = _HoverButton(icon, '')
        self.btn_prev_plot.setToolTip('Show the previous plot')
        fig.canvas.toolbar.addWidget(self.btn_prev_plot)
        self.btn_prev_plot.clicked.connect(self.prev_plot)
        # create buttons - Next Plot
        icon = QIcon(os.path.join(get_images_dir(), 'next_plot.png'))
        self.btn_next_plot = _HoverButton(icon, '')
        self.btn_next_plot.setToolTip('Show the next plot')
        fig.canvas.toolbar.addWidget(self.btn_next_plot)
        self.btn_next_plot.clicked.connect(self.next_plot)
        # create buttons - Close all
        icon = QIcon(os.path.join(get_images_dir(), 'close_all.png'))
        self.btn_close_all = _HoverButton(icon, '')
        self.btn_close_all.setToolTip('Close all the open plots')
        fig.canvas.toolbar.addWidget(self.btn_close_all)
        self.btn_close_all.clicked.connect(self._close_all)

    def _close_all(self, *args):
        r"""Close all the currently open plots."""
        close_all()

    def next_plot(self, *args):
        r"""Bring up the next plot in the series."""
        # get all the figure numbers
        all_figs = plt.get_fignums()
        # get the active figure number
        this_fig = self.fig_number
        # loop through all the figures
        for i in range(len(all_figs)):
            # find the active figure within the list
            if this_fig == all_figs[i]:
                # find the next figure, with allowances for rolling over the list
                if i < len(all_figs)-1:
                    next_fig = all_figs[i+1]
                else:
                    next_fig = all_figs[0]
        # set the appropriate active figure
        fig = plt.figure(next_fig)
        # make it the active window
        fig.canvas.manager.window.raise_()

    def prev_plot(self, *args):
        r"""Bring up the previous plot in the series."""
        # get all the figure numbers
        all_figs = plt.get_fignums()
        # get the active figure number
        this_fig = self.fig_number
        # loop through all the figures
        for i in range(len(all_figs)):
            # find the active figure within the list
            if this_fig == all_figs[i]:
                # find the next figure, with allowances for rolling over the list
                if i > 0:
                    prev_fig = all_figs[i-1]
                else:
                    prev_fig = all_figs[-1]
        # set the appropriate active figure
        fig = plt.figure(prev_fig)
        # make it the active window
        fig.canvas.manager.window.raise_()

#%% Classes - ColorMap
class ColorMap(Frozen):
    r"""
    Colormap class for easier setting of colormaps in matplotlib.

    Parameters
    ----------
    colormap : str, optional
        Name of the colormap to use
    low : int, optional
        Low value to use as an index to a specific color within the map
    high : int, optional
        High value to use as an index to a specific color within the map
    num_colors : int, optional
        If not None, then this replaces the low and high inputs

    Notes
    -----
    #.  Written by David C. Stauffer in July 2015.

    Examples
    --------
    >>> from dstauffman import ColorMap
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> cm = ColorMap('Paired', 1, 2)
    >>> time = np.arange(0, 10, 0.1)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> _ = ax.plot(time, np.sin(time), color=cm.get_color(1))
    >>> _ = ax.plot(time, np.cos(time), color=cm.get_color(2))
    >>> _ = ax.legend(['Sin', 'Cos'])
    >>> plt.show(block=False) # doctest: +SKIP

    Close plot
    >>> plt.close(fig)

    """
    def __init__(self, colormap=DEFAULT_COLORMAP, low=0, high=1, num_colors=None):
        self.num_colors = num_colors
        # check for optional inputs
        if self.num_colors is not None:
            low = 0
            high = num_colors-1
        # get colormap based on high and low limits
        if isinstance(colormap, colors.Colormap):
            cmap = colormap
        else:
            cmap = plt.get_cmap(colormap)
        cnorm = colors.Normalize(vmin=low, vmax=high)
        self.smap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)
        # must initialize the empty scalar mapplable to show the colorbar correctly
        self.smap.set_array([])

    def get_color(self, value):
        r"""Get the color based on the scalar value."""
        return self.smap.to_rgba(value)

    def get_smap(self):
        r"""Return the smap being used."""
        return self.smap

    def set_colors(self, ax):
        r"""Set the colors for the given axis based on internal instance information."""
        if self.num_colors is None:
            raise ValueError("You can't call ColorMap.set_colors unless it was given a num_colors input.")
        try:
            ax.set_prop_cycle('color', [self.get_color(i) for i in range(self.num_colors)])
        except AttributeError: # pragma: no cover
            # for older matplotlib versions, use deprecated set_color_cycle
            ax.set_color_cycle([self.get_color(i) for i in range(self.num_colors)])

#%% Functions - ignore_plot_data
def ignore_plot_data(data, ignore_empties, col=None):
    r"""
    Determine whether to ignore this data or not.

    Parameters
    ----------
    data : (N, M) ndarray
        Data to plot or ignore
    ignore_empties : bool
        Whether to potentially ignore empties or not
    col : int, optional
        Column number to look at to determine if ignoring, if not present, look at entire matrix

    Returns
    -------
    ignore : bool
        Whether data is null (all zeros/nans) and should be ignored.

    Notes
    -----
    #.  Written by David C. Stauffer in April 2017.

    Examples
    --------
    >>> from dstauffman import ignore_plot_data
    >>> import numpy as np
    >>> data = np.zeros((3, 10), dtype=float)
    >>> ignore_empties = True
    >>> col = 2
    >>> ignore = ignore_plot_data(data, ignore_empties, col)
    >>> print(ignore)
    True

    """
    # if data is None, then always ignore it
    if data is None:
        return True
    # if we are not ignoring empties and data is not None, then never ignore
    if not ignore_empties:
        return False
    # otherwise determine if ignoring by seeing if data is all zeros or nans
    if col is None:
        ignore = np.all((data == 0) | np.isnan(data))
    else:
        ignore = np.all((data[:, col] == 0) | np.isnan(data[:, col]))
    return ignore

#%% Functions - close_all
def close_all(figs=None):
    r"""
    Close all the open figures, or if a list is specified, then close all of them.

    Parameters
    ----------
    figs : list of Figure, optional
        Specific figures to be closed.

    Examples
    --------
    >>> from dstauffman import close_all
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> _ = ax.plot(0, 0)
    >>> close_all([fig])

    """
    # Note that it's better to loop through and close the plots individually than to use
    # plt.close('all'), as that can sometimes cause the iPython kernel to quit #DCS: 2015-06-11
    if figs is None:
        for this_fig in plt.get_fignums():
            plt.close(this_fig)
    else:
        for this_fig in figs:
            plt.close(this_fig)
    gc.collect()

#%% Functions - get_axes_scales
def get_axes_scales(type_):
    r"""
    Determine the scale factor and units to apply to the plot based on the desired `type_`.

    Parameters
    ----------
    type_ : str {'unity', 'population', 'percentage', 'per 100K', 'cost', 'years'}
        description of the type of data that is being plotted

    Returns
    -------
    scale : int or float
        Scale factor to multiply the raw values by
    units : str
        Units string to apply to the plot axis label

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.

    Examples
    --------
    >>> from dstauffman import get_axes_scales
    >>> type_ = 'percentage'
    >>> (scale, units) = get_axes_scales(type_)

    >>> print(scale)
    100
    >>> print(units)
    %

    """
    # determine results based on simple switch statement
    if type_.startswith('unit_'): # TODO: make more generic and allow vert_fact to work
        scale = 1
        units = type_[5:]
    elif type_ == 'unity':
        scale = 1
        units = ''
    elif type_ == 'population':
        scale = 1
        units = '#'
    elif type_ == 'percentage':
        scale = 100
        units = '%'
    elif type_ == 'per 1K':
        scale = 1000
        units = 'per 1,000'
    elif type_ == 'per 100K':
        scale = 100000
        units = 'per 100,000'
    elif type_ == 'cost':
        scale = 1e-3
        units = "$K's"
    elif type_ == 'years':
        scale = 1
        units = 'years'
    else:
        raise ValueError('Unexpected data type_ "{}" for plot.'.format(type_))
    return (scale, units)

#%% Functions - whitten
def whitten(color, white=(1, 1, 1, 1), dt=0.30):
    r"""
    Shift an RGBA color towards white.

    Parameters
    ----------
    color : tuple
        Given color
    white : tuple, optional
        Color to *whitten* towards, usually assumed to be white
    dt : float, optional
        Amount to move towards white, from 0 (none) to 1 (all the way), default is 0.3

    Examples
    --------
    >>> from dstauffman import whitten
    >>> color = (1, 0.4, 0)
    >>> new_color = whitten(color)
    >>> print(new_color)
    (1.0, 0.58, 0.3)

    """
    # apply the shift
    new_color = tuple((c*(1-dt) + w*dt for (c, w) in zip(color, white)))
    return new_color

#%% Functions - plot_time_history
def plot_time_history(time, data, label, type_='unity', opts=None, *, plot_indiv=True, \
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
    type_ : str, optional, from {'unity', 'population', 'percentage', 'per 100K', 'cost'}
        description of the type of data that is being plotted, default is 'unity'
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
    >>> from dstauffman import plot_time_history
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> time  = np.arange(0, 10, 0.1)
    >>> data  = np.sin(time)
    >>> label = 'Sin'
    >>> type_ = 'population'
    >>> fig   = plot_time_history(time, data, label, type_)

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

    # determine which type of data to plot
    (scale, units) = get_axes_scales(type_)
    unit_text = ' [' + units + ']' if units else ''

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
def plot_correlation_matrix(data, labels=None, type_='unity', opts=None, *, matrix_name='Correlation Matrix', \
        cmin=0, cmax=1, xlabel='', ylabel='', plot_lower_only=True, label_values=False, x_lab_rot=90, \
        colormap=None):
    r"""
    Visually plot a correlation matrix.

    Parameters
    ----------
    data : array_like
        data for corresponding time history
    labels : list of str, optional
        Names to put on row and column headers
    type_ : str, optional, from {'unity', 'population', 'percentage', 'per 100K', 'cost'}
        description of the type of data that is being plotted, default is 'unity'
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
    (scale, units) = get_axes_scales(type_)
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
                        color=cm.get_color(scale*data[j, i])))
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

#%% Functions - plot_multiline_history
def plot_multiline_history(time, data, label, type_='unity', opts=None, *, legend=None, \
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
    type_ : str, optional, from {'unity', 'population', 'percentage', 'per 100K', 'cost'}
        description of the type of data that is being plotted, default is 'unity'
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
    >>> from dstauffman import plot_multiline_history
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> time  = np.arange(0, 5, 1./12) + 2000
    >>> data  = np.random.rand(len(time), 5)
    >>> mag   = data.cumsum(axis=1)[:,-1]
    >>> data  = 10 * data / np.expand_dims(mag, axis=1)
    >>> label = 'Random Data'
    >>> fig   = plot_multiline_history(time, data, label)

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
    assert time.shape[0] == data.shape[0], 'Time and data must be the same length.'
    if legend is not None:
        assert len(legend) == num_bins, 'Number of data channels does not match the legend.'
    else:
        legend = ['Channel {}'.format(i+1) for i in range(num_bins)]

    # process other inputs
    this_title = label + ' vs. Time'
    (scale, units) = get_axes_scales(type_)
    unit_text = ' [' + units + ']' if units else ''

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

#%% Functions - storefig
def storefig(fig, folder=None, plot_type='png'):
    r"""
    Store the specified figures in the specified folder and with the specified plot type(s).

    Parameters
    ----------
    fig : list or single figure
        Figure object(s) to save to disk
    folder : str
        Location to save figures to
    plot_type : str
        Type of figure to save to disk, like 'png' or 'jpg'

    Raises
    ------
    ValueError
        Specified folder to save figures to doesn't exist.

    Notes
    -----
    #. Uses the figure.canvas.get_window_title property to determine the figure name.

    See Also
    --------
    matplotlib.pyplot.savefig, titleprefix

    Examples
    --------
    Create figure and then save to disk

    >>> from dstauffman import storefig
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> import os
    >>> fig = plt.figure()
    >>> fig.canvas.set_window_title('Figure Title')
    >>> ax = fig.add_subplot(111)
    >>> x = np.arange(0, 10, 0.1)
    >>> y = np.sin(x)
    >>> _ = ax.plot(x, y)
    >>> _ = ax.set_title('X vs Y')
    >>> plt.show(block=False) # doctest: +SKIP
    >>> folder = os.getcwd()
    >>> plot_type = 'png'
    >>> storefig(fig, folder, plot_type)

    Close plot
    >>> plt.close(fig)

    Delete file
    >>> os.remove(os.path.join(folder, 'Figure Title.png'))

    """
    # hard-coded values
    bad_chars_win  = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    bad_chars_unix = ['/']
    is_windows     = platform.system() == 'Windows'
    bad_chars      = bad_chars_win if is_windows else bad_chars_unix
    # make sure figs is a list
    if isinstance(fig, list):
        figs = fig
    else:
        figs = [fig]
    # make sure types is a list
    if not isinstance(plot_type, list):
        types = []
        types.append(plot_type)
    else:
        types = plot_type
    # if no folder was specified, then use the current working directory
    if folder is None:
        folder = os.getcwd() # pragma: no cover
    # confirm that the folder exists
    if not os.path.isdir(folder):
        raise ValueError('The specfied folder "{}" does not exist.'.format(folder))
    # loop through the figures
    for this_fig in figs:
        # get the title of the figure canvas
        this_title = this_fig.canvas.get_window_title()
        # replace any bad characters with underscores
        for ch in bad_chars:
            if ch in this_title:
                this_title = this_title.replace(ch, '_')
        # loop through the plot types
        for this_type in types:
            # save the figure to the specified plot type
            this_fig.savefig(os.path.join(folder, this_title + '.' + this_type), dpi=160, bbox_inches='tight')

#%% Functions - titleprefix
def titleprefix(fig, prefix=''):
    r"""
    Prepend a text string to all the titles on existing figures.

    It also sets the canvas title used by storefig when saving to a file.

    Parameters
    ----------
    fig : list or single figure
        Figure object(s) to save to disk
    prefix : str
        Text to be prepended to the title and figure name

    Notes
    -----
    #.  Written by David C. Stauffer in March 2015.
    #.  Desired this function to also check for suptitles.

    See Also
    --------
    storefig

    Examples
    --------
    Create figure and then change the title
    >>> from dstauffman import titleprefix
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig = plt.figure()
    >>> fig.canvas.set_window_title('Figure Title')
    >>> ax = fig.add_subplot(111)
    >>> x = np.arange(0, 10, 0.1)
    >>> y = np.sin(x)
    >>> _ = ax.plot(x, y)
    >>> _ = ax.set_title('X vs Y')
    >>> plt.show(block=False) # doctest: +SKIP
    >>> prefix = 'Baseline'
    >>> titleprefix(fig, prefix)
    >>> plt.draw() # doctest: +SKIP

    Close plot
    >>> plt.close(fig)

    """
    # check for non-empty prefix
    if not prefix:
        return
    # force figs to be a list
    if isinstance(fig, list):
        figs = fig
    else:
        figs = [fig]
    # loop through figures
    for this_fig in figs:
        # get axis list and loop through them
        for this_axis in this_fig.axes:
            # get title for this axis
            this_title = this_axis.get_title()
            # if the title is empty, then don't do anything
            if not this_title:
                continue
            # modify and set new title
            new_title = prefix + ' - ' + this_title
            this_axis.set_title(new_title)
        # update canvas name
        this_canvas_title = this_fig.canvas.get_window_title()
        this_fig.canvas.set_window_title(prefix + ' - ' + this_canvas_title)

#%% Functions - disp_xlimits
def disp_xlimits(figs, xmin=None, xmax=None):
    r"""
    Set the xlimits to the specified xmin and xmax.

    Parameters
    ----------
    figs : array_like
        List of figures
    xmin : scalar
        Minimum X value
    xmax : scalar
        Maximum X value

    Notes
    -----
    #.  Written by David C. Stauffer in August 2015.

    Examples
    --------
    >>> from dstauffman import disp_xlimits
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig = plt.figure()
    >>> fig.canvas.set_window_title('Figure Title')
    >>> ax = fig.add_subplot(111)
    >>> x = np.arange(0, 10, 0.1)
    >>> y = np.sin(x)
    >>> _ = ax.plot(x, y)
    >>> _ = ax.set_title('X vs Y')
    >>> plt.show(block=False) # doctest: +SKIP
    >>> xmin = 2
    >>> xmax = 5
    >>> disp_xlimits(fig, xmin, xmax)
    >>> plt.draw() # doctest: +SKIP

    Close plot
    >>> plt.close(fig)

    """
    # check for single figure
    if not isinstance(figs, list):
        figs = [figs]
    # loop through figures
    for this_fig in figs:
        # get axis list and loop through them
        for this_axis in this_fig.axes:
            # get xlimits for this axis
            (old_xmin, old_xmax) = this_axis.get_xlim()
            # set the new limits
            if xmin is not None:
                new_xmin = np.max([xmin, old_xmin])
            else:
                new_xmin = old_xmin
            if xmax is not None:
                new_xmax = np.min([xmax, old_xmax])
            else:
                new_xmax = old_xmax
            # modify xlimits
            this_axis.set_xlim((new_xmin, new_xmax))

#%% Functions - setup_plots
def setup_plots(figs, opts, plot_type='time'):
    r"""
    Combine common plot operations into one easy command.

    Parameters
    ----------
    figs : array_like
        List of figures
    opts : class Opts
        Optional plotting controls
    plot_type : optional, {'time', 'time_no_yscale', 'dist', 'dist_no_yscale'}

    Notes
    -----
    #.  Written by David C. Stauffer in May 2015.

    Examples
    --------
    >>> from dstauffman import setup_plots, Opts
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig = plt.figure()
    >>> fig.canvas.set_window_title('Figure Title')
    >>> ax = fig.add_subplot(111)
    >>> x = np.arange(0, 10, 0.1)
    >>> y = np.sin(x)
    >>> _ = ax.plot(x, y)
    >>> _ = ax.set_title('X vs Y')
    >>> _ = ax.set_xlabel('time [years]')
    >>> _ = ax.set_ylabel('value [radians]')
    >>> plt.show(block=False) # doctest: +SKIP
    >>> opts = Opts()
    >>> opts.case_name = 'Testing'
    >>> opts.show_plot = True
    >>> opts.save_plot = False
    >>> setup_plots(fig, opts)

    Close plots
    >>> plt.close(fig)

    """
    # check for single figure
    if not isinstance(figs, list):
        figs = [figs]

    # prepend a title
    if opts.case_name:
        titleprefix(figs, opts.case_name)

    # change the display range
    if plot_type in {'time', 'time_no_yscale'}:
        disp_xlimits(figs, opts.disp_xmin, opts.disp_xmax)

    # things to do if displaying the plots
    if opts.show_plot and Plotter.show_plot: # pragma: no cover
        # add a custom toolbar
        figmenu(figs)
        # show the plot
        plt.show(block=False)

    # optionally save the plot
    if opts.save_plot:
        storefig(figs, opts.save_path, opts.plot_type)
        if opts.show_link & len(figs) > 0:
            print(r'Plots saved to <a href="{}">{}</a>'.format(opts.save_path, opts.save_path))

#%% Functions - figmenu
def figmenu(figs):
    r"""
    Add a custom toolbar to the figures.

    Parameters
    ----------
    figs : class matplotlib.pyplot.Figure, or list of such
        List of figures

    Examples
    --------
    >>> from dstauffman import figmenu
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig = plt.figure()
    >>> fig.canvas.set_window_title('Figure Title')
    >>> ax = fig.add_subplot(111)
    >>> x = np.arange(0, 10, 0.1)
    >>> y = np.sin(x)
    >>> _ = ax.plot(x, y)
    >>> _ = ax.set_title('X vs Y')
    >>> _ = ax.set_xlabel('time [years]')
    >>> _ = ax.set_ylabel('value [radians]')
    >>> plt.show(block=False) # doctest: +SKIP
    >>> figmenu(fig)

    Close plot
    >>> plt.close(fig)

    """
    if not isinstance(figs, list):
        figs.toolbar_custom_ = MyCustomToolbar(figs)
    else:
        for i in range(len(figs)):
            figs[i].toolbar_custom_ = MyCustomToolbar(figs[i])

#%% rgb_ints_to_hex
def rgb_ints_to_hex(int_tuple):
    r"""
    Convert a tuple of ints with (0, 255) to the equivalent hex color code.

    Parameters
    ----------
    int_tuple : (3-tuple) of int
        RGB Integer code colors

    Returns
    -------
    hex_code : str
        Hexidecimal color code

    Examples
    --------
    >>> from dstauffman import rgb_ints_to_hex
    >>> hex_code = rgb_ints_to_hex((79, 129, 189))
    >>> print(hex_code)
    #4f81bd

    """
    def clamp(x, min_=0, max_=255):
        r"""Clamps a value within the specified minimum and maximum."""
        return max(min_, min(x, max_))

    (r, g, b) = int_tuple
    hex_code = "#{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b))
    return hex_code

#%% Functions - get_screen_resolution
def get_screen_resolution():
    r"""
    Gets the current monitor screen resolution.

    Returns
    -------
    screen_width : int
        Screen width in pixels
    screen_heigh : int
        Screen height in pixels

    Notes
    -----
    #.  Written by David C. Stauffer in May 2018.
    #.  There are many ways to do this, but since I'm already using PyQt5, this one appears most
        reliable, especially when run on high DPI screens with scaling turned on.  However, don't
        call this function from within a GUI, as it will close everything.  Just query the desktop
        directly within the GUI.

    Examples
    --------
    >>> from dstauffman import get_screen_resolution
    >>> (screen_width, screen_height) = get_screen_resolution()
    >>> print('{}x{}'.format(screen_width, screen_height)) # doctest: +SKIP

    """
    # check to see if a QApplication exists, and if not, make one
    if QApplication.instance() is None:
        app = QApplication(sys.argv) # pragma: no cover
    else:
        app = QApplication.instance()
    # query the resolution
    screen_resolution = app.desktop().screenGeometry()
    # pull out the desired information
    screen_width = screen_resolution.width()
    screen_height = screen_resolution.height()
    # close the app
    app.closeAllWindows()
    return (screen_width, screen_height)

#%% Unit test
if __name__ == '__main__':
    plt.ioff()
    unittest.main(module='dstauffman.tests.test_plotting', exit=False)
    doctest.testmod(verbose=False)
