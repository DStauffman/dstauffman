# -*- coding: utf-8 -*-
r"""
Plotting module file for the "dstauffman" library.  It defines useful plotting utilities.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

# pylint: disable=E1101

#%% Imports
# normal imports
import doctest
import numpy as np
import os
import sys
import unittest
import warnings
# plotting imports
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.patches import Rectangle
# Qt imports
try:
    from PyQt5.QtWidgets import QApplication, QPushButton
    from PyQt5.QtGui import QIcon
    from PyQt5.QtCore import QSize
except ImportError:
    warnings.warn('PyQt5 was not found. Some funtionality will be limited.')
    QPushButton = object
# model imports
from dstauffman.classes   import Frozen
from dstauffman.constants import DEFAULT_COLORMAP
from dstauffman.utils     import get_images_dir, rms

#%% Private Classes - _HoverButton
class _HoverButton(QPushButton):
    r"""Custom button that allows hovering and icons."""
    def __init__(self, *args, **kwargs):
        # initialize
        super(_HoverButton, self).__init__(*args, **kwargs)
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
    Class that allows customization of when to show or not show plots (for use with testing plotting
    functions)
    """
    # class attribute for plotting flag
    show_plot = True

    def __init__(self, show=None):
        r"""Creates options instance with ability to override defaults."""
        if show is not None:
            type(self).show_plot = bool(show)

    def __str__(self):
        r"""Prints the current plotting flag."""
        return '{}({})'.format(type(self).__name__, self.show_plot)

    @classmethod
    def get_plotter(cls):
        r"""Gets the plotting flag."""
        return cls.show_plot

    @classmethod
    def set_plotter(cls, show):
        r"""Sets the plotting flag."""
        cls.show_plot = bool(show)

#%% Classes - Opts
class Opts(Frozen):
    r"""
    Contains all the optional plotting configurations.
    """
    def __init__(self):
        self.case_name = ''
        self.save_path = os.getcwd()
        self.save_plot = False
        self.plot_type = 'png'
        self.sub_plots = True
        self.show_plot = True
        self.show_link = False
        self.disp_xmin = -np.inf
        self.disp_xmax =  np.inf
        self.rms_xmin  = -np.inf
        self.rms_xmax  =  np.inf
        self.show_rms  = True
        self.names     = list()

    def get_names(self, ix):
        r"""Gets the specified name from the list."""
        if hasattr(self, 'names') and len(self.names) >= ix+1:
            name = self.names[ix]
        else:
            name = ''
        return name

#%% Classes - TruthPlotter
class TruthPlotter(Frozen):
    r"""
    Class wrapper for the different types of truth data to include on plots.
    """
    def __init__(self, time=None, data=None, lo=None, hi=None, type_='time', name='Truth'):
        self.time    = time
        self.data    = None
        self.type_   = type_ # from {'time', 'time_hilo', 'errorbars', 'series'} # TODO: implement this
        self.data_lo = lo
        self.data_hi = hi
        self.name    = name
        # TODO: keep this old API? (otherwise just: self.data = data)
        if data is not None:
            if data.ndim == 1:
                self.data = data
            elif data.shape[1] == 1:
                self.data = data
            elif data.shape[1] == 3:
                self.data    = data[:, 1]
                self.data_lo = data[:, 0]
                self.data_hi = data[:, 2]

    def plot_truth(self, ax, scale=1):
        r"""Adds the information in the TruthPlotter instance to the given axis, with the optional scale."""
        # check for null case
        if self.data is None and self.data_lo is None and self.data_hi is None:
            return
        # get original limits
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        # plot the new data
        if self.data is not None:
            ax.plot(self.time, scale*self.data, 'k.-', linewidth=2, zorder=8, label=self.name)
        if self.data_lo is not None:
            ax.plot(self.time, scale*self.data_lo, '.-', color='0.5', linewidth=2, zorder=6)
        if self.data_hi is not None:
            ax.plot(self.time, scale*self.data_hi, '.-', color='0.5', linewidth=2, zorder=6)
        # restore the original limits, since they might have been changed by the truth data
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

#%% Classes - MyCustomToolbar
class MyCustomToolbar():
    r"""
    Defines a custom toolbar to use in any matplotlib plots.

    Examples
    --------

    >>> from dstauffman import MyCustomToolbar
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> plt.ioff()
    >>> fig = plt.figure()
    >>> fig.canvas.set_window_title('Figure Title')
    >>> x = np.arange(0, 10, 0.1)
    >>> y = np.sin(x)
    >>> plt.plot(x, y) # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> fig.toolbar_custom_ = MyCustomToolbar(fig)

    Close plot
    >>> plt.close(fig)

    """
    def __init__(self, fig):
        r"""Initializes the custom toolbar."""
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
        r"""Closes all the currently open plots."""
        close_all()

    def next_plot(self, *args):
        r"""Brings up the next plot in the series."""
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
        r"""Brings up the previous plot in the series."""
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
    >>> plt.ioff()
    >>> cm = ColorMap('Paired', 1, 2)
    >>> time = np.arange(0, 10, 0.1)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(time, np.sin(time), color=cm.get_color(1)) # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at 0x...>]

    >>> ax.plot(time, np.cos(time), color=cm.get_color(2)) # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at 0x...>]

    >>> plt.legend(['Sin', 'Cos']) # doctest: +ELLIPSIS
    <matplotlib.legend.Legend object at 0x...>

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
        cmap  = plt.get_cmap(colormap)
        cnorm = colors.Normalize(vmin=low, vmax=high)
        self.smap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)
        # must initialize the empty scalar mapplable to show the colorbar correctly
        self.smap.set_array([])

    def get_color(self, value):
        r"""Get the color based on the scalar value."""
        return self.smap.to_rgba(value)

    def get_smap(self):
        r"""Returns the smap being used."""
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

#%% Functions - close_all
def close_all(figs=None):
    r"""
    Close all the open figures, or if a list is specified, then close all of them.
    """
    # Note that it's better to loop through and close the plots individually than to use
    # plt.close('all'), as that can sometimes cause the iPython kernel to quit #DCS: 2015-06-11
    if figs is None:
        for this_fig in plt.get_fignums():
            plt.close(this_fig)
    else:
        for this_fig in figs:
            plt.close(this_fig)

#%% Functions - get_axes_scales
def get_axes_scales(type_):
    r"""
    Determines the scale factor and units to apply to the plot based on the desired `type_`

    Parameters
    ----------
    type_ : str {'unity', 'population', 'percentage', 'per 100K', 'cost'}
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
    if type_ == 'unity':
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
    else:
        raise ValueError('Unexpected data type_ "{}" for plot.'.format(type_))
    return (scale, units)

#%% Functions - plot_time_history
def plot_time_history(time, data, description='', type_='unity', opts=None, *, plot_indiv=True, \
    truth=None, plot_as_diffs=False, colormap=None, second_y_scale=None, rms_in_legend=True, \
    truth_time=None, truth_data=None):
    r"""
    Plots the given data channel versus time, with a generic description argument.

    Parameters
    ----------
    time : array_like
        time history
    data : array_like
        data for corresponding time history
    description : str, optional
        generic text to put on the plot title and figure name, default is empty
    type_ : str, optional, from {'unity', 'population', 'percentage', 'per 100K', 'cost'}
        description of the type of data that is being plotted, default is 'unity'
    opts : class Opts, optional
        plotting options
    plot_indiv : bool, optional
        Plot the individual cycles, default is true
    truth : TruthPlotter, optional
        Truth instance for adding to the plot
    plot_as_diffs : bool, optional, default is False
        Plot each entry in results against the other ones
    second_y_scale : float, optional
        Multiplication scale factor to use to display on a secondary Y axis
    rms_in_legend : bool, optional
        Whether to show the RMS value numerically in the plotting legend

    Returns
    -------
    fig : object
        figure handle

    Notes
    -----
    #.  Written by David C. Stauffer in March 2015.
    #.  Updated by David C. Stauffer in December 2015 to include an optional secondary Y axis.
    #.  Updated by David C. Stauffer in October 2016 to use the new TruthPlotter class.
    #.  If ndim == 2, then dimension 0 is time and dimension 1 is the number of runs.

    Examples
    --------

    >>> from dstauffman import plot_time_history
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> time = np.arange(0, 10, 0.1)
    >>> data = np.sin(time)
    >>> description = 'Sin'
    >>> type_ = 'population'
    >>> fig = plot_time_history(time, data, description, type_) # doctest: +ELLIPSIS

    Close plot
    >>> plt.close()

    """
    # hard-coded values
    time_units = 'year' # TODO: make an input

    # force inputs to be ndarrays
    time = np.asanyarray(time)
    data = np.asanyarray(data)

    # check optional inputs
    if opts is None:
        opts = Opts()
    if colormap is None:
        colormap = DEFAULT_COLORMAP

    # maintain older API
    if truth_data is not None:
        if truth is not None:
            raise ValueError('Attempting to use both APIs, please only use new truth input.')
        else:
            warnings.warn('This API will be removed in the future, please use the new truth input.', DeprecationWarning)
            truth = TruthPlotter(truth_time, truth_data)

    # override the RMS option from opts (both must be true to plot the RMS in the legend)
    rms_in_legend &= opts.show_rms

    # ensure that data is at least 2D
    if data.ndim == 0:
        data = np.atleast_2d(data)
    elif data.ndim == 1:
        data = data[:, np.newaxis] # forces to grow in second dimension, instead of first

    # get number of different series
    num_series = data.shape[1]

    # determine which type of data to plot
    (scale, units) = get_axes_scales(type_)

    if plot_as_diffs:
        # build colormap
        cm = ColorMap(colormap, num_colors=data.shape[1])
        # calculate RMS
        if rms_in_legend:
            rms_data = np.atleast_1d(rms(scale*data, axis=0, ignore_nans=True))
    else:
        # calculate the mean and std of data
        mean = np.mean(data, axis=1)
        std  = np.std(data, axis=1)

        # calculate an RMS
        if rms_in_legend:
            rms_data = rms(scale*mean, axis=0, ignore_nans=True)

    # turn interaction off to make the plots draw all at once on a show() command
    plt.ioff()
    # alias the title
    this_title = description + ' vs. Time'
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
        this_label = opts.get_names(0) + description
        if rms_in_legend:
             this_label += ' (RMS: {:.2f})'.format(rms_data)
        ax.plot(time, scale*mean, 'b.-', linewidth=2, zorder=10, label=this_label)
        ax.errorbar(time, scale*mean, scale*std, linestyle='None', marker='None', ecolor='c', zorder=6)
        # inidividual line plots
        if plot_indiv and data.ndim > 1:
            for ix in range(num_series):
                ax.plot(time, scale*data[:, ix], color='0.75', zorder=1)
    # optionally plot truth (without changing the axis limits)
    if truth is not None:
        truth.plot_truth(ax, scale)
    # add labels and legends
    plt.xlabel('Time [' + time_units + ']')
    plt.ylabel(description + ' [' + units + ']')
    plt.title(this_title)
    plt.legend()
    # show a grid
    plt.grid(True)
    # set the colormap (only applies to lines that don't specify a default color)
    plt.set_cmap(colormap)
    # optionally add second Y axis
    if second_y_scale is not None:
        ax2 = ax.twinx()
        ax2.set_ylim(np.multiply(second_y_scale, ax.get_ylim()))
        if type_ == 'population':
            ax2.set_ylabel('Actual Population [#]')
    # Setup plots
    setup_plots(fig, opts, 'time')
    return fig

#%% Functions - plot_correlation_matrix
def plot_correlation_matrix(data, labels=None, type_='unity', opts=None, *, matrix_name='Correlation Matrix', \
        cmin=0, cmax=1, colormap='cool', xlabel='', ylabel='', plot_lower_only=True, label_values=False, \
        x_lab_rot=90):
    r"""
    Visually plots a correlation matrix.

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
    colormap : str, optional
        Name of colormap to use for plot
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
    # turn interaction off
    plt.ioff()
    # create figure
    fig = plt.figure()
    # set figure title
    fig.canvas.set_window_title(this_title)
    # get handle to axes for use later
    ax = fig.add_subplot(111)
    # set axis color to none
    ax.patch.set_facecolor('none')
    # set title
    plt.title(this_title)
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
    plt.colorbar(cm.get_smap())
    # make square
    plt.axis('equal')
    # set limits and tick labels
    plt.xlim(0, m)
    plt.ylim(0, n)
    plt.xticks(np.arange(0, m)+box_size/2, xlab, rotation=x_lab_rot)
    plt.yticks(np.arange(0, n)+box_size/2, ylab)
    # label axes
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # reverse the y axis
    ax.invert_yaxis()

    # Setup plots
    setup_plots(fig, opts, 'dist')
    return fig

#%% Functions - plot_multiline_history
def plot_multiline_history(time, data, type_='unity', label='', opts=None, *, legend=None, \
        colormap=None, second_y_scale=None, ignore_empties=False):
    r"""
    Plots multiple metrics over time.

    Parameters
    ----------
    time : 1D ndarray
        time history
    data : 2D or 3D ndarray
        data for corresponding time history, 2D: time by value in each category
    type_ : str, optional, from {'unity', 'population', 'percentage', 'per 100K', 'cost'}
        description of the type of data that is being plotted, default is 'unity'
    label : str, optional
        Disease name to label on the plots
    opts : class Opts, optional
        plotting options
    legend : list of str, optional
        Names to use for each channel of data
    colormap : str, optional
        Name of colormap to use for plot
    second_y_scale : float, optional
        Multiplication scale factor to use to display on a secondary Y axis
    ignore_empties : bool, optional
        Removes any entries from the plot and legend that contain only zeros or only NaNs

    Returns
    -------
    fig : object
        figure handle, if None, no figure was created

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.

    Examples
    --------

    >>> from dstauffman import plot_multiline_history
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> time  = np.arange(0, 5, 1./12) + 2000
    >>> data  = np.random.rand(len(time), 5)
    >>> mag   = data.cumsum(axis=1)[:,-1]
    >>> data  = 10 * data / np.expand_dims(mag, axis=1)
    >>> fig   = plot_multiline_history(time, data)

    Close plot
    >>> plt.close(fig)

    """
    # check optional inputs
    if opts is None:
        opts = Opts()
    if colormap is None:
        colormap = DEFAULT_COLORMAP

    # check for valid data
    if data is None or (ignore_empties and (np.all(data == 0) or np.all(np.isnan(data)))):
        print(' ' + label + ' plot skipped due to missing data.')
        return None

    # process other inputs
    description = label if label else 'Values over time'
    (scale, units) = get_axes_scales(type_)
    unit_text = ' [' + units + ']' if units else ''
    num_bins = data.shape[1]
    if legend is not None:
        assert len(legend) == num_bins, 'Number of data channels does not match the legend.'
    else:
        legend = ['Channel {}'.format(i+1) for i in range(num_bins)]

    # turn interactive plotting off
    plt.ioff()

    # get colormap based on high and low limits
    cm = ColorMap(colormap, num_colors=num_bins)

    # plot data
    fig = plt.figure()
    fig.canvas.set_window_title(description + unit_text)
    ax = fig.add_subplot(111)
    cm.set_colors(ax)
    for i in range(num_bins):
        if not ignore_empties or (not np.all(data[:, i] == 0) and not np.all(np.isnan(data[:, i]))):
            ax.plot(time, scale*data[:, i], '.-', label=legend[i])

    # add labels and legends
    plt.xlabel('Time [year]')
    plt.ylabel(fig.canvas.get_window_title())
    plt.title(description + ' vs. Time')
    plt.legend()
    plt.grid(True)

    # optionally add second Y axis
    if second_y_scale is not None:
        ax2 = ax.twinx()
        ax2.set_ylim(np.multiply(second_y_scale,ax.get_ylim()))
        if type_ == 'population':
            ax2.set_ylabel('Actual Population [#]')

    # setup plots
    setup_plots(fig, opts, 'time')
    return fig

#%% Functions - plot_bar_breakdown
def plot_bar_breakdown(time, data, label='', opts=None, legend=None, colormap=None, \
        ignore_empties=False):
    r"""
    Plots the pie chart like breakdown by percentage in each category over time.

    Parameters
    ----------
    time : array_like
        time history
    data : array_like
        data for corresponding time history, 2D: time by ratio in each category
    label : str, optional
        Disease name to label on the plots
    opts : class Opts, optional
        plotting options
    legend : list of str, optional
        Names to use for each channel of data
    colormap : str, optional
        Name of colormap to use for plot
    ignore_empties : bool, optional
        Removes any entries from the plot and legend that contain only zeros or only NaNs

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
    >>> time = np.arange(0, 5, 1./12) + 2000
    >>> data = np.random.rand(len(time), 5)
    >>> mag  = data.cumsum(axis=1)[:,-1]
    >>> data = data / np.expand_dims(mag, axis=1)
    >>> fig  = plot_bar_breakdown(time, data)

    Close plots
    >>> plt.close(fig)

    """
    # check optional inputs
    if opts is None:
        opts = Opts()
    if colormap is None:
        colormap = DEFAULT_COLORMAP

    # check for valid data
    if data is None or (ignore_empties and (np.all(data == 0) or np.all(np.isnan(data)))):
        print(' ' + label + ' plot skipped due to missing data.')
        return

    # hard-coded values
    description = 'Ratios over time'
    if label:
        description = label + ' ' + description
    scale    = 100
    units    = '%'
    width    = time[1] - time[0]
    num_bins = data.shape[1]
    if legend is not None:
        assert len(legend) == num_bins, 'Number of data channels does not match the legend.'
    else:
        legend = ['Series {}'.format(i+1) for i in range(num_bins)]

    # turn interactive plotting off
    plt.ioff()

    # get colormap based on high and low limits
    cm = ColorMap(colormap, 0, num_bins-1)

    # figure out where the bottoms should be to stack the data
    bottoms = np.concatenate((np.zeros((len(time),1)), np.cumsum(data, axis=1)), axis=1)

    # plot breakdown
    fig = plt.figure()
    fig.canvas.set_window_title(description + ' [' + units + ']')
    ax = fig.add_subplot(111)
    for i in range(num_bins):
        if not ignore_empties or (not np.all(data[:, i] == 0) and not np.all(np.isnan(data[:, i]))):
            ax.bar(time-width/2, scale*data[:, i], width=width.astype(float), bottom=scale*bottoms[:, i], \
                label=legend[i], color=cm.get_color(i), edgecolor='none')
    plt.ylabel(label + ' [' + units + ']')
    plt.ylim(0, 100)
    plt.grid(True)
    plt.legend()
    plt.title(fig.canvas.get_window_title())

    # Setup plots
    setup_plots(fig, opts, 'time')
    return fig

#%% Functions - plot_bpe_convergence
def plot_bpe_convergence(costs, opts=None):
    r"""
    Plots the BPE convergence rate by iteration on a log scale.

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

    # turn interaction off to make the plots draw all at once on a show() command
    plt.ioff()
    # alias the title
    this_title = 'Convergence by Iteration'
    # create the figure and set the title
    fig = plt.figure()
    fig.canvas.set_window_title(this_title)
    # add an axis and plot the data
    ax = fig.add_subplot(111)
    ax.semilogy(time, costs, 'b.-', linewidth=2)
    # add labels
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title(this_title)
    plt.xticks(time, labels)
    # show a grid
    plt.grid(True)
    # Setup plots
    setup_plots(fig, opts, 'time')
    return fig

#%% Functions - storefig
def storefig(fig, folder=None, plot_type='png'):
    r"""
    Stores the specified figures in the specified folder and with the specified plot type(s)

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
    >>> plt.ioff()
    >>> fig = plt.figure()
    >>> fig.canvas.set_window_title('Figure Title')
    >>> x = np.arange(0, 10, 0.1)
    >>> y = np.sin(x)
    >>> plt.plot(x, y) # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.title('X vs Y') # doctest: +ELLIPSIS
    <matplotlib.text.Text object at 0x...>
    >>> plt.show(block=False) # doctest: +SKIP
    >>> folder = os.getcwd()
    >>> plot_type = 'png'
    >>> storefig(fig, folder, plot_type)

    Close plot
    >>> plt.close()

    Delete file
    >>> os.remove(os.path.join(folder, 'Figure Title.png'))

    """
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
        # loop through the plot types
        for this_type in types:
            # save the figure to the specified plot type
            this_fig.savefig(os.path.join(folder, this_title + '.' + this_type), dpi=160, bbox_inches='tight')

#%% Functions - titleprefix
def titleprefix(fig, prefix=''):
    r"""
    Prepends a text string to all the titles on existing figures.

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
    >>> plt.ioff()
    >>> fig = plt.figure()
    >>> fig.canvas.set_window_title('Figure Title')
    >>> x = np.arange(0, 10, 0.1)
    >>> y = np.sin(x)
    >>> plt.plot(x, y) # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.title('X vs Y') # doctest: +ELLIPSIS
    <matplotlib.text.Text object at 0x...>
    >>> plt.show(block=False) # doctest: +SKIP
    >>> prefix = 'Baseline'
    >>> titleprefix(fig, prefix)
    >>> plt.draw() # doctest: +SKIP

    Close plot
    >>> plt.close()

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
    Sets the xlimits to the specified xmin and xmax.

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
    >>> plt.ioff()
    >>> fig = plt.figure()
    >>> fig.canvas.set_window_title('Figure Title')
    >>> x = np.arange(0, 10, 0.1)
    >>> y = np.sin(x)
    >>> plt.plot(x, y) # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.title('X vs Y') # doctest: +ELLIPSIS
    <matplotlib.text.Text object at 0x...>
    >>> plt.show(block=False) # doctest: +SKIP
    >>> xmin = 2
    >>> xmax = 5
    >>> disp_xlimits(fig, xmin, xmax)
    >>> plt.draw() # doctest: +SKIP

    Close plot
    >>> plt.close()

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
    Combines common plot operations into one easy command.

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
    >>> plt.ioff()
    >>> fig = plt.figure()
    >>> fig.canvas.set_window_title('Figure Title')
    >>> x = np.arange(0, 10, 0.1)
    >>> y = np.sin(x)
    >>> plt.plot(x, y) # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.title('X vs Y') # doctest: +ELLIPSIS
    <matplotlib.text.Text object at 0x...>
    >>> plt.xlabel('time [years]') # doctest: +SKIP
    >>> plt.ylabel('value [radians]') # doctest: +SKIP
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
    if opts.show_plot and Plotter.show_plot:
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
    Adds a custom toolbar to the figures.

    Parameters
    ----------
    figs : class matplotlib.pyplot.Figure, or list of such
        List of figures

    Examples
    --------

    >>> from dstauffman import figmenu
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> plt.ioff()
    >>> fig = plt.figure()
    >>> fig.canvas.set_window_title('Figure Title')
    >>> x = np.arange(0, 10, 0.1)
    >>> y = np.sin(x)
    >>> plt.plot(x, y) # doctest: +ELLIPSIS
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.title('X vs Y') # doctest: +ELLIPSIS
    <matplotlib.text.Text object at 0x...>
    >>> plt.xlabel('time [years]') # doctest: +SKIP
    >>> plt.ylabel('value [radians]') # doctest: +SKIP
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
    Converts a tuple of ints with (0, 255) to the equivalent hex color code.

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

#%% Unit test
if __name__ == '__main__':
    unittest.main(module='tests.test_plotting', exit=False)
    doctest.testmod(verbose=False)
