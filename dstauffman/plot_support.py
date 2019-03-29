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
from dstauffman.paths import get_images_dir
from dstauffman.utils import pprint_dict

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

#%% Functions - get_color_lists
def get_color_lists():
    r"""
    Gets different color lists to use for plotting
    """
    color_lists              = {}
    color_lists['default']   = 'Paired' #'Dark2' # 'YlGn' # 'gnuplot2' # 'cubehelix'
    color_lists['single']    = colors.ListedColormap(('xkcd:red',))
    color_lists['double']    = colors.ListedColormap(('xkcd:red', 'xkcd:blue'))
    color_lists['vec']       = colors.ListedColormap(('xkcd:red', 'xkcd:green', 'xkcd:blue'))
    color_lists['quat']      = colors.ListedColormap(('xkcd:red', 'xkcd:green', 'xkcd:blue', 'xkcd:chocolate'))
    color_lists['dbl_diff']  = colors.ListedColormap(('xkcd:red', 'xkcd:blue', 'xkcd:fuchsia', 'xkcd:cyan'))
    color_lists['vec_diff']  = colors.ListedColormap(('xkcd:red', 'xkcd:green', 'xkcd:blue',
        'xkcd:fuchsia', 'xkcd:lightgreen', 'xkcd:cyan'))
    color_lists['quat_diff'] = colors.ListedColormap(('xkcd:red', 'xkcd:green', 'xkcd:blue',
        'xkcd:chocolate', 'xkcd:fuchsia', 'xkcd:lightgreen', 'xkcd:cyan', 'xkcd:brown'))
    return color_lists

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
    if data is None or np.all(data == None):
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

#%% Functions - show_zero_ylim
def show_zero_ylim(ax):
    r"""
    Forces the given axes to always include the point zero.

    Parameters
    ----------
    ax : class matplotlib.axis.Axis
        Figure axis

    Examples
    --------
    >>> from dstauffman import show_zero_ylim
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot([1, 5, 10], [200, 250, 240], '.-')
    >>> show_zero_ylim(ax)

    >>> plt.close(fig)

    """
    # optionally force zero to be plotted
    ylim = ax.get_ylim()
    if min(ylim) > 0:
        ax.set_ylim(bottom=0)
    if max(ylim) < 0:
        ax.set_ylim(top=0)

#%% Functions - plot_second_yunits
def plot_second_yunits(ax, ylab, multiplier):
    r"""
    Plots a second Y axis on the right side of the plot with a different scaling.

    Parameters
    ----------
    ax : class matplotlib.axis.Axis
        Figure axis
    ylab : str
        Label for new axis
    multiplier : float
        Multiplication factor

    Examples
    --------
    >>> from dstauffman import plot_second_yunits
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot([1, 5, 10], [1e-6, 3e-6, 2.5e-6], '.-')
    >>> ax.set_ylabel('Value [rad]')
    >>> ylab = 'Value [µrad]'
    >>> multiplier = 1e6
    >>> plot_second_yunits(ax, ylab, multiplier)

    """
    # plot second Y axis
    ax2 = ax.twinx()
    ax2.set_ylim(np.multiply(multiplier, ax.get_ylim()))
    ax2.set_ylabel(ylab)

#%% Functions - plot_rms_lines
def plot_rms_lines(ax, x, y, show_in_legend=True):
    r"""
    Plots a vertical line at the RMS start and stop times.

    Summary
    -------
    There are two vertical lines created.  The first at x(1) and the second at x(2).
    The first line is orange, and the second is lavender.  Both have magenta crosses at
    the top and bottom.  The lines are added to the plot regardless of the figure hold state.

    Parameters
    ----------
    x : (2,) tuple
        xmin and xmax values at which to draw the lines
    y : (2,) tuple
        ymin and ymax values at which to extend the lines vertically [num]
    show_in_legend : bool, optional
        show the lines when a legend is turned on

    Notes
    -----
    #.  Added to DStauffman's MATLAB libary from GARSE in Sept 2013.
    #.  Ported to Python by David C. Stauffer in March 2019.

    Examples
    --------
    >>> from dstauffman import plot_rms_lines
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(np.arange(10), np.arange(10), label='Data')
    >>> x = (2, 5)
    >>> y = (1, 10)
    >>> plot_rms_lines(ax, x, y, show_in_legend=False)
    >>> ax.legend()

    >>> plt.close(fig)

    """
    if show_in_legend:
        label_one = 'RMS Start Time'
        label_two = 'RMS Stop Time'
    else:
        label_one = ''
        label_two = ''
    ax.plot([x[0], x[0]], y, linestyle='--', color=[   1, 0.75, 0], marker='+', markeredgecolor='m', markersize=10, label=label_one)
    ax.plot([x[1], x[1]], y, linestyle='--', color=[0.75, 0.75, 1], marker='+', markeredgecolor='m', markersize=10, label=label_two)

#%% Unit test
if __name__ == '__main__':
    plt.ioff()
    unittest.main(module='dstauffman.tests.test_plot_support', exit=False)
    doctest.testmod(verbose=False)
