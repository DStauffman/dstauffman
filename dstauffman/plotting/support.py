r"""
Defines useful plotting utilities.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""

#%% Imports
from __future__ import annotations
import datetime
import doctest
import gc
import operator
import os
import platform
import re
import sys
from typing import Dict, List, Literal, Optional, overload, Tuple, TYPE_CHECKING, Union
import unittest
import warnings

try:
    from PyQt5.QtCore import QSize
    from PyQt5.QtGui import QIcon
    from PyQt5.QtWidgets import QApplication, QPushButton
except ModuleNotFoundError: # pragma: no cover
    warnings.warn('PyQt5 was not found. Some funtionality will be limited.')
    QPushButton = object

from dstauffman import Frozen, get_images_dir, HAVE_MPL, HAVE_NUMPY, HAVE_SCIPY, is_datetime

if HAVE_MPL:
    from matplotlib.axes import Axes
    import matplotlib.cm as cmx
    import matplotlib.colors as colors
    from matplotlib.dates import date2num
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt

    # Newer date stamps on axes, done here as this is the lowest level of the plotting submodule
    from matplotlib.dates import ConciseDateConverter
    import matplotlib.units as munits
    converter = ConciseDateConverter()
    munits.registry[datetime.date] = converter
    munits.registry[datetime.datetime] = converter
if HAVE_NUMPY:
    import numpy as np
    inf = np.inf
    if HAVE_MPL:
        munits.registry[np.datetime64] = converter
else:
    from math import inf
if HAVE_SCIPY:
    import scipy.stats as st

#%% Constants
# Default colormap to use on certain plots
DEFAULT_COLORMAP: str = 'Paired' #'Dark2' # 'YlGn' # 'gnuplot2' # 'cubehelix'

# Whether to include a classification on any generated plots
DEFAULT_CLASSIFICATION: str = ''

if TYPE_CHECKING:
    _FigOrListFig = Union[Figure, List[Figure]]

#%% Set Matplotlib global settings
if HAVE_MPL:
    plt.rcParams['figure.dpi']     = 160 # 160 for 4K monitors, 100 otherwise
    plt.rcParams['figure.figsize'] = [11., 8.5] # makes figures the same size as the paper, keeping aspect ratios even
    plt.rcParams['figure.max_open_warning'] = 80 # Max number of figures to open before through a warning, 0 means unlimited
    plt.rcParams['date.autoformatter.minute'] = '%H:%M:%S' # makes seconds show, and not day, default is '%d %H:%M'

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

#%% Classes - MyCustomToolbar
class MyCustomToolbar():
    r"""
    Define a custom toolbar to use in any matplotlib plots.

    Examples
    --------
    >>> from dstauffman.plotting import MyCustomToolbar
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
    >>> from dstauffman.plotting import ColorMap
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> cm = ColorMap('Paired', num_colors=12)
    >>> time = np.arange(0, 10, 0.1)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> _ = ax.plot(time, np.sin(time), color=cm.get_color(0))
    >>> _ = ax.plot(time, np.cos(time), color=cm.get_color(1))
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
        if colormap is None:
            cmap = plt.get_cmap(DEFAULT_COLORMAP)
        elif isinstance(colormap, colors.Colormap):
            cmap = colormap
        elif isinstance(colormap, type(self)):
            cmap = None
        else:
            cmap = plt.get_cmap(colormap)
        if cmap is None:
            self.smap = colormap.get_smap()
        else:
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
        r"""Set the colors for the given axes based on internal instance information."""
        if self.num_colors is None:
            raise ValueError("You can't call ColorMap.set_colors unless it was given a num_colors input.")
        try:
            ax.set_prop_cycle('color', [self.get_color(i) for i in range(self.num_colors)])
        except AttributeError: # pragma: no cover
            # for older matplotlib versions, use deprecated set_color_cycle
            ax.set_color_cycle([self.get_color(i) for i in range(self.num_colors)])

#%% Functions - close_all
def close_all(figs: _FigOrListFig = None) -> None:
    r"""
    Close all the open figures, or if a list is specified, then close all of them.

    Parameters
    ----------
    figs : list of Figure, optional
        Specific figures to be closed.

    Examples
    --------
    >>> from dstauffman.plotting import close_all
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
@overload
def get_color_lists(return_as_colormap: Literal[True] = ...) -> Dict[str, ColorMap]: ...

@overload
def get_color_lists(return_as_colormap: Literal[False]) -> Dict[str, Union[colors.ListedColormap, str]]: ...

def get_color_lists(return_as_colormap: bool = False) -> Union[Dict[str, Union[colors.ListedColormap, str]], Dict[str, ColorMap]]:
    r"""
    Gets different color lists to use for plotting.

    Returns
    -------
    color_lists : dict
        Lists of colors, either as str or matplotlib.colors.ListedColormap
        includes:
            default   : Default scheme to use when plotting
            same      : Each color will be the same, and match the first color from default
            same_old  : Each color will be the same, with the matplotlib default medium bluish color (#1f77b4)
            single    : When you only want one color
            double    : For two colors
            vec       : For three colors
            quat      : For four colors
            dbl_diff  : For 2x2 related colors, giving four total
            vec_diff  : For 3x2 related colors, giving six total
            quat_diff : For 4x2 related colors, giving eight total

    Examples
    --------
    >>> from dstauffman.plotting import get_color_lists
    >>> color_lists = get_color_lists()
    >>> print(color_lists['default'])
    Paired

    >>> color_lists_map = get_color_lists(return_as_colormap=True)
    >>> print(color_lists_map['default'].get_color(0)) # doctest: +ELLIPSIS
    (0.65098..., 0.80784..., 0.890196..., 1.0)

    """
    color_lists: Dict[str, Union[colors.ListedColormap, str]] = {}
    color_lists['default']     = 'Paired' #'Dark2' # 'YlGn' # 'gnuplot2' # 'cubehelix'
    # single colors
    first_color                = ColorMap(color_lists['default'], num_colors=1).get_color(0)
    color_lists['same']        = colors.ListedColormap([first_color for _ in range(8)])
    color_lists['same_old']    = colors.ListedColormap(['#1f77b4' for _ in range(8)])
    color_lists['single']      = colors.ListedColormap(('xkcd:red', ))
    # doubles
    color_lists['double']      = colors.ListedColormap(('xkcd:red', 'xkcd:blue'))
    color_lists['dbl_off']     = colors.ListedColormap(('xkcd:fuchsia', 'xkcd:cyan'))
    color_lists['dbl_diff']    = colors.ListedColormap(color_lists['dbl_off'].colors + color_lists['double'].colors)  # type: ignore[union-attr]
    color_lists['dbl_diff_r']  = colors.ListedColormap(color_lists['double'].colors + color_lists['dbl_off'].colors)  # type: ignore[union-attr]
    # triples
    color_lists['vec']         = colors.ListedColormap(('xkcd:red', 'xkcd:green', 'xkcd:blue'))
    color_lists['vec_off']     = colors.ListedColormap(('xkcd:fuchsia', 'xkcd:lightgreen', 'xkcd:cyan'))
    color_lists['vec_diff']    = colors.ListedColormap(color_lists['vec_off'].colors + color_lists['vec'].colors)  # type: ignore[union-attr]
    color_lists['vec_diff_r']  = colors.ListedColormap(color_lists['vec'].colors + color_lists['vec_off'].colors)  # type: ignore[union-attr]
    # quads
    color_lists['quat']        = colors.ListedColormap(('xkcd:red', 'xkcd:green', 'xkcd:blue', 'xkcd:chocolate'))
    color_lists['quat_off']    = colors.ListedColormap(('xkcd:fuchsia', 'xkcd:lightgreen', 'xkcd:cyan', 'xkcd:brown'))
    color_lists['quat_diff']   = colors.ListedColormap(color_lists['quat_off'].colors + color_lists['quat'].colors)  # type: ignore[union-attr]
    color_lists['quat_diff_r'] = colors.ListedColormap(color_lists['quat'].colors + color_lists['quat_off'].colors)  # type: ignore[union-attr]
    if return_as_colormap:
        color_list_maps: Dict[str, ColorMap] = {}
        for (key, value) in color_lists.items():
            if key == 'default':
                assert isinstance(value, str)
                if value == 'Paired':
                    num_colors = 12
                elif value == 'tab10':
                    num_colors = 10
                elif value == 'tab20':
                    num_colors = 20
                else:
                    num_colors = 1
            else:
                assert isinstance(value, colors.ListedColormap)
                num_colors = value.N
            color_list_maps[key] = ColorMap(value, num_colors=num_colors)
        return color_list_maps
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
    >>> from dstauffman.plotting import ignore_plot_data
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
    >>> from dstauffman.plotting import whitten
    >>> color = (1, 0.4, 0)
    >>> new_color = whitten(color)
    >>> print(new_color)
    (1.0, 0.58, 0.3)

    """
    # apply the shift
    new_color = tuple((c*(1-dt) + w*dt for (c, w) in zip(color, white)))
    return new_color

#%% Functions - resolve_name
def resolve_name(name: str, force_win: bool = None, rep_token: str = '_', strip_classification: bool = True) -> str:
    r"""
    Resolves the given name to something that can be saved on the current OS.

    Parameters
    ----------
    name : str
        Name of the file
    force_win : bool, optional
        Flag to it to Windows or Unix methods, mostly for testing
    rep_token : str, optional
        Character to use to replace the bad ones with, default is underscore

    Returns
    -------
    new_name : str
        Name of the file with any invalid characters replaced

    Examples
    --------
    >>> from dstauffman.plotting import resolve_name
    >>> name = '(U//FOUO) Bad name /\ <>!'
    >>> force_win = True
    >>> new_name = resolve_name(name, force_win=force_win)
    >>> print(new_name)
    Bad name __ __!

    """
    # hard-coded values
    bad_chars_win  = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    bad_chars_unix = ['/']

    # determine OS and thus which characters are bad
    if force_win is None:
        is_windows = platform.system() == 'Windows'
    else:
        is_windows = force_win
    bad_chars      = bad_chars_win if is_windows else bad_chars_unix

    # initialize output
    new_name = name

    # strip any leading classification text
    if strip_classification:
        new_name = re.sub(r'^\(\S*\)\s', '', new_name, count=1)

    # replace any bad characters with underscores
    for ch in bad_chars:
        if ch in new_name:
            new_name = new_name.replace(ch, rep_token)
    return new_name

#%% Functions - storefig
def storefig(fig: _FigOrListFig, folder: str = None, plot_type: Union[str, List[str]] = 'png') -> None:
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
    #.  Uses the figure.canvas.get_window_title property to determine the figure name.  If that is
        not set or default ('image'), then it tries the figure suptitle or first axes title.

    See Also
    --------
    matplotlib.pyplot.savefig, titleprefix

    Examples
    --------
    Create figure and then save to disk
    >>> from dstauffman.plotting import storefig
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
    # make sure figs is a list
    if isinstance(fig, list):
        figs = fig
    else:
        figs = [fig]
    # make sure types is a list
    if not isinstance(plot_type, list):
        types: List[str] = []
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
    throw_warning = False
    for this_fig in figs:
        # get the title of the figure canvas
        raw_title = this_fig.canvas.get_window_title()
        if raw_title is None or raw_title == 'image':
            # special case when you have a displayless backend, check the suptitle, then the title
            # from the first axes
            throw_warning = True
            if (sup := this_fig._suptitle) is not None:
                raw_title = sup.get_text()
            else:
                try:
                    raw_title = this_fig.axes[0].get_title()
                except:
                    pass
        this_title = resolve_name(raw_title)
        # loop through the plot types
        for this_type in types:
            # save the figure to the specified plot type
            this_fig.savefig(os.path.join(folder, this_title + '.' + this_type), dpi=160, \
                             bbox_inches='tight', pad_inches=0.01)
    if throw_warning:
        warnings.warn('No window titles found, using the plot title instead (usually because there is no display).')

#%% Functions - titleprefix
def titleprefix(fig: _FigOrListFig, prefix: str = '') -> None:
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
    >>> from dstauffman.plotting import titleprefix
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
        # get axes list and loop through them
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
        # update the suptitle (if it exists)
        if (sup := this_fig._suptitle) is not None:
            sup.set_text(prefix + ' - ' + sup.get_text())

#%% Functions - disp_xlimits
def disp_xlimits(fig_or_axis, xmin=None, xmax=None):
    r"""
    Set the xlimits to the specified xmin and xmax.

    Parameters
    ----------
    fig_or_axis : matlpotlib.pyplot.Axes or matplotlib.pyplot.Figure or list of them
        List of figures/axes to process
    xmin : scalar
        Minimum X value
    xmax : scalar
        Maximum X value

    Notes
    -----
    #.  Written by David C. Stauffer in August 2015.
    #.  Modified by David C. Stauffer in May 2020 to come out of setup_plots and into lower level
        routines.

    Examples
    --------
    >>> from dstauffman.plotting import disp_xlimits
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
    # check for single item
    if not isinstance(fig_or_axis, list):
        fig_or_axis = [fig_or_axis]
    # loop through items and collect axes
    ax = []
    for this in fig_or_axis:
        if isinstance(this, Figure):
            ax.extend(this.axes)
        elif isinstance(this, Axes):
            ax.append(this)
        else:
            raise ValueError('Unexpected item that is neither a figure nor axes.')
    # loop through axes
    for this_axis in ax:
        # get xlimits for this axis
        (old_xmin, old_xmax) = this_axis.get_xlim()
        # set the new limits
        if xmin is not None:
            if is_datetime(xmin):
                new_xmin = np.maximum(date2num(xmin), old_xmin)
            else:
                new_xmin = np.max([xmin, old_xmin])
        else:
            new_xmin = old_xmin
        if xmax is not None:
            if is_datetime(xmax):
                new_xmax = np.minimum(date2num(xmax), old_xmax)
            else:
                new_xmax = np.min([xmax, old_xmax])
        else:
            new_xmax = old_xmax
        # check for bad conditions
        if np.isinf(new_xmin) or np.isnan(new_xmin):
            new_xmin = old_xmin
        if np.isinf(new_xmax) or np.isnan(new_xmax):
            new_xmax = old_xmax
        # modify xlimits
        this_axis.set_xlim((new_xmin, new_xmax))

#%% Functions - zoom_ylim
def zoom_ylim(ax, time=None, data=None, *, t_start=-inf, t_final=inf, channel=None, pad=0.1):
    r"""
    Zooms the Y-axis to the data for the given time bounds, with an optional pad.

    Parameters
    ----------
    ax : class matplotlib.axes.Axes
        Figure axes
    time : (N, ) ndarray
        Time history
    data : (N, ) or (N, M) ndarray
        Data history
    t_start : float
        Starting time to zoom data to
    t_final : float
        Final time to zoom data to
    channel : int, optional
        Column within 2D data to look at
    pad : int
        Amount of pad, as a percentage of delta range, to show around the plot bounds

    Notes
    -----
    #.  Written by David C. Stauffer in August 2019.

    Examples
    --------
    >>> from dstauffman.plotting import disp_xlimits, zoom_ylim
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig = plt.figure()
    >>> fig.canvas.set_window_title('Figure Title')
    >>> ax = fig.add_subplot(111)
    >>> time = np.arange(1, 10, 0.1)
    >>> data = time ** 2
    >>> _ = ax.plot(time, data)
    >>> _ = ax.set_title('X vs Y')
    >>> plt.show(block=False) # doctest: +SKIP

    Zoom X-axis and show how Y doesn't rescale
    >>> t_start = 3
    >>> t_final = 5.0001
    >>> disp_xlimits(fig, t_start, t_final)
    >>> plt.draw() # doctest: +SKIP

    Force Y-axis to rescale to data
    >>> zoom_ylim(ax, time, data, t_start=t_start, t_final=t_final, pad=0)
    >>> plt.draw() # doctest: +SKIP

    Close plot
    >>> plt.close(fig)

    """
    # If not given, find time/data from the plot itself
    if time is None:
        time = np.hstack([artist.get_xdata() for artist in ax.lines])
    if data is None:
        data = np.hstack([artist.get_ydata() for artist in ax.lines])
    # convert datetimes as appropriate for comparisons
    if is_datetime(time):
        time = date2num(time)
    # find the relevant time indices
    ix_time = (time >= t_start) & (time <= t_final)
    # exit if no data is in this time window
    if ~np.any(ix_time):
        warnings.warn('No data matched the given time interval.')
        return
    # pull out the minimums/maximums from the data
    if channel is None:
        if data.ndim == 1:
            this_ymin = np.min(data[ix_time])
            this_ymax = np.max(data[ix_time])
        else:
            this_ymin = np.min(data[ix_time, :])
            this_ymax = np.max(data[ix_time, :])
    else:
        this_ymin = np.min(data[ix_time, channel])
        this_ymax = np.max(data[ix_time, channel])
    # optionally pad the bounds
    if pad < 0:
        raise ValueError('The pad cannot be negative.')
    if pad > 0:
        delta = this_ymax - this_ymin
        this_ymax += pad*delta
        this_ymin -= pad*delta
    # check for the case where the data is constant and the limits are the same
    if this_ymin == this_ymax:
        if this_ymin == 0:
            # data is exactly zero, show from -1 to 1
            this_ymin = -1
            this_ymax = 1
        else:
            # data is constant, pad by given amount or 10% if pad is zero
            pad = pad if pad > 0 else 0.1
            this_ymin = (1-pad) * this_ymin
            this_ymax = (1+pad) * this_ymax
    # get the current limits
    (old_ymin, old_ymax) = ax.get_ylim()
    # compare the new bounds to the old ones and update as appropriate
    if this_ymin > old_ymin:
        ax.set_ylim(bottom=this_ymin)
    if this_ymax < old_ymax:
        ax.set_ylim(top=this_ymax)

#%% Functions - figmenu
def figmenu(figs: _FigOrListFig) -> None:
    r"""
    Add a custom toolbar to the figures.

    Parameters
    ----------
    figs : class matplotlib.pyplot.Figure, or list of such
        List of figures

    Examples
    --------
    >>> from dstauffman.plotting import figmenu
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
def rgb_ints_to_hex(int_tuple: Tuple[int, int, int]) -> str:
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
    >>> from dstauffman.plotting import rgb_ints_to_hex
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
def get_screen_resolution() -> Tuple[int, int]:
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
    >>> from dstauffman.plotting import get_screen_resolution
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
def show_zero_ylim(ax: Axes) -> None:
    r"""
    Forces the given axes to always include the point zero.

    Parameters
    ----------
    ax : class matplotlib.axes.Axes
        Figure axes

    Examples
    --------
    >>> from dstauffman.plotting import show_zero_ylim
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> _ = ax.plot([1, 5, 10], [200, 250, 240], '.-')
    >>> show_zero_ylim(ax)

    >>> plt.close(fig)

    """
    # optionally force zero to be plotted
    ylim = ax.get_ylim()
    if min(ylim) > 0:
        ax.set_ylim(bottom=0)
    if max(ylim) < 0:
        ax.set_ylim(top=0)

#%% Functions - plot_second_units_wrapper
def plot_second_units_wrapper(ax: Axes, second_yscale: Union[None, int, float, Dict[str, float]]) -> Axes:
    r"""
    Wrapper to plot_second_yunits that allows numeric or dict options.

    Parameters
    ----------
    ax : class matplotlib.axes.Axes
        Figure axes
    second_yscale : dict or int or float
        Scale factor to apply, or dict with key for label and value for factor

    Returns
    -------
    ax2 : class matplotlib.axes.Axes
        New Figure axes with the second label

    Notes
    -----
    #.  If second_yscale is just a number, then no units are displayed, but if a key and value,
        then if it has brakets, replace the entire label, otherwise only replace what is in the
        old label within the brackets

    Examples
    --------
    >>> from dstauffman.plotting import plot_second_units_wrapper
    >>> import matplotlib.pyplot as plt
    >>> description = 'Values over time'
    >>> ylabel = 'Value [rad]'
    >>> second_yscale = {u'Better Units [µrad]': 1e6}
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> _ = ax.plot([1, 5, 10], [1e-6, 3e-6, 2.5e-6], '.-')
    >>> _ = ax.set_ylabel(ylabel)
    >>> _ = ax.set_title(description)
    >>> _ = plot_second_units_wrapper(ax, second_yscale)

    >>> plt.close(fig)

    """
    # initialize output
    ax2 = None
    # check if processing anything
    if second_yscale is not None:
        # determine what type of input was given
        if isinstance(second_yscale, (int, float)):
            key = ''
            value = second_yscale
        else:
            key = list(second_yscale.keys())[0]
            value = second_yscale[key]
        # check if we got a no-op value
        if not np.isnan(value) and value != 0:
            # if all is good, build the new label and call the lower level function
            old_label = ax.get_ylabel()
            ix1       = old_label.find('[')
            ix2       = key.find('[')
            if ix2 >= 0:
                # new label has units, so use them
                new_label = key
            elif ix1 >= 0 and key:
                # new label is only units, replace them in the old label
                new_label = old_label[:ix1] + '[' + key + ']'
            else:
                # neither label has units, just label them
                new_label = key
            ax2 = plot_second_yunits(ax, new_label, value)
    return ax2

#%% Functions - plot_second_yunits
def plot_second_yunits(ax: Axes, ylab: str, multiplier: float) -> Axes:
    r"""
    Plots a second Y axis on the right side of the plot with a different scaling.

    Parameters
    ----------
    ax : class matplotlib.axes.Axes
        Figure axes
    ylab : str
        Label for new axes
    multiplier : float
        Multiplication factor

    Returns
    -------
    ax2 : class matplotlib.axes.Axes
        New Figure axes with the second label

    Examples
    --------
    >>> from dstauffman.plotting import plot_second_yunits
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> _ = ax.plot([1, 5, 10], [1e-6, 3e-6, 2.5e-6], '.-')
    >>> _ = ax.set_ylabel('Value [rad]')
    >>> ylab = u'Value [µrad]'
    >>> multiplier = 1e6
    >>> _ = plot_second_yunits(ax, ylab, multiplier)

    >>> plt.close(fig)

    """
    # plot second Y axis
    ax2 = ax.twinx()
    ax2.set_ylim(np.multiply(multiplier, ax.get_ylim()))
    ax2.set_ylabel(ylab)
    return ax2

#%% Functions - get_rms_indices
def get_rms_indices(time_one=None, time_two=None, time_overlap=None, *, xmin=-inf, xmax=inf):
    r"""
    Gets the indices and time points for doing RMS calculations and plotting RMS lines.

    Parameters
    ----------
    time_one : array_like
        Time vector one
    time_two : array_like
        Time vector two
    time_overlap : array_like
        Time vector of points in both arrays
    xmin : float
        Minimum time to include in calculation
    xmax : float
        Maximum time to include in calculation

    Returns
    -------
    ix : dict
        Dictionary of indices, with fields:
        pts : [2, ] float
            Time to start and end the RMS calculations from
        one : (A, ) ndarray of bool
            Array of indices into time_one between the rms bounds
        two : (B, ) ndarray of bool
            Array of indices into time_two between the rms bounds
        overlap : (C, ) ndarray of bool
            Array of indices into time_overlap between the rms bounds

    Notes
    -----
    #.  Written by David C. Stauffer in May 2020 when it needed to handle datetime64 objects.

    Examples
    --------
    >>> from dstauffman.plotting import get_rms_indices
    >>> import numpy as np
    >>> time_one     = np.arange(11)
    >>> time_two     = np.arange(2, 13)
    >>> time_overlap = np.arange(2, 11)
    >>> xmin         = 1
    >>> xmax         = 8
    >>> ix = get_rms_indices(time_one, time_two, time_overlap, xmin=xmin, xmax=xmax)
    >>> print(ix['pts'])
    [1, 8]

    """
    def _process(time, t_bound, func):
        r"""Determines if the given time should be processed."""
        if is_datetime(time):
            # if datetime, it's either the datetime.datetime version, or np.datetime64 version
            if isinstance(time, datetime.datetime):
                # process if any of the data is in the bound
                process = func(time, t_bound)
            else:
                process = not np.isnat(time)
        else:
            if time is None:
                process = False
            else:
                process = not np.isnan(time) and not np.isinf(time) and func(time, t_bound)
        return process

    # TODO: functionalize this more so there is less repeated code
    # initialize output
    ix = {'pts': [], 'one': np.array([], dtype=bool), 'two': np.array([], dtype=bool), 'overlap': np.array([], dtype=bool)}
    # alias some flags
    have1 = time_one is not None and np.size(time_one) > 0
    have2 = time_two is not None and np.size(time_two) > 0
    have3 = time_overlap is not None
    # get the min/max times
    if have1:
        if have2:
            # have both
            t_min = np.minimum(np.min(time_one), np.min(time_two))
            t_max = np.maximum(np.max(time_one), np.max(time_two))
        else:
            # have only time 1
            t_min = np.min(time_one)
            t_max = np.max(time_one)
    else:
        if have2:
            # have only time 2
            t_min = np.min(time_two)
            t_max = np.max(time_two)
        else:
            # have neither time 1 nor time 2
            raise AssertionError('At least one time vector must be given.')
    if _process(xmin, t_max, operator.lt):
        if have1: p1_min = time_one >= xmin
        if have2: p2_min = time_two >= xmin
        if have3: p3_min = time_overlap >= xmin
        ix['pts'].append(np.maximum(xmin, t_min))
    else:
        if have1: p1_min = np.ones(time_one.shape,     dtype=bool)
        if have2: p2_min = np.ones(time_two.shape,     dtype=bool)
        if have3: p3_min = np.ones(time_overlap.shape, dtype=bool)
        ix['pts'].append(t_min)
    if _process(xmax, t_min, operator.gt):
        if have1: p1_max = time_one <= xmax
        if have2: p2_max = time_two <= xmax
        if have3: p3_max = time_overlap <= xmax
        ix['pts'].append(np.minimum(xmax, t_max))
    else:
        if have1: p1_max = np.ones(time_one.shape,     dtype=bool)
        if have2: p2_max = np.ones(time_two.shape,     dtype=bool)
        if have3: p3_max = np.ones(time_overlap.shape, dtype=bool)
        ix['pts'].append(t_max)
    assert len(ix['pts']) == 2 and ix['pts'][0] <= ix['pts'][1], 'Time points aren\'t as expected: "{}"'.format(ix['pts'])
    # calculate indices
    if have1: ix['one']     = p1_min & p1_max
    if have2: ix['two']     = p2_min & p2_max
    if have3: ix['overlap'] = p3_min & p3_max
    return ix

#%% Functions - plot_vert_lines
def plot_vert_lines(ax, x, *, show_in_legend=True, colormap=None, labels=None):
    r"""
    Plots a vertical line at the RMS start and stop times.

    Summary
    -------
    There are two vertical lines created.  The first at x(1) and the second at x(2).
    The first line is orange, and the second is lavender.  Both have magenta crosses at
    the top and bottom.  The lines are added to the plot regardless of the figure hold state.

    Parameters
    ----------
    ax : class matplotlib.axes.Axes
        Figure axes
    x : (N,) tuple (nominally N=2)
        X values at which to draw the vertical lines
    show_in_legend : bool, optional
        show the lines when a legend is turned on
    colormap : matplotlib.colors.Colormap, optional
        Colormap to use, default has two colors
    labels : list of str, optional
        Labels to use when other than two lines

    Notes
    -----
    #.  Added to Stauffer's MATLAB libary from GARSE in Sept 2013.
    #.  Ported to Python by David C. Stauffer in March 2019.
    #.  Updated by David C. Stauffer in May 2020 to be more generic.

    Examples
    --------
    >>> from dstauffman.plotting import plot_vert_lines
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> _ = ax.plot(np.arange(10), np.arange(10), label='Data')
    >>> x = (2, 5)
    >>> plot_vert_lines(ax, x, show_in_legend=False)
    >>> _ = ax.legend()

    >>> plt.close(fig)

    """
    # optional inputs
    if colormap is None:
        colormap = colors.ListedColormap([(1., 0.75, 0.), (0.75, 0.75, 1.)])
    cm = ColorMap(colormap, num_colors=len(x))
    if labels is None:
        labels = ['RMS Start Time', 'RMS Stop Time']
    # plot vertical lines
    for (i, this_x) in enumerate(x):
        this_color = cm.get_color(i)
        this_label = labels[i] if show_in_legend else ''
        ax.axvline(this_x, linestyle='--', color=this_color, marker='+', markeredgecolor='m', markersize=10, label=this_label)

#%% plot_phases
def plot_phases(ax, times, colormap='tab10', labels=None, *, group_all=False):
    r"""
    Plots some labeled phases as semi-transparent patchs on the given axes.

    Parameters
    ----------
    ax : (Axes)
        Figure axes
    times : (1xN) or (2xN) list of times, if it has two rows, then the second are the end points
         otherwise assume the sections are continuous.

    Examples
    --------
    >>> from dstauffman.plotting import plot_phases, get_color_lists
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
    if not group_all:
        cm = ColorMap(colormap=colormap, num_colors=num_segments)
    elif colormap == 'tab10':
        # change to responible default for group_all case
        colormap = 'xkcd:black'

    # get the limits of the plot
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    # create second row of times if not specified (assumes each phase goes all the way to the next one)
    if times.ndim == 1:
        times = np.vstack((times, np.hstack((times[1:], max(times[-1], xlims[1])))))

    # loop through all the phases
    for i in range(num_segments):
        # get the label and color for this phase
        this_color = cm.get_color(i) if not group_all else colormap
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
            this_label = labels[i] if not group_all else labels
            ax.annotate(this_label, xy=(x1, y2), xycoords='data', horizontalalignment='left', \
                verticalalignment='top', fontsize=15, rotation=-90)

    # reset any limits that might have changed due to the patches
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

#%% Functions - get_classification
def get_classification(classify: str) -> Tuple[str, str]:
    r"""
    Gets the classification and any caveats from the text in OPTS.

    Parameters
    ----------
    classify : (str)
        Text to put on plots for classification purposes

    Returns
    -------
    classification : str)
        Classification to use, from {'U', 'C', 'S', 'TS'}
    caveat : str
        The extra caveats beyond the main classification

    See Also
    --------
    plot_classification

    Notes
    -----
    #.  Written by David C. Stauffer in March 2020.

    Examples
    --------
    >>> from dstauffman.plotting import get_classification
    >>> classify = 'UNCLASSIFIED//MADE UP CAVEAT'
    >>> (classification, caveat) = get_classification(classify)
    >>> print(classification)
    U

    >>> print(caveat)
    //MADE UP CAVEAT

    """
    # check for empty case, default to unclassified
    if not classify:
        # check if not using any classification and if so return empties
        if not DEFAULT_CLASSIFICATION:
            return ('', '')
        # DCS: modify this section if you want a different default on your system (potentially put into a file instead?)
        classify = DEFAULT_CLASSIFICATION

    # get the classification based solely on the first letter and check that it is valid
    classification = classify[0]
    assert classification in {'U', 'C', 'S', 'T'}, 'Unexpected classification of "{}" found'.format(classification)

    # pull out anything past the first // as the caveat(s)
    slashes = classify.find('//')
    if slashes == -1:
        caveat = ''
    else:
        caveat = classify[slashes:]

    return (classification, caveat)

#%% Functions - plot_classification
def plot_classification(ax: Axes, classification: str = 'U', *, caveat: str = '', \
                        test: bool = False, location: str = 'figure'):
    r"""
    Displays the classification in a box on each figure.
    Includes the option of printing another box for testing purposes.

    Parameters
    ----------
    ax : class matplotlib.axes.Axes
        Figure axes
    classification : str
        Level of classification, from {'U', 'C', 'S', 'T', 'TS'}
    caveat : str, optional
        Any additional caveats beyone the classification level
    test : bool, optional
        Whether to print the testing box, default is false
    location : str, optional
        Where to put the label, from {'axes', 'axis', 'figure', 'left', 'top'}

    See Also
    --------
    setup_plots

    Change Log
    ----------
    #.  Written by David C. Stauffer in August 2019 based on Matlab version.
    #.  Updated by David C. Stauffer in March 2020 to support caveats.
    #.  Updated by David C. Stauffer in May 2020 to give more placement options.

    Examples
    --------
    >>> from dstauffman.plotting import plot_classification
    >>> import matplotlib.pyplot as plt
    >>> fig1 = plt.figure()
    >>> ax1 = fig1.add_subplot(111)
    >>> _ = ax1.plot([0, 10], [0, 10], '.-b')
    >>> plot_classification(ax1, 'U', test=False, location='figure')
    >>> plt.show(block=False) # doctest: +SKIP

    >>> fig2 = plt.figure()
    >>> ax2 = fig2.add_subplot(111)
    >>> _ = ax2.plot(0, 0)
    >>> plot_classification(ax2, 'S', caveat='//MADE UP CAVEAT', test=True, location='figure')
    >>> plt.show(block=False) # doctest: +SKIP

    >>> fig3 = plt.figure()
    >>> ax3 = fig3.add_subplot(111)
    >>> _ = ax3.plot(1, 1)
    >>> plot_classification(ax3, 'C', test=True, location='axis')
    >>> plt.show(block=False) # doctest: +SKIP

    >>> plt.close(fig1)
    >>> plt.close(fig2)
    >>> plt.close(fig3)

    """
    # simple check to exit if not using
    if not classification:
        return

    # plot warning before trying to draw the other box
    if test:
        ax.text(0.5, 0.97, 'This plot classification is labeled for test purposes only', \
            color='r', horizontalalignment='center', verticalalignment='top', \
            bbox=dict(facecolor='none', edgecolor='r'), transform=ax.transAxes)

    # add classification box
    if classification == 'U':
        color    = (0., 0., 0.)
        text_str = 'UNCLASSIFIED'
    elif classification == 'C':
        color    = (0., 0., 1.)
        text_str = 'CONFIDENTIAL'
    elif classification in 'S':
        color    = (1., 0., 0.)
        text_str = 'SECRET'
    elif classification in {'TS','T'}:
        color    = (1., 0.65, 0.)
        text_str = 'TOP SECRET'
    else:
        raise ValueError('Unexpected value for classification: "{}".'.format(classification))
    text_color = color

    # add optional caveats
    if caveat:
        text_str += caveat

    # allow other color options for certain caveats
    if '//FAKE COLOR' in caveat:
        color      = (0.0, 0.8, 0.0)
        text_color = (0.2, 0.2, 0.2)

    # add classification box
    if location in {'axes', 'axis'}:
        # inside the axes
        ax.text(0.99, 0.01, text_str, color=text_color, horizontalalignment='right', \
            verticalalignment='bottom', fontweight='bold', fontsize=12, \
            bbox={'facecolor':'none', 'edgecolor':color, 'linewidth':2}, transform=ax.transAxes)
        return
    # other locations within the figure
    vert_align = 'bottom'
    if location == 'figure':
        text_pos   = (1., 0.005)
        horz_align = 'right'
    elif location == 'left':
        text_pos   = (0., 0.005)
        horz_align = 'left'
    elif location == 'top':
        text_pos   = (0., 0.995)
        horz_align = 'left'
        vert_align = 'top'
    else:
        raise ValueError(f'Unexpected location given: "{location}"')
    # create the label
    ax.annotate('\n  ' + text_str + '  ', text_pos, xycoords='figure fraction', \
        color=text_color, weight='bold', fontsize=12, horizontalalignment=horz_align, \
        verticalalignment=vert_align, linespacing=0, annotation_clip=False, \
        bbox=dict(boxstyle='square', facecolor='none', edgecolor=color, linewidth=2))
    # add border
    fig = ax.figure
    r1 = Rectangle((0., 0.), 1., 1., facecolor='none', edgecolor=color, clip_on=False, \
        linewidth=3, transform=fig.transFigure)
    fig.patches.extend([r1])

#%% Functions - align_plots
def align_plots(figs: _FigOrListFig, pos: Tuple[int, int] = None) -> None:
    """
    Aligns all the figures in one location.

    Parameters
    ----------
    figs : list of matplotlib.Figure
        List of figures to align together

    Notes
    -----
    #.  Written by David C. Stauffer in June 2020.

    Examples
    --------
    >>> from dstauffman.plotting import align_plots, make_time_plot
    >>> #fig1 = make_time_plot('Plot 1', 0, 0) # TODO: get this working
    >>> #fig2 = make_time_plot('Plot 2', 1, 1)
    >>> #figs = [fig1, fig2]
    >>> #align_plots(figs)

    Close plots
    >>> #plt.close(fig1)
    >>> #plt.close(fig2)

    """
    # initialize position if given
    x_pos: Optional[int] = None
    y_pos: Optional[int] = None
    if pos is not None:
        (x_pos, y_pos) = pos
    # loop through figures
    for fig in figs:
        # use position from first plot if you don't already have it
        if x_pos is None or y_pos is None:
            (x_pos, y_pos, _, _) = fig.canvas.manager.window.geometry().getRect()
        # move the plot
        fig.canvas.manager.window.move(x_pos, y_pos)

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
    >>> from dstauffman.plotting import z_from_ci
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
    unittest.main(module='dstauffman.tests.test_plotting_support', exit=False)
    doctest.testmod(verbose=False)
