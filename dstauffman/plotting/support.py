r"""
Defines useful plotting utilities.

Notes
-----
#.  Written by David C. Stauffer in March 2015.
"""  # pylint: disable=too-many-lines

#%% Imports
from __future__ import annotations

import datetime
import doctest
from functools import partial
import gc
from itertools import repeat
import operator
import os
from pathlib import Path
import platform
import re
import sys
from typing import Dict, List, Literal, Optional, overload, Tuple, TYPE_CHECKING, Union
import unittest
import warnings

try:
    from qtpy.QtCore import QSize
    from qtpy.QtGui import QIcon
    from qtpy.QtWidgets import QApplication, QPushButton

    _HAVE_QT = True
except ModuleNotFoundError:
    warnings.warn("Qt (PyQt5, PyQt6, PySide2, PySide6) was not found. Some funtionality will be limited.")
    QPushButton = object  # type: ignore[assignment, misc]
    _HAVE_QT = False

from dstauffman import (
    convert_date,
    Frozen,
    get_images_dir,
    get_username,
    HAVE_DS,
    HAVE_MPL,
    HAVE_NUMPY,
    HAVE_PANDAS,
    HAVE_SCIPY,
    is_datetime,
    IS_WINDOWS,
)

if HAVE_MPL:
    from matplotlib.axes import Axes
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.cm as cmx
    import matplotlib.colors as colors

    # Newer date stamps on axes, done here as this is the lowest level of the plotting submodule
    from matplotlib.dates import ConciseDateConverter, date2num
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt
    import matplotlib.units as munits

    converter = ConciseDateConverter()
    munits.registry[datetime.date] = converter
    munits.registry[datetime.datetime] = converter
    if HAVE_DS:
        import datashader as ds
        from datashader.mpl_ext import alpha_colormap, dsshow
        import datashader.transfer_functions as tf
if HAVE_NUMPY:
    import numpy as np

    inf = np.inf
    if HAVE_MPL:
        munits.registry[np.datetime64] = converter
else:
    from math import inf
if HAVE_PANDAS:
    import pandas as pd
if HAVE_SCIPY:
    import scipy.stats as st
try:
    from PIL import Image
except ImportError:
    pass

#%% Constants
# Default colormap to use on certain plots
DEFAULT_COLORMAP: str = "Dark2"  # "Paired", "Dark2", "tab10", "tab20"

# Whether to include a classification on any generated plots
DEFAULT_CLASSIFICATION: str = ""

if TYPE_CHECKING:
    _FigOrListFig = Union[Figure, List[Figure]]

COLOR_LISTS: Dict[str, colors.ListedColormap] = {}
if HAVE_MPL:
    # fmt: off
    # default colormap
    COLOR_LISTS["default"]  = cmx.get_cmap(DEFAULT_COLORMAP)
    assert isinstance(COLOR_LISTS["default"], colors.ListedColormap), "Expecting a ListedColormap for the default."
    # single colors
    COLOR_LISTS["same"]     = colors.ListedColormap(tuple(repeat(cmx.get_cmap(DEFAULT_COLORMAP).colors[0], 8)))
    COLOR_LISTS["same_old"] = colors.ListedColormap(tuple(repeat("#1f77b4", 8)))
    COLOR_LISTS["single"]   = colors.ListedColormap(("xkcd:red",))
    # doubles
    COLOR_LISTS["double"]   = colors.ListedColormap(("xkcd:red", "xkcd:blue"))
    COLOR_LISTS["dbl_off"]  = colors.ListedColormap(("xkcd:fuchsia", "xkcd:cyan"))
    # triples
    COLOR_LISTS["vec"]      = colors.ListedColormap(("xkcd:red", "xkcd:green", "xkcd:blue"))
    COLOR_LISTS["vec_off"]  = colors.ListedColormap(("xkcd:fuchsia", "xkcd:lightgreen", "xkcd:cyan"))
    # quads
    COLOR_LISTS["quat"]     = colors.ListedColormap(("xkcd:red", "xkcd:green", "xkcd:blue", "xkcd:chocolate"))
    COLOR_LISTS["quat_off"] = colors.ListedColormap(("xkcd:fuchsia", "xkcd:lightgreen", "xkcd:cyan", "xkcd:brown"))
    # double combinations
    COLOR_LISTS["dbl_diff"]    = colors.ListedColormap(COLOR_LISTS["dbl_off"].colors + COLOR_LISTS["double"].colors)
    COLOR_LISTS["dbl_diff_r"]  = colors.ListedColormap(COLOR_LISTS["double"].colors + COLOR_LISTS["dbl_off"].colors)
    # triple combinations
    COLOR_LISTS["vec_diff"]    = colors.ListedColormap(COLOR_LISTS["vec_off"].colors + COLOR_LISTS["vec"].colors)
    COLOR_LISTS["vec_diff_r"]  = colors.ListedColormap(COLOR_LISTS["vec"].colors + COLOR_LISTS["vec_off"].colors)
    # quad combinations
    COLOR_LISTS["quat_diff"]   = colors.ListedColormap(COLOR_LISTS["quat_off"].colors + COLOR_LISTS["quat"].colors)
    COLOR_LISTS["quat_diff_r"] = colors.ListedColormap(COLOR_LISTS["quat"].colors + COLOR_LISTS["quat_off"].colors)
    # fmt: on

#%% Set Matplotlib global settings
if HAVE_MPL:
    plt.rcParams["figure.dpi"] = 160  # 160 for 4K monitors, 100 otherwise
    plt.rcParams["figure.figsize"] = [11.0, 8.5]  # makes figures the same size as the paper, keeping aspect ratios even
    plt.rcParams["figure.max_open_warning"] = 80  # Max number of figures to open before through a warning, 0 means unlimited
    plt.rcParams["date.autoformatter.minute"] = "%H:%M:%S"  # makes seconds show, and not day, default is "%d %H:%M"

# Whether a display exists to draw on or not
# TODO: make this public?
_HAVE_DISPLAY = IS_WINDOWS or bool(os.environ.get("DISPLAY", None))

#%% Classes - _HoverButton
class _HoverButton(QPushButton):
    r"""Custom button that allows hovering and icons."""

    def __init__(self, *args, **kwargs):
        # initialize
        super().__init__(*args, **kwargs)
        # Enable mouse hover event tracking
        self.setMouseTracking(True)
        self.setStyleSheet("border: 0px;")
        # set icon
        for this_arg in args:
            if isinstance(this_arg, QIcon):
                self.setIcon(this_arg)
                self.setIconSize(QSize(24, 24))

    def enterEvent(self, event):  # pylint: disable=unused-argument
        r"""Draw border on hover."""
        self.setStyleSheet("border: 1px; border-style: solid;")  # pragma: no cover

    def leaveEvent(self, event):  # pylint: disable=unused-argument
        r"""Delete border after hover."""
        self.setStyleSheet("border: 0px;")  # pragma: no cover


#%% Classes - MyCustomToolbar
class MyCustomToolbar:
    r"""
    Define a custom toolbar to use in any matplotlib plots.

    Examples
    --------
    >>> from dstauffman.plotting import MyCustomToolbar
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig = plt.figure()
    >>> fig.canvas.manager.set_window_title("Figure Title")
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
        # check to see if a display exists and if not, then return without creating buttons
        if not _HAVE_DISPLAY:
            return
        # check to see if a QApplication exists, and if not, make one
        if QApplication.instance() is None:
            self.qapp = QApplication(sys.argv)  # pragma: no cover
        else:
            self.qapp = QApplication.instance()
        # Store the figure number for use later (Note this works better than relying on plt.gcf()
        # to determine which figure actually triggered the button events.)
        self.fig_number = fig.number
        # Check if you have a canvas to draw on, and if not, return without creating buttons
        if fig.canvas.toolbar is None:
            return
        # create buttons - Prev Plot
        icon = QIcon(str(get_images_dir() / "prev_plot.png"))
        self.btn_prev_plot = _HoverButton(icon, "")
        self.btn_prev_plot.setToolTip("Show the previous plot")
        fig.canvas.toolbar.addWidget(self.btn_prev_plot)
        self.btn_prev_plot.clicked.connect(self.prev_plot)
        # create buttons - Next Plot
        icon = QIcon(str(get_images_dir() / "next_plot.png"))
        self.btn_next_plot = _HoverButton(icon, "")
        self.btn_next_plot.setToolTip("Show the next plot")
        fig.canvas.toolbar.addWidget(self.btn_next_plot)
        self.btn_next_plot.clicked.connect(self.next_plot)
        # create buttons - Close all
        icon = QIcon(str(get_images_dir() / "close_all.png"))
        self.btn_close_all = _HoverButton(icon, "")
        self.btn_close_all.setToolTip("Close all the open plots")
        fig.canvas.toolbar.addWidget(self.btn_close_all)
        self.btn_close_all.clicked.connect(self._close_all)

    def _close_all(self, *args):  # pylint: disable=unused-argument
        r"""Close all the currently open plots."""
        close_all()

    def next_plot(self, *args):  # pylint: disable=unused-argument
        r"""Bring up the next plot in the series."""
        # get all the figure numbers
        all_figs = plt.get_fignums()
        # get the active figure number
        this_fig = self.fig_number
        # loop through all the figures
        for i in range(len(all_figs)):  # pylint: disable=consider-using-enumerate
            # find the active figure within the list
            if this_fig == all_figs[i]:
                # find the next figure, with allowances for rolling over the list
                if i < len(all_figs) - 1:
                    next_fig = all_figs[i + 1]
                else:
                    next_fig = all_figs[0]
        # set the appropriate active figure
        fig = plt.figure(next_fig)
        # make it the active window
        fig.canvas.manager.window.raise_()

    def prev_plot(self, *args):  # pylint: disable=unused-argument
        r"""Bring up the previous plot in the series."""
        # get all the figure numbers
        all_figs = plt.get_fignums()
        # get the active figure number
        this_fig = self.fig_number
        # loop through all the figures
        for i in range(len(all_figs)):  # pylint: disable=consider-using-enumerate
            # find the active figure within the list
            if this_fig == all_figs[i]:
                # find the next figure, with allowances for rolling over the list
                if i > 0:
                    prev_fig = all_figs[i - 1]
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
    >>> cm = ColorMap("Paired", num_colors=12)
    >>> time = np.arange(0, 10, 0.1)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> _ = ax.plot(time, np.sin(time), color=cm.get_color(0))
    >>> _ = ax.plot(time, np.cos(time), color=cm.get_color(1))
    >>> _ = ax.legend(["Sin", "Cos"])
    >>> plt.show(block=False) # doctest: +SKIP

    Close plot
    >>> plt.close(fig)

    """

    def __init__(self, colormap=DEFAULT_COLORMAP, low=0, high=1, num_colors=None):
        self.num_colors = num_colors
        # check for optional inputs
        if self.num_colors is not None:
            low = 0
            high = num_colors - 1
        elif isinstance(colormap, colors.ListedColormap):
            low = 0
            high = colormap.N
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
            ax.set_prop_cycle("color", [self.get_color(i) for i in range(self.num_colors)])
        except AttributeError:  # pragma: no cover
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
    # plt.close("all"), as that can sometimes cause the iPython kernel to quit #DCS: 2015-06-11
    if figs is None:
        for this_fig in plt.get_fignums():
            plt.close(this_fig)
    else:
        for this_fig in figs:
            plt.close(this_fig)
    gc.collect()


#%% Functions - get_nondeg_colorlists
def get_nondeg_colorlists(num_channels: int) -> colors.ListedColormap:
    r"""
    Get a nice colormap for the given number of channels to plot and use for non-deg comparisons.

    Parameters
    ----------
    num_channels : int
        Number of channels to plot

    Returns
    -------
    clist : matplotlib.colors.ListedColormap
        Ordered colormap with the given list of colors (times three)

    Notes
    -----
    #.  This function returns three times the number of colors you need, with the first two sets
        visually related to each other, and the third as a repeat of the first.
    #.  Written by David C. Stauffer in March 2021.

    Examples
    --------
    >>> from dstauffman.plotting import get_nondeg_colorlists
    >>> num_channels = 2
    >>> clist = get_nondeg_colorlists(num_channels)
    >>> print(clist.colors)
    ('xkcd:red', 'xkcd:blue', 'xkcd:fuchsia', 'xkcd:cyan', 'xkcd:red', 'xkcd:blue')

    """
    if num_channels == 1:
        clist = colors.ListedColormap(("#1f77b4", "xkcd:blue", "#1f77b4"))
    elif num_channels == 2:
        clist = colors.ListedColormap(COLOR_LISTS["dbl_diff_r"].colors + COLOR_LISTS["double"].colors)
    elif num_channels == 3:
        clist = colors.ListedColormap(COLOR_LISTS["vec_diff_r"].colors + COLOR_LISTS["vec"].colors)
    elif num_channels == 4:
        clist = colors.ListedColormap(COLOR_LISTS["quat_diff_r"].colors + COLOR_LISTS["quat"].colors)
    else:
        ix = [x % 10 for x in range(num_channels)]
        cmap1 = cmx.get_cmap("tab10")
        cmap2 = cmx.get_cmap("tab20")
        temp = (
            tuple(cmap1.colors[x] for x in ix) + tuple(cmap2.colors[2 * x + 1] for x in ix) + tuple(cmap1.colors[x] for x in ix)
        )
        clist = colors.ListedColormap(temp)
    return clist


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
    if data is None or np.all(data == None):  # noqa: E711  # pylint: disable=singleton-comparison
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
    new_color = tuple((c * (1 - dt) + w * dt for (c, w) in zip(color, white)))
    return new_color


#%% Functions - get_figure_title
@overload
def get_figure_title(fig: Figure, raise_warning: Literal[False] = ...) -> str:
    ...


@overload
def get_figure_title(fig: Figure, raise_warning: Literal[True]) -> Tuple[str, bool]:
    ...


def get_figure_title(fig: Figure, raise_warning: bool = False) -> Union[str, Tuple[str, bool]]:
    r"""
    Gets the name of the given figure.  First trying the canvas, then the suptitle, then the title.

    Parameters
    ----------
    fig : class Figure
        Figure to get the title from
    raise_warning : bool, optional
        Whether to return a flag about any warnings

    Returns
    -------
    raw_title : str
        Figure title
    throw_warning : bool
        Whether a warning should be thrown

    Notes
    -----
    #.  Functionalized out of storefig by David C. Stauffer in May 2022 to use elsewhere.

    Examples
    --------
    >>> from dstauffman.plotting import plot_time_history, get_figure_title
    >>> import matplotlib.pyplot as plt
    >>> fig = plot_time_history("My Title", 0, 0)
    >>> title = get_figure_title(fig)
    >>> print(title)
    My Title

    Close plot
    >>> plt.close(fig)

    """
    # preallocate if a warning should be thrown
    throw_warning = False
    # get the title of the figure canvas
    raw_title = fig.canvas.manager.get_window_title()
    if raw_title is None or raw_title == "image":
        # special case when you have a displayless backend, check the suptitle, then the title
        # from the first axes
        throw_warning = True
        if (sup := fig._suptitle) is not None:  # pylint: disable=protected-access
            raw_title = sup.get_text()
        else:
            try:
                raw_title = fig.axes[0].get_title()
            except:
                pass
    if raw_title is None:
        raw_title = "None"
    if raise_warning:
        return raw_title, throw_warning
    return raw_title  # type: ignore[no-any-return]


#%% Functions - resolve_name
def resolve_name(name: str, force_win: bool = None, rep_token: str = "_", strip_classification: bool = True) -> str:
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
    >>> name = "(U//FOUO) Bad name /\ <>!"
    >>> force_win = True
    >>> new_name = resolve_name(name, force_win=force_win)
    >>> print(new_name)
    Bad name __ __!

    """
    # hard-coded values
    bad_chars_win = ["<", ">", ":", '"', "/", "\\", "|", "?", "*"]
    bad_chars_unix = ["/"]

    # determine OS and thus which characters are bad
    if force_win is None:
        is_windows = platform.system() == "Windows"
    else:
        is_windows = force_win
    bad_chars = bad_chars_win if is_windows else bad_chars_unix

    # initialize output
    new_name = name

    # strip any leading classification text
    if strip_classification:
        new_name = re.sub(r"^\(\S*\)\s", "", new_name, count=1)

    # replace any bad characters with underscores
    for ch in bad_chars:
        if ch in new_name:
            new_name = new_name.replace(ch, rep_token)
    return new_name


#%% Functions - storefig
def storefig(
    fig: _FigOrListFig, folder: Union[str, Path] = None, plot_type: Union[str, List[str]] = "png", show_warn: bool = True
) -> None:
    r"""
    Store the specified figures in the specified folder and with the specified plot type(s).

    Parameters
    ----------
    fig : list or single figure
        Figure object(s) to save to disk
    folder : str
        Location to save figures to
    plot_type : str
        Type of figure to save to disk, like "png" or "jpg"
    show_warn : bool, optional, default is True
        Whether to show a warning if the plot title is used instead of thewindow canvas (i.e. you don't have a display)

    Raises
    ------
    ValueError
        Specified folder to save figures to doesn't exist.

    Notes
    -----
    #.  Uses the figure.canvas.manager.get_window_title property to determine the figure name.  If
        that is not set or default ("image"), then it tries the figure suptitle or first axes title.

    See Also
    --------
    matplotlib.pyplot.savefig, titleprefix

    Examples
    --------
    Create figure and then save to disk
    >>> from dstauffman.plotting import storefig
    >>> from dstauffman import get_tests_dir
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig = plt.figure()
    >>> fig.canvas.manager.set_window_title("Figure Title")
    >>> ax = fig.add_subplot(111)
    >>> x = np.arange(0, 10, 0.1)
    >>> y = np.sin(x)
    >>> _ = ax.plot(x, y)
    >>> _ = ax.set_title("X vs Y")
    >>> plt.show(block=False) # doctest: +SKIP
    >>> folder = get_tests_dir()
    >>> plot_type = "png"
    >>> storefig(fig, folder, plot_type)

    Close plot
    >>> plt.close(fig)

    Delete potential file(s)
    >>> folder.joinpath("Figure Title.png").unlink(missing_ok=True)
    >>> folder.joinpath("X vs Y.png").unlink(missing_ok=True)

    """
    # make sure figs is a list
    if isinstance(fig, list):
        figs = fig
    else:
        figs = [fig]
    # make sure folder is a Path
    if isinstance(folder, str):
        folder = Path(folder).resolve()
    # make sure types is a list
    if not isinstance(plot_type, list):
        types: List[str] = []
        types.append(plot_type)
    else:
        types = plot_type
    # if no folder was specified, then use the current working directory
    if folder is None:
        folder = Path.cwd()  # pragma: no cover
    # confirm that the folder exists
    if not folder.is_dir():
        raise ValueError(f'The specfied folder "{folder}" does not exist.')
    # loop through the figures
    throw_warning = False
    for this_fig in figs:
        (raw_title, need_warning) = get_figure_title(this_fig, raise_warning=True)
        throw_warning |= need_warning
        this_title = resolve_name(raw_title)
        # loop through the plot types
        for this_type in types:
            # save the figure to the specified plot type
            this_fig.savefig(folder.joinpath(this_title + "." + this_type), dpi=160, bbox_inches="tight", pad_inches=0.01)
    if throw_warning and show_warn:
        warnings.warn("No window titles found, using the plot title instead (usually because there is no display).")


#%% Functions - titleprefix
def titleprefix(fig: _FigOrListFig, prefix: str = "", process_all: bool = False) -> None:
    r"""
    Prepend a text string to all the titles on existing figures.

    It also sets the canvas title used by storefig when saving to a file.

    Parameters
    ----------
    fig : list or single figure
        Figure object(s) to save to disk
    prefix : str
        Text to be prepended to the title and figure name
    process_all : bool, optional, default is False
        Whether to process all the children axes even if a suptitle is found

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
    >>> fig.canvas.manager.set_window_title("Figure Title")
    >>> ax = fig.add_subplot(111)
    >>> x = np.arange(0, 10, 0.1)
    >>> y = np.sin(x)
    >>> _ = ax.plot(x, y)
    >>> _ = ax.set_title("X vs Y")
    >>> plt.show(block=False) # doctest: +SKIP
    >>> prefix = "Baseline"
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
        # update canvas name
        this_canvas_title = this_fig.canvas.manager.get_window_title()
        this_fig.canvas.manager.set_window_title(prefix + " - " + this_canvas_title)
        # update the suptitle (if it exists)
        if (sup := this_fig._suptitle) is not None:  # pylint: disable=protected-access
            sup.set_text(prefix + " - " + sup.get_text())
        elif process_all or sup is None:
            # get axes list and loop through them
            for this_axis in this_fig.axes:
                # get title for this axis
                this_title = this_axis.get_title()
                # if the title is empty, then don't do anything
                if not this_title:
                    continue
                # modify and set new title
                new_title = prefix + " - " + this_title
                this_axis.set_title(new_title)


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
    >>> fig.canvas.manager.set_window_title("Figure Title")
    >>> ax = fig.add_subplot(111)
    >>> x = np.arange(0, 10, 0.1)
    >>> y = np.sin(x)
    >>> _ = ax.plot(x, y)
    >>> _ = ax.set_title("X vs Y")
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
            raise ValueError("Unexpected item that is neither a figure nor axes.")
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
    >>> fig.canvas.manager.set_window_title("Figure Title")
    >>> ax = fig.add_subplot(111)
    >>> time = np.arange(1, 10, 0.1)
    >>> data = time ** 2
    >>> _ = ax.plot(time, data)
    >>> _ = ax.set_title("X vs Y")
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
    # exit if the plotted data are not numeric
    if not np.issubdtype(data.dtype, np.number):
        return
    # convert datetimes as appropriate for comparisons
    if is_datetime(time):
        time = date2num(time)
    # find the relevant time indices
    ix_time = (time >= t_start) & (time <= t_final)
    # exit if no data is in this time window
    if ~np.any(ix_time):
        warnings.warn("No data matched the given time interval.")
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
        raise ValueError("The pad cannot be negative.")
    if pad > 0:
        delta = this_ymax - this_ymin
        this_ymax += pad * delta
        this_ymin -= pad * delta
    # check for the case where the data is constant and the limits are the same
    if this_ymin == this_ymax:
        if this_ymin == 0:
            # data is exactly zero, show from -1 to 1
            this_ymin = -1
            this_ymax = 1
        else:
            # data is constant, pad by given amount or 10% if pad is zero
            pad = pad if pad > 0 else 0.1
            this_ymin = (1 - pad) * this_ymin
            this_ymax = (1 + pad) * this_ymax
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
    >>> fig.canvas.manager.set_window_title("Figure Title")
    >>> ax = fig.add_subplot(111)
    >>> x = np.arange(0, 10, 0.1)
    >>> y = np.sin(x)
    >>> _ = ax.plot(x, y)
    >>> _ = ax.set_title("X vs Y")
    >>> _ = ax.set_xlabel("time [years]")
    >>> _ = ax.set_ylabel("value [radians]")
    >>> plt.show(block=False) # doctest: +SKIP
    >>> figmenu(fig)

    Close plot
    >>> plt.close(fig)

    """
    if not _HAVE_QT:
        return
    if not isinstance(figs, list):
        figs.toolbar_custom_ = MyCustomToolbar(figs)
    else:
        for fig in figs:
            fig.toolbar_custom_ = MyCustomToolbar(fig)


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
    hex_code = f"#{clamp(r):02x}{clamp(g):02x}{clamp(b):02x}"
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
    #.  There doesn't seem to be one standard way to do this, thus this function.  Changed in
        February 2022 as .desktop became deprecated.  The .screens method will hopefully work
        for a while instead.
    #.  Don't call this function from within a GUI, as it will close everything.  Just
        query the desktop directly within the GUI.

    Examples
    --------
    >>> from dstauffman.plotting import get_screen_resolution
    >>> (screen_width, screen_height) = get_screen_resolution()
    >>> print("{}x{}".format(screen_width, screen_height)) # doctest: +SKIP

    """
    # if you don't have a display, then return zeros
    if not _HAVE_DISPLAY:
        return (0, 0)
    # check to see if a QApplication exists, and if not, make one
    app: QApplication
    if QApplication.instance() is None:
        app = QApplication(sys.argv)  # pragma: no cover
    else:
        app = QApplication.instance()  # type: ignore[assignment]
    # query the resolution
    screen_resolution = app.screens()[0].geometry()
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

    Notes
    -----
    #.  Updated by David C. Stauffer in March 2022 to pad by 10% when the results have a narrow range.

    Examples
    --------
    >>> from dstauffman.plotting import show_zero_ylim
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> _ = ax.plot([1, 5, 10], [200, 250, 240], ".-")
    >>> show_zero_ylim(ax)

    >>> plt.close(fig)

    """
    # get the current y-limits, and the delta between them
    ylim = ax.get_ylim()
    dt = ylim[1] - ylim[0]
    changed = False
    # set zero to always be included
    if min(ylim) > 0:
        changed = True
        ax.set_ylim(bottom=0)
        new_dt = ylim[1] - 0
    if max(ylim) < 0:
        assert not changed, "Should never change both axes."
        changed = True
        ax.set_ylim(top=0)
        new_dt = 0 - ylim[0]
    # if the old values fit within the top 5% of the original values, then pad by 10%
    if changed and abs(dt) < abs(0.05 * new_dt):
        ax.set_ylim(*(1.1 * x for x in ax.get_ylim()))


#%% Functions - plot_second_units_wrapper
def plot_second_units_wrapper(ax: Axes, second_units: Union[None, int, float, Tuple[str, float]]) -> Axes:
    r"""
    Wrapper to plot_second_yunits that allows numeric or dict options.

    Parameters
    ----------
    ax : class matplotlib.axes.Axes
        Figure axes
    second_units : dict or int or float
        Scale factor to apply, or dict with key for label and value for factor

    Returns
    -------
    ax2 : class matplotlib.axes.Axes
        New Figure axes with the second label

    Notes
    -----
    #.  If second_units is just a number, then no units are displayed, but if a key and value,
        then if it has brakets, replace the entire label, otherwise only replace what is in the
        old label within the brackets

    Examples
    --------
    >>> from dstauffman.plotting import plot_second_units_wrapper
    >>> import matplotlib.pyplot as plt
    >>> description = "Values over time"
    >>> ylabel = "Value [rad]"
    >>> second_units = ("Better Units [µrad]", 1e6)
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> _ = ax.plot([1, 5, 10], [1e-6, 3e-6, 2.5e-6], ".-")
    >>> _ = ax.set_ylabel(ylabel)
    >>> _ = ax.set_title(description)
    >>> _ = plot_second_units_wrapper(ax, second_units)

    >>> plt.close(fig)

    """
    # initialize output
    ax2 = None
    # check if processing anything
    if second_units is not None:
        # determine what type of input was given
        if isinstance(second_units, (int, float)):
            label = ""
            value = second_units
        else:
            label = second_units[0]
            value = second_units[1]
        # check if we got a no-op value
        if not np.isnan(value) and value != 0 and value != 1:
            # if all is good, build the new label and call the lower level function
            old_label = ax.get_ylabel()
            ix1 = old_label.find("[")
            ix2 = label.find("[")
            if ix2 >= 0:
                # new label has units, so use them
                new_label = label
            elif ix1 >= 0 and label:
                # new label is only units, replace them in the old label
                new_label = old_label[:ix1] + "[" + label + "]"
            else:
                # neither label has units, just label them
                new_label = label
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
    >>> _ = ax.plot([1, 5, 10], [1e-6, 3e-6, 2.5e-6], ".-")
    >>> _ = ax.set_ylabel("Value [rad]")
    >>> ylab = "Value [µrad]"
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
    >>> print(ix["pts"])
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
    ix = {"pts": [], "one": np.array([], dtype=bool), "two": np.array([], dtype=bool), "overlap": np.array([], dtype=bool)}
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
            raise AssertionError("At least one time vector must be given.")
    if _process(xmin, t_max, operator.lt):
        if have1:
            p1_min = time_one >= xmin
        if have2:
            p2_min = time_two >= xmin
        if have3:
            p3_min = time_overlap >= xmin
        ix["pts"].append(np.maximum(xmin, t_min))
    else:
        if have1:
            p1_min = np.ones(time_one.shape, dtype=bool)
        if have2:
            p2_min = np.ones(time_two.shape, dtype=bool)
        if have3:
            p3_min = np.ones(time_overlap.shape, dtype=bool)
        ix["pts"].append(t_min)
    if _process(xmax, t_min, operator.gt):
        if have1:
            p1_max = time_one <= xmax
        if have2:
            p2_max = time_two <= xmax
        if have3:
            p3_max = time_overlap <= xmax
        ix["pts"].append(np.minimum(xmax, t_max))
    else:
        if have1:
            p1_max = np.ones(time_one.shape, dtype=bool)
        if have2:
            p2_max = np.ones(time_two.shape, dtype=bool)
        if have3:
            p3_max = np.ones(time_overlap.shape, dtype=bool)
        ix["pts"].append(t_max)
    assert len(ix["pts"]) == 2 and ix["pts"][0] <= ix["pts"][1], f'Time points aren\'t as expected: "{ix["pts"]}"'
    # calculate indices
    if have1:
        ix["one"] = p1_min & p1_max
    if have2:
        ix["two"] = p2_min & p2_max
    if have3:
        ix["overlap"] = p3_min & p3_max
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
    >>> _ = ax.plot(np.arange(10), np.arange(10), label="Data")
    >>> x = (2, 5)
    >>> plot_vert_lines(ax, x, show_in_legend=False)
    >>> _ = ax.legend()

    >>> plt.close(fig)

    """
    # optional inputs
    if colormap is None:
        colormap = colors.ListedColormap([(1.0, 0.75, 0.0), (0.75, 0.75, 1.0)])
    cm = ColorMap(colormap, num_colors=len(x))
    if labels is None:
        labels = ["RMS Start Time", "RMS Stop Time"]
    # plot vertical lines
    for (i, this_x) in enumerate(x):
        this_color = cm.get_color(i)
        this_label = labels[i] if show_in_legend else ""
        ax.axvline(this_x, linestyle="--", color=this_color, marker="+", markeredgecolor="m", markersize=10, label=this_label)


#%% plot_phases
def plot_phases(ax, times, colormap="tab10", labels=None, *, group_all=False):
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
    >>> from dstauffman.plotting import plot_phases, COLOR_LISTS
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> fig = plt.figure()
    >>> fig.canvas.manager.set_window_title("Sine Wave")
    >>> ax = fig.add_subplot(111)
    >>> time = np.arange(101)
    >>> data = np.cos(time / 10)
    >>> _ = ax.plot(time, data, ".-")
    >>> times = np.array([5, 20, 60, 90])
    >>> # times = np.array([[5, 20, 60, 90], [10, 60, 90, 95]])
    >>> labels = ["Part 1", "Phase 2", "Watch Out", "Final"]
    >>> colors = COLOR_LISTS["quat"]
    >>> plot_phases(ax, times, colors, labels)
    >>> plt.show(block=False) # doctest: +SKIP

    Close plot
    >>> plt.close(fig)

    """
    # hard-coded values
    transparency = 0.2  # 1 = opaque

    # get number of segments
    if times.ndim == 1:
        num_segments = times.size
    else:
        num_segments = times.shape[1]

    # check for optional arguments
    if not group_all:
        cm = ColorMap(colormap=colormap, num_colors=num_segments)
    elif colormap == "tab10":
        # change to responible default for group_all case
        colormap = "xkcd:black"

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
        if is_datetime(x1):
            # convert to floats, as the annotate command can't handle this case
            x1 = date2num(x1)
            x2 = date2num(x2)
        # create the shaded box
        ax.add_patch(
            Rectangle(
                (x1, 0),
                x2 - x1,
                1,
                facecolor=this_color,
                edgecolor=this_color,
                alpha=transparency,
                transform=ax.get_xaxis_transform(),
                clip_on=True,
            )
        )
        # create the label
        if labels is not None:
            this_label = labels[i] if not group_all else labels
            if bool(this_label):
                ax.annotate(
                    this_label,
                    xy=(x1, 0.99),
                    xycoords=ax.get_xaxis_transform(),
                    horizontalalignment="left",
                    verticalalignment="top",
                    fontsize=15,
                    rotation=-90,
                    clip_on=True,
                )

    # reset any limits that might have changed due to the patches
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)


#%% Functions - get_classification
def get_classification(classify: str) -> Tuple[str, str]:
    r"""
    Gets the classification and any caveats from the text in the classify string.

    Parameters
    ----------
    classify : (str)
        Text to put on plots for classification purposes

    Returns
    -------
    classification : str)
        Classification to use, from {"U", "C", "S", "TS"}
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
    >>> classify = "UNCLASSIFIED//MADE UP CAVEAT"
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
            return ("", "")
        # DCS: modify this section if you want a different default on your system (potentially put into a file instead?)
        classify = DEFAULT_CLASSIFICATION

    # get the classification based solely on the first letter and check that it is valid
    classification = classify[0]
    assert classification in frozenset({"U", "C", "S", "T"}), f'Unexpected classification of "{classification}" found'

    # pull out anything past the first // as the caveat(s)
    slashes = classify.find("//")
    if slashes == -1:
        caveat = ""
    else:
        caveat = classify[slashes:]

    return (classification, caveat)


#%% Functions - plot_classification
def plot_classification(ax: Axes, classification: str = "U", *, caveat: str = "", test: bool = False, location: str = "figure"):
    r"""
    Displays the classification in a box on each figure.
    Includes the option of printing another box for testing purposes.

    Parameters
    ----------
    ax : class matplotlib.axes.Axes
        Figure axes
    classification : str
        Level of classification, from {"U", "C", "S", "T", "TS"}
    caveat : str, optional
        Any additional caveats beyone the classification level
    test : bool, optional
        Whether to print the testing box, default is false
    location : str, optional
        Where to put the label, from {"axes", "axis", "figure", "left", "top"}

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
    >>> _ = ax1.plot([0, 10], [0, 10], ".-b")
    >>> plot_classification(ax1, "U", test=False, location="figure")
    >>> plt.show(block=False) # doctest: +SKIP

    >>> fig2 = plt.figure()
    >>> ax2 = fig2.add_subplot(111)
    >>> _ = ax2.plot(0, 0)
    >>> plot_classification(ax2, "S", caveat="//MADE UP CAVEAT", test=True, location="figure")
    >>> plt.show(block=False) # doctest: +SKIP

    >>> fig3 = plt.figure()
    >>> ax3 = fig3.add_subplot(111)
    >>> _ = ax3.plot(1, 1)
    >>> plot_classification(ax3, "C", test=True, location="axis")
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
        ax.text(
            0.5,
            0.97,
            "This plot classification is labeled for test purposes only",
            color="r",
            horizontalalignment="center",
            verticalalignment="top",
            bbox=dict(facecolor="none", edgecolor="r"),
            transform=ax.transAxes,
        )

    # add classification box
    if classification == "U":
        color = (0.0, 0.0, 0.0)
        text_str = "UNCLASSIFIED"
    elif classification == "C":
        color = (0.0, 0.0, 1.0)
        text_str = "CONFIDENTIAL"
    elif classification in "S":
        color = (1.0, 0.0, 0.0)
        text_str = "SECRET"
    elif classification in {"TS", "T"}:
        color = (1.0, 0.65, 0.0)
        text_str = "TOP SECRET"
    else:
        raise ValueError(f'Unexpected value for classification: "{classification}".')
    text_color = color

    # add optional caveats
    if caveat:
        text_str += caveat

    # allow other color options for certain caveats
    if "//FAKE COLOR" in caveat:
        color = (0.0, 0.8, 0.0)
        text_color = (0.2, 0.2, 0.2)

    # add classification box
    if location in {"axes", "axis"}:
        # inside the axes
        ax.text(
            0.99,
            0.01,
            text_str,
            color=text_color,
            horizontalalignment="right",
            verticalalignment="bottom",
            fontweight="bold",
            fontsize=12,
            bbox={"facecolor": "none", "edgecolor": color, "linewidth": 2},
            transform=ax.transAxes,
        )
        return
    # other locations within the figure
    vert_align = "bottom"
    if location == "figure":
        text_pos = (1.0, 0.005)
        horz_align = "right"
    elif location == "left":
        text_pos = (0.0, 0.005)
        horz_align = "left"
    elif location == "top":
        text_pos = (0.0, 0.995)
        horz_align = "left"
        vert_align = "top"
    else:
        raise ValueError(f'Unexpected location given: "{location}"')
    # create the label
    ax.annotate(
        "\n  " + text_str + "  ",
        text_pos,
        xycoords="figure fraction",
        color=text_color,
        weight="bold",
        fontsize=12,
        horizontalalignment=horz_align,
        verticalalignment=vert_align,
        linespacing=0,
        annotation_clip=False,
        bbox=dict(boxstyle="square", facecolor="none", edgecolor=color, linewidth=2),
    )
    # add border
    fig = ax.figure
    r1 = Rectangle(
        (0.0, 0.0), 1.0, 1.0, facecolor="none", edgecolor=color, clip_on=False, linewidth=3, transform=fig.transFigure
    )
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
    >>> #fig1 = make_time_plot("Plot 1", 0, 0) # TODO: get this working
    >>> #fig2 = make_time_plot("Plot 2", 1, 1)
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
def z_from_ci(ci: Union[int, float]) -> float:
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
    >>> print("{:.2f}".format(z))
    1.96

    """
    return st.norm.ppf(1 - (1 - ci) / 2)  # type: ignore[no-any-return]


#%% Functions - ci_from_z
def ci_from_z(z: Union[int, float]) -> float:
    r"""
    Calculates the confidence interval that matches the Z score.

    Parameters
    ----------
    z : float
        Desired z value

    Returns
    -------
    ci : float
        Desired confidence interval

    Notes
    -----
    #.  Written by David C. Stauffer in April 2022.

    Examples
    --------
    >>> from dstauffman.plotting import ci_from_z
    >>> z = 2
    >>> ci = ci_from_z(z)
    >>> print("{:.4f}".format(ci))
    0.9545

    """
    return st.norm.cdf(z) - st.norm.cdf(-z)  # type: ignore[no-any-return]


#%% Functions - save_figs_to_pdf
def save_figs_to_pdf(figs: Union[Figure, List[Figure]] = None, filename: Path = Path("figs.pdf")) -> None:
    r"""
    Saves the given figures to a PDF file.

    Parameters
    ----------
    figs : figure or List[figure] or None
        Figures to save, None means save all open figures
    filename : str, optional
        Name of the file to save the figures to, defaults to "figs.pdf" in the current folder

    Notes
    -----
    #.  Written by David C. Stauffer in March 2021.

    Examples
    --------
    >>> from dstauffman.plotting import close_all, plot_time_history, save_figs_to_pdf
    >>> from dstauffman import get_tests_dir
    >>> fig = plot_time_history("test", 0, 0)
    >>> filename = get_tests_dir() / "figs.pdf"
    >>> save_figs_to_pdf(fig, filename)  # doctest: +SKIP

    Delete file and close figure
    >>> filename.unlink(missing_ok=True)
    >>> close_all([fig])

    """
    # Optional inputs
    if figs is None:
        figs = plt.get_fignums()
    if isinstance(figs, Figure):
        figs = [figs]
    assert isinstance(figs, list)

    # Create PDF
    with PdfPages(filename) as pdf:
        for fig in figs:
            pdf.savefig(fig)

        # Set metadata for PDF file
        d = pdf.infodict()
        d["Title"] = "PDF Figures"
        d["Author"] = get_username()
        d["CreationDate"] = datetime.datetime.now()
        d["ModDate"] = d["CreationDate"]


#%% Functions - save_images_to_pdf
def save_images_to_pdf(
    figs: Union[Figure, List[Figure]] = None, folder: Path = None, plot_type: str = "png", filename: Path = Path("figs.pdf")
):
    r"""
    Uses figure names to find the already saved images and combine them into a PDF file.

    Parameters
    ----------
    figs : figure or List[figure] or None
        Figures to save, None means save all open figures
    folder : Path, optional
        Folder to load the figures from
    plot_type: str, optional
        Type of figure to try and load from
    filename : str, optional
        Name of the file to save the figures to, defaults to "figs.pdf" in the current folder

    Notes
    -----
    #.  Written by David C. Stauffer in August 2021.
    #.  Note that save_figs_to_pdf saves vectorized images to PDF and is usually the better
        solution. This function is intended to be called just after storefig, to use the PNG
        versions as jpgs instead, which is better if there are several hundred thousand
        points in the plots.

    Examples
    --------
    >>> from dstauffman.plotting import close_all, plot_time_history, save_images_to_pdf, storefig
    >>> from dstauffman import get_tests_dir
    >>> fig = plot_time_history("test", 0, 0)
    >>> folder = get_tests_dir()
    >>> filename = get_tests_dir() / "figs.pdf"
    >>> plot_type = "png"
    >>> storefig(fig, folder, plot_type)
    >>> save_images_to_pdf(fig, folder, plot_type, filename)

    Delete file and close figure
    >>> filename.unlink(missing_ok=True)
    >>> image_filename = folder / "test.png"
    >>> image_filename.unlink(missing_ok=True)
    >>> close_all([fig])

    """
    # Optional inputs
    if figs is None:
        figs = [plt.figure(i) for i in plt.get_fignums()]
    if isinstance(figs, Figure):
        figs = [figs]
    assert isinstance(figs, list)
    if folder is None:
        folder = Path.cwd()

    # create the metadata
    meta: Dict[str, Union[str, datetime.datetime]] = {}
    meta["Title"] = "PDF Figures"
    meta["Author"] = get_username()
    meta["CreationDate"] = datetime.datetime.now()
    meta["ModDate"] = meta["CreationDate"]

    # Create PDF of images
    images = []
    for (ix, fig) in enumerate(figs):
        fig_title = get_figure_title(fig)
        this_image = folder.joinpath(fig_title + "." + plot_type)
        image_rgba = Image.open(this_image)
        image_jpg = image_rgba.convert("RGB")
        if ix == 0:
            im = image_jpg
        else:
            images.append(image_jpg)
    im.save(filename, save_all=True, append_images=images, metadata=meta)  # TODO: metadata not saving?


#%% add_datashaders
def add_datashaders(datashaders):
    r"""Adds the collection of datashaders to the axes."""
    if not HAVE_DS:
        raise RuntimeError("You must have datashader installed to execute this.")
    # overlay the datashaders
    for this_ds in datashaders:
        # check for dates and convert as appropriate
        if is_datetime(this_ds["time"]):
            df = pd.DataFrame(
                {"time": convert_date(this_ds["time"], "matplotlib", old_form=this_ds["time_units"]), "data": this_ds["data"]}
            )
        else:
            df = pd.DataFrame({"time": this_ds["time"], "data": this_ds["data"]})
        if "value" in this_ds:
            df["value"] = this_ds["value"]
        # TODO: check for strings on Y axis and convert to values instead
        this_axes = this_ds["ax"]
        if "color" in this_ds:
            cmap = alpha_colormap(this_ds["color"], min_alpha=40, max_alpha=255)
        elif "colormap" in this_ds:
            cmap = this_ds["colormap"]
        else:
            raise ValueError(f"Color information was not in datashader with keys: {this_ds.keys()}")
        vmin = this_ds.get("vmin", None)
        vmax = this_ds.get("vmax", None)
        aspect = this_ds.get("aspect", "auto")
        agg = ds.mean("value") if "value" in this_ds and this_ds["value"] is not None else ds.count()
        norm = this_ds.get("norm", "log")
        dsshow(
            df,
            ds.Point("time", "data"),
            agg,
            norm=norm,
            cmap=cmap,
            ax=this_axes,
            aspect=aspect,
            vmin=vmin,
            vmax=vmax,
            x_range=this_axes.get_xlim(),
            y_range=this_axes.get_ylim(),
            shade_hook=partial(tf.dynspread, threshold=0.8, max_px=6, how="over"),
        )


#%% fig_ax_factory
@overload
def fig_ax_factory(
    num_figs: Optional[int],
    num_axes: Union[int, List[int]],
    *,
    suptitle: Union[str, List[str]],
    layout: str,
    sharex: bool,
    passthrough: Literal[False] = ...,
) -> Tuple[Tuple[Figure, Axes], ...]:
    ...


@overload
def fig_ax_factory(
    num_figs: Optional[int],
    num_axes: Union[int, List[int]],
    *,
    suptitle: Union[str, List[str]],
    layout: str,
    sharex: bool,
    passthrough: Literal[True],
) -> Tuple[None, ...]:
    ...


def fig_ax_factory(
    num_figs: int = None,
    num_axes: Union[int, List[int]] = 1,
    *,
    suptitle: Union[str, List[str]] = "",
    layout: str = "rows",
    sharex: bool = True,
    passthrough: bool = False,
) -> Union[Tuple[Tuple[Figure, Axes], ...], Tuple[None, ...]]:
    r"""
    Creates the figures and axes for use in a given plotting function.

    Parameters
    ----------
    num_figs
    suptitle
    num_axes
    layout
    sharex
    passthrough

    Notes
    -----
    #.  Written by David C. Stauffer in February 2022.

    Examples
    --------
    >>> from dstauffman.plotting import fig_ax_factory
    >>> import matplotlib.pyplot as plt
    >>> fig_ax = fig_ax_factory()
    >>> (fig, ax) = fig_ax[0]
    >>> assert isinstance(fig, plt.Figure)
    >>> assert isinstance(ax, plt.Axes)

    Close plot
    >>> plt.close(fig)

    """
    if isinstance(num_axes, int):
        is_1d = True
        if layout == "rows":
            num_row = num_axes
            num_col = 1
        elif layout == "cols":
            num_row = 1
            num_col = num_axes
        else:
            raise ValueError(f'Unexpected layout: "{layout}".')
    else:
        is_1d = False
        if layout not in {"rowwise", "colwise"}:
            raise ValueError(f'Unexpected layout: "{layout}".')
        num_row = num_axes[0]
        num_col = num_axes[1]
    if num_figs is None:
        num_figs = 1
    if passthrough:
        return tuple(None for _ in range(num_figs * num_row * num_col))
    figs: List[Figure] = []
    axes: Union[List[Axes], List[List[Axes]], List[List[List[Axes]]]] = []
    for i in range(num_figs):
        (fig, ax) = plt.subplots(num_row, num_col, sharex=sharex)
        if bool(suptitle):
            this_title = suptitle[i] if isinstance(suptitle, list) else suptitle
            fig.suptitle(this_title)
            fig.canvas.manager.set_window_title(this_title)
        figs.append(fig)
        axes.append(ax)
    fig_ax: Tuple[Tuple[Figure, Axes], ...]
    if is_1d:
        assert isinstance(num_axes, int)
        if num_axes == 1:
            fig_ax = tuple((figs[f], axes[f]) for f in range(num_figs))
        else:
            fig_ax = tuple((figs[f], axes[f][i]) for f in range(num_figs) for i in range(num_axes))
    else:
        if layout == "rowwise":
            fig_ax = tuple((figs[f], axes[f][i, j]) for f in range(num_figs) for i in range(num_row) for j in range(num_col))  # type: ignore[call-overload]
        elif layout == "colwise":
            fig_ax = tuple((figs[f], axes[f][i, j]) for f in range(num_figs) for j in range(num_col) for i in range(num_row))  # type: ignore[call-overload]
    return fig_ax


#%% Unit test
if __name__ == "__main__":
    plt.ioff()
    unittest.main(module="dstauffman.tests.test_plotting_support", exit=False)
    doctest.testmod(verbose=False)
