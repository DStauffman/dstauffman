r"""
Defines useful plotting utilities.

Notes
-----
#.  Written by David C. Stauffer in March 2015.

"""  # pylint: disable=too-many-lines

# %% Imports
from __future__ import annotations

import datetime
import doctest
from functools import partial
import gc
import io
from itertools import repeat
import operator
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any, Callable, Literal, overload, Protocol, TYPE_CHECKING, TypedDict
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
except ImportError:
    warnings.warn("QTPY was found, but failed to import, likely due to a missing library dependency.")
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
    import matplotlib as mpl
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
    from PIL.PngImagePlugin import PngInfo
except ModuleNotFoundError:
    pass

if TYPE_CHECKING:
    from numpy.typing import NDArray

    _B = NDArray[np.bool_]
    _CM = str | colors.Colormap | colors.ListedColormap  # + ColorMap defined below
    _D = NDArray[np.datetime64]
    _I = NDArray[np.int_]
    _N = NDArray[np.floating]
    _M = NDArray[np.floating]  # 2D
    _Time = int | float | np.floating | datetime.datetime | datetime.date | np.datetime64 | None
    _Times = int | float | np.floating | datetime.datetime | np.datetime64 | _D | _I | _N | list[_N] | list[_D] | tuple[_N, ...] | tuple[_D, ...] | None  # fmt: skip
    _Data = int | float | np.floating | _I | _N | _M | list[_I] | list[_N] | list[_I | _N] | tuple[_I, ...] | tuple[_N, ...] | tuple[_I | _N, ...] | None  # fmt: skip
    _FigOrListFig = Figure | list[Figure]

    class _RmsIndices(TypedDict):
        pts: list[int]
        one: _B
        two: _B
        overlap: _B


# %% Constants
# Default colormap to use on certain plots
DEFAULT_COLORMAP: str = "Dark2"  # "Paired", "Dark2", "tab10", "tab20"

# Whether to include a classification on any generated plots
DEFAULT_CLASSIFICATION: str = ""

COLOR_LISTS: dict[str, colors.ListedColormap] = {}
if HAVE_MPL:
    # fmt: off
    # default colormap
    assert isinstance(mpl.colormaps[DEFAULT_COLORMAP], colors.ListedColormap), "Expecting a ListedColormap for the default."
    COLOR_LISTS["default"]  = mpl.colormaps[DEFAULT_COLORMAP]  # type: ignore[assignment]
    # single colors
    COLOR_LISTS["same"]     = colors.ListedColormap(tuple(repeat(mpl.colormaps[DEFAULT_COLORMAP].colors[0], 8)))  # type: ignore[attr-defined]
    COLOR_LISTS["same_old"] = colors.ListedColormap(tuple(repeat("#1f77b4", 8)))
    COLOR_LISTS["single"]   = colors.ListedColormap(("#1f77b4",))  # tab20 first color
    COLOR_LISTS["sing_off"] = colors.ListedColormap(("#aec7e8",))  # tab20 second color
    # doubles
    COLOR_LISTS["double"]   = colors.ListedColormap(("xkcd:red", "xkcd:blue"))
    COLOR_LISTS["dbl_off"]  = colors.ListedColormap(("xkcd:fuchsia", "xkcd:cyan"))
    # triples
    COLOR_LISTS["vec"]      = colors.ListedColormap(("xkcd:red", "xkcd:green", "xkcd:blue"))
    COLOR_LISTS["vec_off"]  = colors.ListedColormap(("xkcd:fuchsia", "xkcd:lightgreen", "xkcd:cyan"))
    # quads
    COLOR_LISTS["quat"]     = colors.ListedColormap(("xkcd:red", "xkcd:green", "xkcd:blue", "xkcd:chocolate"))
    COLOR_LISTS["quat_off"] = colors.ListedColormap(("xkcd:fuchsia", "xkcd:lightgreen", "xkcd:cyan", "xkcd:brown"))
    # single combinations
    COLOR_LISTS["sing_diff"]   = colors.ListedColormap(COLOR_LISTS["sing_off"].colors + COLOR_LISTS["single"].colors)  # type: ignore[operator]
    COLOR_LISTS["sing_diff_r"] = colors.ListedColormap(COLOR_LISTS["single"].colors + COLOR_LISTS["sing_off"].colors)  # type: ignore[operator]
    COLOR_LISTS["sing_comp"]   = colors.ListedColormap(("xkcd:red", "xkcd:green", "xkcd:blue"))  # Note: this intentionally breaks the pattern
    COLOR_LISTS["sing_comp_r"] = colors.ListedColormap(("xkcd:blue", "xkcd:green", "xkcd:red"))  # Note: this intentionally breaks the pattern
    # double combinations
    COLOR_LISTS["dbl_diff"]    = colors.ListedColormap(COLOR_LISTS["dbl_off"].colors + COLOR_LISTS["double"].colors)  # type: ignore[operator]
    COLOR_LISTS["dbl_diff_r"]  = colors.ListedColormap(COLOR_LISTS["double"].colors + COLOR_LISTS["dbl_off"].colors)  # type: ignore[operator]
    COLOR_LISTS["dbl_comp"]    = colors.ListedColormap(COLOR_LISTS["dbl_diff"].colors + COLOR_LISTS["double"].colors)  # type: ignore[operator]
    COLOR_LISTS["dbl_comp_r"]  = colors.ListedColormap(COLOR_LISTS["dbl_diff_r"].colors + COLOR_LISTS["double"].colors)  # type: ignore[operator]
    # triple combinations
    COLOR_LISTS["vec_diff"]    = colors.ListedColormap(COLOR_LISTS["vec_off"].colors + COLOR_LISTS["vec"].colors)  # type: ignore[operator]
    COLOR_LISTS["vec_diff_r"]  = colors.ListedColormap(COLOR_LISTS["vec"].colors + COLOR_LISTS["vec_off"].colors)  # type: ignore[operator]
    COLOR_LISTS["vec_comp"]    = colors.ListedColormap(COLOR_LISTS["vec_diff"].colors + COLOR_LISTS["vec"].colors)  # type: ignore[operator]
    COLOR_LISTS["vec_comp_r"]  = colors.ListedColormap(COLOR_LISTS["vec_diff_r"].colors + COLOR_LISTS["vec"].colors)  # type: ignore[operator]
    # quad combinations
    COLOR_LISTS["quat_diff"]   = colors.ListedColormap(COLOR_LISTS["quat_off"].colors + COLOR_LISTS["quat"].colors)  # type: ignore[operator]
    COLOR_LISTS["quat_diff_r"] = colors.ListedColormap(COLOR_LISTS["quat"].colors + COLOR_LISTS["quat_off"].colors)  # type: ignore[operator]
    COLOR_LISTS["quat_comp"]   = colors.ListedColormap(COLOR_LISTS["quat_diff"].colors + COLOR_LISTS["quat"].colors)  # type: ignore[operator]
    COLOR_LISTS["quat_comp_r"] = colors.ListedColormap(COLOR_LISTS["quat_diff_r"].colors + COLOR_LISTS["quat"].colors)  # type: ignore[operator]
    # Matlab
    COLOR_LISTS["matlab"]      = colors.ListedColormap(("#0072bd", "#d95319", "#edb120", "#7e2f8e", "#77ac30", "#4dbeee", "#a2142f"))
    COLOR_LISTS["matlab_old"]  = colors.ListedColormap(("#0000ff", "#008000", "#ff0000", "#00bfbf", "#bf00bf", "#bfbf00", "#404040"))
    # fmt: on

# %% Set Matplotlib global settings
if HAVE_MPL:
    plt.rcParams["figure.dpi"] = 160  # 160 for 4K monitors, 100 otherwise
    plt.rcParams["figure.figsize"] = [11.0, 8.5]  # makes figures the same size as the paper, keeping aspect ratios even
    plt.rcParams["figure.max_open_warning"] = 80  # Max number of figures to open before through a warning, 0 means unlimited
    plt.rcParams["date.autoformatter.minute"] = "%H:%M:%S"  # makes seconds show, and not day, default is "%d %H:%M"

# Whether a display exists to draw on or not
# TODO: make this public?
_HAVE_DISPLAY = IS_WINDOWS or bool(os.environ.get("DISPLAY", None))

_QUALITATIVE_COLORMAPS: frozenset[str] = frozenset(
    {"Pastel1", "Pastel2", "Paired", "Accent", "Dark2", "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c"}
)


# %% Classes - _HoverButton
class _HoverButton(QPushButton):
    r"""Custom button that allows hovering and icons."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
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

    def enterEvent(self, event: Any) -> None:  # pylint: disable=unused-argument
        r"""Draw border on hover."""
        self.setStyleSheet("border: 1px; border-style: solid;")  # pragma: no cover

    def leaveEvent(self, event: Any) -> None:  # pylint: disable=unused-argument
        r"""Delete border after hover."""
        self.setStyleSheet("border: 0px;")  # pragma: no cover


# %% Classes - ExtraPlotter
class ExtraPlotter(Protocol):
    r"""Custom Protocol to type the extra_plotter argument to all the plots."""

    def __call__(self, fig: Figure, ax: list[Axes]) -> None: ...  # noqa: D102


# %% Classes - MyCustomToolbar
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

    def __init__(self, fig: Figure) -> None:
        r"""Initialize the custom toolbar."""
        # check to see if a display exists and if not, then return without creating buttons
        if not _HAVE_DISPLAY:
            return
        # check to see if a QApplication exists, and if not, make one
        if QApplication.instance() is None:
            self.qapp = QApplication(sys.argv)  # pragma: no cover
        else:
            self.qapp = QApplication.instance()  # type: ignore[assignment]
        # Store the figure number for use later (Note this works better than relying on plt.gcf()
        # to determine which figure actually triggered the button events.)
        self.fig_number = fig.number
        # Check if you have a canvas to draw on, and if not, return without creating buttons
        if fig.canvas.toolbar is None or is_notebook():
            return
        # create buttons - Prev Plot
        icon = QIcon(str(get_images_dir() / "prev_plot.png"))
        self.btn_prev_plot = _HoverButton(icon, "")
        self.btn_prev_plot.setToolTip("Show the previous plot")
        fig.canvas.toolbar.addWidget(self.btn_prev_plot)  # type: ignore[attr-defined]
        self.btn_prev_plot.clicked.connect(self.prev_plot)
        # create buttons - Next Plot
        icon = QIcon(str(get_images_dir() / "next_plot.png"))
        self.btn_next_plot = _HoverButton(icon, "")
        self.btn_next_plot.setToolTip("Show the next plot")
        fig.canvas.toolbar.addWidget(self.btn_next_plot)  # type: ignore[attr-defined]
        self.btn_next_plot.clicked.connect(self.next_plot)
        # create buttons - Close all
        icon = QIcon(str(get_images_dir() / "close_all.png"))
        self.btn_close_all = _HoverButton(icon, "")
        self.btn_close_all.setToolTip("Close all the open plots")
        fig.canvas.toolbar.addWidget(self.btn_close_all)  # type: ignore[attr-defined]
        self.btn_close_all.clicked.connect(self._close_all)

    def _close_all(self, *args: Any) -> None:  # pylint: disable=unused-argument
        r"""Close all the currently open plots."""
        close_all()

    def next_plot(self, *args: Any) -> None:  # pylint: disable=unused-argument
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
        assert fig.canvas.manager is not None
        fig.canvas.manager.window.raise_()  # type: ignore[attr-defined]

    def prev_plot(self, *args: Any) -> None:  # pylint: disable=unused-argument
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
        assert fig.canvas.manager is not None
        fig.canvas.manager.window.raise_()  # type: ignore[attr-defined]


# %% Classes - ColorMap
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
    >>> plt.show(block=False)  # doctest: +SKIP

    Close plot
    >>> plt.close(fig)

    """

    def __init__(
        self,
        colormap: _CM | ColorMap | None = DEFAULT_COLORMAP,
        low: int | float = 0,
        high: int | float = 1,
        num_colors: int | None = None,
    ) -> None:
        self.num_colors = num_colors
        # check for optional inputs
        if self.num_colors is not None:
            low = 0
            high = num_colors - 1  # type: ignore[operator]
        elif isinstance(colormap, colors.ListedColormap):
            low = 0
            high = colormap.N - 1
        # get colormap based on high and low limits
        if colormap is None:
            cmap = plt.get_cmap(DEFAULT_COLORMAP)
        elif isinstance(colormap, colors.Colormap):
            cmap = colormap
        elif isinstance(colormap, type(self)):
            cmap = None
        else:
            assert isinstance(colormap, str), "Only expecting string colormaps to get to this point."
            if colormap in _QUALITATIVE_COLORMAPS:
                # special case to always use in the given order
                cmap = plt.get_cmap(colormap)
                assert isinstance(cmap, colors.ListedColormap), "Qualitative colormaps are assumed to be listed."
                low = 0
                high = cmap.N - 1
            else:
                cmap = plt.get_cmap(colormap, self.num_colors)
            if isinstance(cmap, colors.ListedColormap):
                low = 0
                high = cmap.N - 1
        if cmap is None:
            self.smap = colormap.get_smap()  # type: ignore[union-attr]
        else:
            cnorm = colors.Normalize(vmin=low, vmax=high)
            self.smap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)
        # must initialize the empty scalar mapplable to show the colorbar correctly
        self.smap.set_array([])

    def get_color(self, value: float | int | np.floating | np.int_) -> tuple[float, float, float, float]:
        r"""Get the color based on the scalar value."""
        return self.smap.to_rgba(value)  # type: ignore[arg-type, return-value]

    def get_smap(self) -> cmx.ScalarMappable:
        r"""Return the smap being used."""
        return self.smap

    def set_colors(self, ax: Axes) -> None:
        r"""Set the colors for the given axes based on internal instance information."""
        if self.num_colors is None:
            raise ValueError("You can't call ColorMap.set_colors unless it was given a num_colors input.")
        try:
            ax.set_prop_cycle("color", [self.get_color(i) for i in range(self.num_colors)])
        except AttributeError:  # pragma: no cover
            # for older matplotlib versions, use deprecated set_color_cycle
            ax.set_color_cycle([self.get_color(i) for i in range(self.num_colors)])  # type: ignore[attr-defined]

    def _repr_png_(self) -> bytes | None:
        """Generate a PNG representation of the Colormap."""
        pixels = np.zeros((64, 512, 4), dtype=np.uint8)
        if self.num_colors is not None:
            cuts = np.floor(np.linspace(0, 512, self.num_colors + 1)).astype(int)
            for i in range(self.num_colors):
                pixels[:, cuts[i] : cuts[i + 1], :] = 255 * np.asanyarray(self.get_color(i))
        else:
            assert self.smap.norm.vmin is not None
            assert self.smap.norm.vmax is not None
            for i, j in enumerate(np.linspace(self.smap.norm.vmin, self.smap.norm.vmax, 255)):
                pixels[:, 2 * i : 2 * (i + 1)] = 255 * np.asanyarray(self.get_color(j))
        png_bytes = io.BytesIO()
        title = "ColorMap"
        author = "dstauffman.plotting"
        try:
            # for when PIL is not installed
            pnginfo = PngInfo()
        except Exception:  # pylint: disable=broad-exception-caught
            return None
        pnginfo.add_text("Title", title)
        pnginfo.add_text("Description", title)
        pnginfo.add_text("Author", author)
        pnginfo.add_text("Software", author)
        Image.fromarray(pixels).save(png_bytes, format="png", pnginfo=pnginfo)
        return png_bytes.getvalue()


# %% Functions - is_notebook
def is_notebook() -> bool:
    r"""
    Determines if you are running in a Jupyter/IPython notebook or not.

    Outputs
    -------
    bool
        Whether you are in a notebook

    Notes
    -----
    #.  Taken from https://stackoverflow.com/a/39662359/5042185 in December 2022.

    Examples
    --------
    >>> from dstauffman.plotting import is_notebook
    >>> print(is_notebook())
    False

    """
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        if shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        return False  # Other unknown type (?)
    except NameError:
        return False  # Likely standard Python interpreter


# %% Functions - close_all
def close_all(fig: _FigOrListFig | None = None) -> None:
    r"""
    Close all the open figures, or if a list is specified, then close all of them.

    Parameters
    ----------
    fig : list of Figures or single Figure, optional
        Specific figures to be closed.

    Examples
    --------
    >>> from dstauffman.plotting import close_all
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> _ = ax.plot(0, 0)
    >>> close_all(fig)

    """
    # Note that it's better to loop through and close the plots individually than to use
    # plt.close("all"), as that can sometimes cause the iPython kernel to quit #DCS: 2015-06-11
    # Note that it's better to clear the figure, then close it to really ensure the memory is
    # returned #DCS: 2024-05-28
    # Note that we are filtering matplotlib UserWarning's to avoid bug
    # https://github.com/matplotlib/matplotlib/issues/9970 #DCS: 2024-06-10
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=UserWarning, module=r"matplotlib\..*")
        if fig is None:
            for this_fig in plt.get_fignums():
                plt.figure(this_fig).clear()
                plt.close(this_fig)
        else:
            if isinstance(fig, Figure):
                fig.clear()
                plt.close(fig)
            else:
                for this_fig in fig:  # type: ignore[assignment]
                    this_fig.clear()  # type: ignore[attr-defined]
                    plt.close(this_fig)
    gc.collect()


# %% Functions - get_nondeg_colorlists
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
        clist = COLOR_LISTS["dbl_comp_r"]
    elif num_channels == 3:
        clist = COLOR_LISTS["vec_comp_r"]
    elif num_channels == 4:
        clist = COLOR_LISTS["quat_comp_r"]
    else:
        ix = [x % 10 for x in range(num_channels)]
        cmap1 = mpl.colormaps["tab10"]
        cmap2 = mpl.colormaps["tab20"]
        temp = (
            tuple(cmap1.colors[x] for x in ix) + tuple(cmap2.colors[2 * x + 1] for x in ix) + tuple(cmap1.colors[x] for x in ix)  # type: ignore[attr-defined]
        )
        clist = colors.ListedColormap(temp)
    return clist


# %% Functions - ignore_plot_data
def ignore_plot_data(data: _Data | None, ignore_empties: bool, col: int | None = None) -> bool:
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
    >>> data = np.zeros((3, 10))
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
        ignore = np.all((data[:, col] == 0) | np.isnan(data[:, col]))  # type: ignore[call-overload, index]
    return ignore  # type: ignore[return-value]


# %% Functions - whitten
def whitten(color: tuple[float, ...], white: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0), dt: float = 0.30) -> tuple[float, ...]:
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


# %% Functions - get_figure_title
@overload
def get_figure_title(fig: Figure, raise_warning: Literal[False] = ...) -> str: ...
@overload
def get_figure_title(fig: Figure, raise_warning: Literal[True]) -> tuple[str, bool]: ...
def get_figure_title(fig: Figure, raise_warning: bool = False) -> str | tuple[str, bool]:
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
    raw_title: str | None
    # get the title of the figure canvas
    if (manager := fig.canvas.manager) is None:
        raw_title = "None"
        throw_warning = True
    else:
        raw_title = manager.get_window_title()
        if raw_title is None or raw_title == "image":
            # special case when you have a displayless backend, check the suptitle, then the title
            # from the first axes
            throw_warning = True
            if (sup := fig._suptitle) is not None:  # type: ignore[attr-defined]  # pylint: disable=protected-access
                raw_title = sup.get_text()
            else:
                try:
                    raw_title = fig.axes[0].get_title()
                except Exception:  # pylint: disable=broad-exception-caught
                    pass
        if raw_title is None:
            raw_title = "None"
    # by this point raw_title should always be set to something, either "None" or "image" or a valid value
    assert raw_title is not None
    if raise_warning:
        return raw_title, throw_warning
    return raw_title


# %% Functions - resolve_name
def resolve_name(name: str, force_win: bool | None = None, rep_token: str = "_", strip_classification: bool = True) -> str:
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
    >>> name = r"(U//FOUO) Bad name /\ <>!"
    >>> force_win = True
    >>> new_name = resolve_name(name, force_win=force_win)
    >>> print(new_name)
    Bad name __ __!

    """
    # hard-coded values
    bad_chars_win = ["<", ">", ":", '"', "/", "\\", "|", "?", "*", "\n"]
    bad_chars_unix = ["/", "\n"]

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


# %% Functions - storefig
def storefig(
    fig: _FigOrListFig,
    folder: str | Path | None = None,
    plot_type: str | list[str] = "png",
    show_warn: bool = True,
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
    >>> plt.show(block=False)  # doctest: +SKIP
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
        types: list[str] = []
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


# %% Functions - titleprefix
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
    >>> plt.show(block=False)  # doctest: +SKIP
    >>> prefix = "Baseline"
    >>> titleprefix(fig, prefix)
    >>> plt.draw()  # doctest: +SKIP

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
        # get the manager
        manager = this_fig.canvas.manager
        assert manager is not None
        # update canvas name
        this_canvas_title = manager.get_window_title()
        manager.set_window_title(prefix + " - " + this_canvas_title)
        # update the suptitle (if it exists)
        if (sup := this_fig._suptitle) is not None:  # type: ignore[attr-defined]  # pylint: disable=protected-access
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


# %% Functions - disp_xlimits
def disp_xlimits(  # noqa: C901
    fig_or_axis: Figure | Axes | list[Figure | Axes], xmin: _Time | None = None, xmax: _Time | None = None
) -> None:
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
    >>> plt.show(block=False)  # doctest: +SKIP
    >>> xmin = 2
    >>> xmax = 5
    >>> disp_xlimits(fig, xmin, xmax)
    >>> plt.draw()  # doctest: +SKIP

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
            if is_datetime(xmin):  # type: ignore[arg-type]
                new_xmin = np.maximum(date2num(xmin), old_xmin)
            else:
                new_xmin = np.max([xmin, old_xmin])
        else:
            new_xmin = old_xmin
        if xmax is not None:
            if is_datetime(xmax):  # type: ignore[arg-type]
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


# %% Functions - zoom_ylim
def zoom_ylim(  # noqa: C901
    ax: Axes,
    time: _Times | None = None,
    data: _Data | None = None,
    *,
    t_start: _Time = -inf,
    t_final: _Time = inf,
    channel: int | None = None,
    pad: float = 0.1,
) -> None:
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
    pad : float
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
    >>> plt.show(block=False)  # doctest: +SKIP

    Zoom X-axis and show how Y doesn't rescale
    >>> t_start = 3
    >>> t_final = 5.0001
    >>> disp_xlimits(fig, t_start, t_final)
    >>> plt.draw()  # doctest: +SKIP

    Force Y-axis to rescale to data
    >>> zoom_ylim(ax, time, data, t_start=t_start, t_final=t_final, pad=0)
    >>> plt.draw()  # doctest: +SKIP

    Close plot
    >>> plt.close(fig)

    """
    if time is None and data is None and not bool(ax.lines):
        warnings.warn("No data found on plot, so nothing was zoomed.")
        return
    # If not given, find time/data from the plot itself
    if time is None:
        time = np.hstack([artist.get_xdata() for artist in ax.lines])
    if data is None:
        data = np.hstack([artist.get_ydata() for artist in ax.lines])
    # exit if the plotted data are not numeric
    if not np.issubdtype(data.dtype, np.number):  # type: ignore[union-attr]
        return
    # convert datetimes as appropriate for comparisons
    if is_datetime(time):
        time = date2num(time)
    # find the relevant time indices
    ix_time = (time >= t_start) & (time <= t_final)  # type: ignore[call-overload, operator]
    # exit if no data is in this time window
    if ~np.any(ix_time):
        warnings.warn("No data matched the given time interval.")
        return
    # pull out the minimums/maximums from the data
    if channel is None:
        if data.ndim == 1:  # type: ignore[union-attr]
            this_ymin = np.min(data[ix_time])  # type: ignore[index]
            this_ymax = np.max(data[ix_time])  # type: ignore[index]
        else:
            this_ymin = np.min(data[ix_time, :])  # type: ignore[call-overload, index]
            this_ymax = np.max(data[ix_time, :])  # type: ignore[call-overload, index]
    else:
        this_ymin = np.min(data[ix_time, channel])  # type: ignore[call-overload, index]
        this_ymax = np.max(data[ix_time, channel])  # type: ignore[call-overload, index]
    # optionally pad the bounds
    if pad < 0.0:
        raise ValueError("The pad cannot be negative.")
    if pad > 0.0:
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
            pad = pad if pad > 0.0 else 0.1
            this_ymin = (1 - pad) * this_ymin
            this_ymax = (1 + pad) * this_ymax
    # get the current limits
    (old_ymin, old_ymax) = ax.get_ylim()
    # compare the new bounds to the old ones and update as appropriate
    if this_ymin > old_ymin:
        ax.set_ylim(bottom=this_ymin)  # type: ignore[arg-type]
    if this_ymax < old_ymax:
        ax.set_ylim(top=this_ymax)  # type: ignore[arg-type]


# %% Functions - figmenu
def figmenu(fig: _FigOrListFig) -> None:
    r"""
    Add a custom toolbar to the figures.

    Parameters
    ----------
    fig : class matplotlib.pyplot.Figure, or list of such
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
    >>> plt.show(block=False)  # doctest: +SKIP
    >>> figmenu(fig)

    Close plot
    >>> plt.close(fig)

    """
    if not _HAVE_QT:
        return
    if not isinstance(fig, list):
        fig.toolbar_custom_ = MyCustomToolbar(fig)  # type: ignore[attr-defined]
    else:
        for this_fig in fig:
            this_fig.toolbar_custom_ = MyCustomToolbar(this_fig)  # type: ignore[attr-defined]


# %% rgb_ints_to_hex
def rgb_ints_to_hex(int_tuple: tuple[int, int, int]) -> str:
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

    def clamp(x: int, min_: int = 0, max_: int = 255) -> int:
        r"""Clamps a value within the specified minimum and maximum."""
        return max(min_, min(x, max_))

    (r, g, b) = int_tuple
    hex_code = f"#{clamp(r):02x}{clamp(g):02x}{clamp(b):02x}"
    return hex_code


# %% Functions - get_screen_resolution
def get_screen_resolution() -> tuple[int, int]:
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
    >>> print("{}x{}".format(screen_width, screen_height))  # doctest: +SKIP

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


# %% Functions - show_zero_ylim
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


# %% Functions - plot_second_units_wrapper
def plot_second_units_wrapper(ax: Axes, second_units: int | float | tuple[str, float] | None) -> Axes | None:
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


# %% Functions - plot_second_yunits
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
    # calculate new limits, with checks for overflows
    try:
        with np.errstate(over="raise"):
            new_limits = tuple(np.multiply(multiplier, ax.get_ylim()))
    except FloatingPointError:
        new_limits = (np.finfo(float).min, np.finfo(float).max)
    # plot second Y axis
    ax2 = ax.twinx()
    ax2.set_ylim(new_limits)
    ax2.set_ylabel(ylab)
    return ax2


# %% Functions - get_rms_indices
def get_rms_indices(  # noqa: C901
    time_one: _Times | None = None,
    time_two: _Times | None = None,
    time_overlap: _Times | None = None,
    *,
    xmin: _Time = -inf,
    xmax: _Time = inf,
) -> _RmsIndices:
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
    [np.int64(1), np.int64(8)]

    """

    def _process(time: _Times | None, t_bound: _Time | None, func: Callable) -> bool:
        r"""Determines if the given time should be processed."""
        if is_datetime(time):
            # if datetime, it's either the datetime.datetime version, or np.datetime64 version
            if isinstance(time, datetime.datetime):
                # process if any of the data is in the bound
                process = func(time, t_bound)
            else:
                process = not np.isnat(time)  # type: ignore[arg-type]
        else:
            if time is None:
                process = False
            else:
                process = not np.isnan(time) and not np.isinf(time) and func(time, t_bound)  # type: ignore[arg-type]
        return process  # type: ignore[no-any-return]

    # TODO: functionalize this more so there is less repeated code
    # initialize output
    temp: list[int] = []
    ix: _RmsIndices = {
        "pts": temp,
        "one": np.array([], dtype=bool),
        "two": np.array([], dtype=bool),
        "overlap": np.array([], dtype=bool),
    }
    # alias some flags
    have1 = time_one is not None and np.size(time_one) > 0  # type: ignore[arg-type]
    have2 = time_two is not None and np.size(time_two) > 0  # type: ignore[arg-type]
    have3 = time_overlap is not None
    # get the min/max times
    if have1:
        if have2:
            # have both
            t_min = np.minimum(np.min(time_one), np.min(time_two))  # type: ignore[arg-type]
            t_max = np.maximum(np.max(time_one), np.max(time_two))  # type: ignore[arg-type]
        else:
            # have only time 1
            t_min = np.min(time_one)  # type: ignore[arg-type]
            t_max = np.max(time_one)  # type: ignore[arg-type]
    else:
        if have2:
            # have only time 2
            t_min = np.min(time_two)  # type: ignore[arg-type]
            t_max = np.max(time_two)  # type: ignore[arg-type]
        else:
            # have neither time 1 nor time 2
            raise AssertionError("At least one time vector must be given.")
    p1_min: _B
    p2_min: _B
    p3_min: _B
    p1_max: _B
    p2_max: _B
    p3_max: _B
    if _process(xmin, t_max, operator.lt):  # type: ignore[arg-type]
        if have1:
            p1_min = time_one >= xmin  # type: ignore[assignment, call-overload, operator]
        if have2:
            p2_min = time_two >= xmin  # type: ignore[assignment, call-overload, operator]
        if have3:
            p3_min = time_overlap >= xmin  # type: ignore[assignment, call-overload, operator]
        ix["pts"].append(np.maximum(xmin, t_min))  # type: ignore[arg-type]
    else:
        if have1:
            p1_min = np.ones(time_one.shape, dtype=bool)  # type: ignore[union-attr]
        if have2:
            p2_min = np.ones(time_two.shape, dtype=bool)  # type: ignore[union-attr]
        if have3:
            p3_min = np.ones(time_overlap.shape, dtype=bool)  # type: ignore[union-attr]
        ix["pts"].append(t_min)
    if _process(xmax, t_min, operator.gt):  # type: ignore[arg-type]
        if have1:
            p1_max = time_one <= xmax  # type: ignore[assignment, call-overload, operator]
        if have2:
            p2_max = time_two <= xmax  # type: ignore[assignment, call-overload, operator]
        if have3:
            p3_max = time_overlap <= xmax  # type: ignore[assignment, call-overload, operator]
        ix["pts"].append(np.minimum(xmax, t_max))  # type: ignore[arg-type]
    else:
        if have1:
            p1_max = np.ones(time_one.shape, dtype=bool)  # type: ignore[union-attr]
        if have2:
            p2_max = np.ones(time_two.shape, dtype=bool)  # type: ignore[union-attr]
        if have3:
            p3_max = np.ones(time_overlap.shape, dtype=bool)  # type: ignore[union-attr]
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


# %% Functions - plot_vert_lines
def plot_vert_lines(
    ax: Axes,
    x: tuple[_Time, ...] | list[_Time] | _Times,
    *,
    show_in_legend: bool = True,
    colormap: _CM | ColorMap | None = None,
    labels: list[str] | tuple[str, ...] | None = None,
) -> None:
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
    cm = ColorMap(colormap, num_colors=len(x))  # type: ignore[arg-type]
    if labels is None:
        labels = ["RMS Start Time", "RMS Stop Time"]
    # plot vertical lines
    for i, this_x in enumerate(x):  # type: ignore[arg-type]
        this_color = cm.get_color(i)
        this_label = labels[i] if show_in_legend else ""
        ax.axvline(this_x, linestyle="--", color=this_color, marker="+", markeredgecolor="m", markersize=10, label=this_label)


# %% plot_phases
def plot_phases(  # noqa: C901
    ax: Axes,
    times: _D | _N | list[float] | list[np.datetime64] | tuple[_D, _D] | tuple[_N, _N] | tuple[float, float],
    colormap: _CM | ColorMap | None = "tab10",
    labels: list[str] | str | None = None,
    *,
    group_all: bool = False,
    use_legend: bool = False,
    transparency: float = 0.2,  # 1.0 = opaque
) -> None:
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
    >>> _ = ax.plot(time, data, ".-", label="data")
    >>> times = np.array([5, 20, 60, 90])
    >>> # times = np.array([[5, 20, 60, 90], [10, 60, 90, 95]])
    >>> labels = ["Part 1", "Phase 2", "Watch Out", "Final"]
    >>> colors = COLOR_LISTS["quat"]
    >>> plot_phases(ax, times, colors, labels, use_legend=False)
    >>> _ = ax.legend(loc="best")
    >>> plt.show(block=False)  # doctest: +SKIP

    Close plot
    >>> plt.close(fig)

    """
    # get the limits of the plot
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    # get number of segments and list out times
    if isinstance(times, tuple):
        assert len(times) == 2, "Expect exactly two elements in tuple."
        t1 = np.atleast_1d(times[0])
        t2 = np.atleast_1d(times[1])
        assert t1.size == t2.size, "Expecting both time vectors to be the same size."
    elif isinstance(times, list) or times.ndim == 1:
        t1 = np.asanyarray(times)
        t2 = np.hstack((times[1:], max(times[-1], xlims[1])))  # type: ignore[arg-type, type-var]
    elif times.ndim == 2:
        t1 = times[0, :]
        t2 = times[1, :]
    else:
        raise ValueError("Unexpected size for times.")

    # check for optional arguments
    if not group_all:
        cm = ColorMap(colormap=colormap, num_colors=np.size(t1))
    elif colormap == "tab10":
        # change to responsible default for group_all case
        colormap = "xkcd:black"

    # loop through all the phases
    for i, (x1, x2) in enumerate(zip(t1, t2)):
        # get the label and color for this phase
        this_color = cm.get_color(i) if not group_all else colormap
        if labels is not None:
            if group_all:
                assert isinstance(labels, str), "Labels must be a string if grouping all."
                if use_legend:
                    this_label = labels if i == 0 else ""
                else:
                    this_label = labels
            else:
                assert isinstance(labels, list), "Labels must be a list if not grouping all."
                this_label = labels[i]
        # create the shaded box
        ax.axvspan(
            x1,
            x2,
            facecolor=this_color,
            edgecolor=this_color,
            alpha=transparency,
            clip_on=True,
            label=this_label if use_legend else "",
        )
        # create the label
        if labels is not None and not use_legend and bool(this_label):
            xy = (date2num(x1), 0.99) if is_datetime(x1) else (x1, 0.99)
            ax.annotate(
                this_label,
                xy=xy,
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


# %% Functions - get_classification
def get_classification(classify: str) -> tuple[str, str]:
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


# %% Functions - plot_classification
def plot_classification(  # noqa: C901
    ax: Axes, classification: str = "U", *, caveat: str = "", test: bool = False, location: str = "figure"
) -> None:
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
    >>> plt.show(block=False)  # doctest: +SKIP

    >>> fig2 = plt.figure()
    >>> ax2 = fig2.add_subplot(111)
    >>> _ = ax2.plot(0, 0)
    >>> plot_classification(ax2, "S", caveat="//MADE UP CAVEAT", test=True, location="figure")
    >>> plt.show(block=False)  # doctest: +SKIP

    >>> fig3 = plt.figure()
    >>> ax3 = fig3.add_subplot(111)
    >>> _ = ax3.plot(1, 1)
    >>> plot_classification(ax3, "C", test=True, location="axis")
    >>> plt.show(block=False)  # doctest: +SKIP

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
            bbox={"facecolor": "none", "edgecolor": "r"},
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
        bbox={"boxstyle": "square", "facecolor": "none", "edgecolor": color, "linewidth": 2},
    )
    # add border
    fig = ax.figure
    assert fig is not None
    r1 = Rectangle(
        (0.0, 0.0), 1.0, 1.0, facecolor="none", edgecolor=color, clip_on=False, linewidth=3, transform=fig.transFigure
    )
    fig.patches.extend([r1])


# %% Functions - align_plots
def align_plots(fig: _FigOrListFig, pos: tuple[int, int] | None = None) -> None:
    """
    Aligns all the figures in one location.

    Parameters
    ----------
    fig : list or single instance of matplotlib.Figure
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
    # force figs to be a list
    if isinstance(fig, list):
        figs = fig
    else:
        figs = [fig]
    # initialize position if given
    x_pos: int | None = None
    y_pos: int | None = None
    if pos is not None:
        (x_pos, y_pos) = pos
    # loop through figures
    for this_fig in figs:
        # get the manager
        manager = this_fig.canvas.manager
        assert manager is not None
        # use position from first plot if you don't already have it
        if x_pos is None or y_pos is None:
            (x_pos, y_pos, _, _) = manager.window.geometry().getRect()  # type: ignore[attr-defined]
        # move the plot
        manager.window.move(x_pos, y_pos)  # type: ignore[attr-defined]


# %% Functions - z_from_ci
def z_from_ci(ci: int | float) -> float:
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


# %% Functions - ci_from_z
def ci_from_z(z: int | float) -> float:
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


# %% Functions - save_figs_to_pdf
def save_figs_to_pdf(
    figs: Figure | list[Figure] | None = None,
    filename: Path = Path("figs.pdf"),
    *,
    rasterized: bool = False,
) -> None:
    r"""
    Saves the given figures to a PDF file.

    Parameters
    ----------
    figs : figure or list[figure] or None
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
    >>> close_all(fig)

    """
    # Optional inputs
    if figs is None:
        figs = plt.get_fignums()  # type: ignore[assignment]
    if isinstance(figs, Figure):
        figs = [figs]
    assert isinstance(figs, list)
    assert len(figs) > 0, "There must be at least one figure to create a PDF from."

    creation_date = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    # Created PDF (rasterized form)
    if rasterized:
        # build a list of all the rasterized images using IO buffers
        img_bufs = []
        img_list = []
        for fig in figs:
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format="png")
            img_list.append(Image.open(img_buf))
            img_bufs.append(img_buf)
        # save to PDF
        dpi = int(plt.rcParams["figure.dpi"])
        resolution = [int(x * dpi) for x in plt.rcParams["figure.figsize"]]
        img_list[0].save(
            filename,
            "PDF",
            save_all=True,
            append_images=img_list[1:],
            dpi=[dpi, dpi],
            resolution=resolution,
            title="PDF Figures",
            author=get_username(),
            creationDate=creation_date,
            modDate=creation_date,
        )
        for img_buf in img_bufs:
            img_buf.close()
        return

    # Create PDF (vectorized form)
    with PdfPages(filename) as pdf:
        for fig in figs:
            pdf.savefig(fig)

        # Set metadata for PDF file
        d = pdf.infodict()
        d["Title"] = "PDF Figures"
        d["Author"] = get_username()
        d["CreationDate"] = creation_date
        d["ModDate"] = creation_date


# %% Functions - save_images_to_pdf
def save_images_to_pdf(
    figs: Figure | list[Figure] | None = None,
    folder: Path | None = None,
    plot_type: str = "png",
    filename: Path = Path("figs.pdf"),
) -> None:
    r"""
    Uses figure names to find the already saved images and combine them into a PDF file.

    Parameters
    ----------
    figs : figure or list[figure] or None
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
    >>> close_all(fig)

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
    meta: dict[str, str | datetime.datetime] = {}
    meta["Title"] = "PDF Figures"
    meta["Author"] = get_username()
    meta["CreationDate"] = datetime.datetime.now()
    meta["ModDate"] = meta["CreationDate"]

    # Create PDF of images
    images = []
    for ix, fig in enumerate(figs):
        fig_title = resolve_name(get_figure_title(fig))
        this_image = folder.joinpath(fig_title + "." + plot_type)
        image_rgba = Image.open(this_image)
        image_jpg = image_rgba.convert("RGB")
        if ix == 0:
            im = image_jpg
        else:
            images.append(image_jpg)
    if len(images) > 0:
        im.save(filename, save_all=True, append_images=images, metadata=meta)  # TODO: metadata not saving?


# %% add_datashaders
def add_datashaders(
    datashaders: list[dict[str, Any]],
    *,
    threshold: float = 0.8,
    max_px: int = 6,
    how: str = "over",
    zorder: int | None = None,
    alpha: float = 1.0,
) -> None:
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
            shade_hook=partial(tf.dynspread, threshold=threshold, max_px=max_px, how=how),
            zorder=zorder,
            alpha=alpha,
        )


# %% fig_ax_factory
@overload
def fig_ax_factory(
    num_figs: int | None,
    num_axes: int | tuple[int, int],
    *,
    suptitle: str | list[str],
    layout: str,
    sharex: bool,
    passthrough: Literal[False] = ...,
) -> tuple[tuple[Figure, Axes], ...]: ...
@overload
def fig_ax_factory(
    num_figs: int | None,
    num_axes: int | tuple[int, int],
    *,
    suptitle: str | list[str],
    layout: str,
    sharex: bool,
    passthrough: Literal[True],
) -> tuple[None, ...]: ...
def fig_ax_factory(  # noqa: C901
    num_figs: int | None = None,
    num_axes: int | tuple[int, int] = 1,
    *,
    suptitle: str | list[str] = "",
    layout: str = "rows",
    sharex: bool = True,
    passthrough: bool = False,
) -> tuple[tuple[Figure, Axes], ...] | tuple[None, ...]:
    r"""
    Creates the figures and axes for use in a given plotting function.

    Parameters
    ----------
    num_figs : int
        Number of figures to produce
    num_axes : int or (int, int)
        Total number of axes
    suptitle : str
        Title to put over all axes
    layout : str
        Axes layout, from {"rows", "cols", "rowwise", "colwise"}
    sharex : bool
        Whether to share the X axis
    passthrough : bool
        Whether to include everything and return a tuple of None's with the correct length

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
        assert len(num_axes) == 2, "Expected a tuple with exactly two elements."
        num_row, num_col = num_axes
    if num_figs is None:
        num_figs = 1
    if passthrough:
        return tuple(None for _ in range(num_figs * num_row * num_col))
    figs: list[Figure] = []
    axes: list[Axes] | list[list[Axes]] | list[list[list[Axes]]] = []
    for i in range(num_figs):
        (fig, ax) = plt.subplots(num_row, num_col, sharex=sharex)
        if bool(suptitle):
            this_title = suptitle[i] if isinstance(suptitle, list) else suptitle
            fig.suptitle(this_title)
            assert fig.canvas.manager is not None
            fig.canvas.manager.set_window_title(this_title)
        figs.append(fig)
        axes.append(ax)
    fig_ax: tuple[tuple[Figure, Axes], ...]
    if is_1d:
        assert isinstance(num_axes, int)
        if num_axes == 1:
            fig_ax = tuple((figs[f], axes[f]) for f in range(num_figs))  # type: ignore[misc]
        else:
            fig_ax = tuple((figs[f], axes[f][i]) for f in range(num_figs) for i in range(num_axes))  # type: ignore[index, misc]
    else:
        if layout == "rowwise":
            fig_ax = tuple((figs[f], axes[f][i, j]) for f in range(num_figs) for i in range(num_row) for j in range(num_col))  # type: ignore[call-overload, index]
        elif layout == "colwise":
            fig_ax = tuple((figs[f], axes[f][i, j]) for f in range(num_figs) for j in range(num_col) for i in range(num_row))  # type: ignore[call-overload, index]
    return fig_ax


# %% Unit test
if __name__ == "__main__":
    plt.ioff()
    unittest.main(module="dstauffman.tests.test_plotting_support", exit=False)
    doctest.testmod(verbose=False)
