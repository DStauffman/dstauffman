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
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, TypedDict, Union
import unittest

from slog import LogLevel

from dstauffman import (
    convert_date,
    convert_time_units,
    find_in_range,
    Frozen,
    get_unit_conversion,
    HAVE_MPL,
    HAVE_NUMPY,
    histcounts,
    np_digitize,
)
from dstauffman.plotting.generic import make_bar_plot, make_difference_plot, make_time_plot
from dstauffman.plotting.support import (
    ColorMap,
    ExtraPlotter,
    figmenu,
    get_classification,
    ignore_plot_data,
    is_notebook,
    plot_classification,
    plot_second_yunits,
    storefig,
    titleprefix,
)

if HAVE_MPL:
    from matplotlib.axes import Axes
    from matplotlib.collections import PatchCollection
    import matplotlib.colors as colors
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    import matplotlib.pyplot as plt
    import matplotlib.transforms as transforms
if HAVE_NUMPY:
    import numpy as np

    inf = np.inf
    isfinite = np.isfinite
else:
    from math import inf, isfinite  # type: ignore[assignment]

if TYPE_CHECKING:
    from typing_extensions import NotRequired, Unpack

    _D = np.typing.NDArray[np.datetime64]
    _I = np.typing.NDArray[np.int_]
    _N = np.typing.NDArray[np.float64]
    _M = np.typing.NDArray[np.float64]  # 2D
    _CM = Union[str, colors.Colormap, colors.ListedColormap, ColorMap]
    _Data = Union[int, float, _I, _N, _M, List[_I], List[_N], List[Union[_I, _N]], Tuple[_I, ...], Tuple[_N, ...], Tuple[Union[_I, _N], ...]]  # fmt: skip
    _Time = Union[None, int, float, datetime.datetime, datetime.date, np.datetime64, np.int_, np.float64]
    _Times = Union[int, float, datetime.datetime, np.datetime64, _D, _I, _N, List[_N], List[_D], Tuple[_N, ...], Tuple[_D, ...]]
    _DeltaTime = Union[int, float, np.timedelta64]
    _Figs = List[Figure]

    class _OptsKwargs(TypedDict):
        case_name: NotRequired[str]
        date_zero: NotRequired[Optional[datetime.datetime]]
        save_plot: NotRequired[bool]
        save_path: NotRequired[Optional[Path]]
        show_plot: NotRequired[bool]
        show_link: NotRequired[bool]
        plot_type: NotRequired[str]
        show_warn: NotRequired[bool]
        sub_plots: NotRequired[bool]
        sing_line: NotRequired[bool]
        disp_xmin: NotRequired[_Time]
        disp_xmax: NotRequired[_Time]
        rms_xmin: NotRequired[_Time]
        rms_xmax: NotRequired[_Time]
        show_rms: NotRequired[bool]
        use_mean: NotRequired[bool]
        lab_vert: NotRequired[bool]
        show_zero: NotRequired[bool]
        quat_comp: NotRequired[bool]
        show_xtra: NotRequired[bool]
        time_base: NotRequired[str]
        time_unit: NotRequired[str]
        colormap: NotRequired[Optional[_CM]]
        leg_spot: NotRequired[str]
        classify: NotRequired[str]
        names: NotRequired[List[str]]

    class _TimeKwargs(TypedDict):
        name: NotRequired[str]
        save_plot: NotRequired[bool]
        save_path: NotRequired[Optional[Path]]
        elements: NotRequired[Union[None, List[str], Tuple[str, ...]]]
        units: NotRequired[str]
        time_units: NotRequired[str]
        start_date: NotRequired[str]
        rms_xmin: NotRequired[_Time]
        rms_xmax: NotRequired[_Time]
        disp_xmin: NotRequired[_Time]
        disp_xmax: NotRequired[_Time]
        single_lines: NotRequired[bool]
        colormap: NotRequired[Optional[_CM]]
        use_mean: NotRequired[bool]
        plot_zero: NotRequired[bool]
        show_rms: NotRequired[bool]
        legend_loc: NotRequired[str]
        second_units: NotRequired[Union[None, str, int, float, Tuple[str, float]]]
        leg_scale: NotRequired[Union[None, str, int, float, Tuple[str, float]]]
        ylabel: NotRequired[Union[None, str, List[str]]]
        data_as_rows: NotRequired[bool]
        extra_plotter: NotRequired[Optional[ExtraPlotter]]
        use_zoh: NotRequired[bool]
        label_vert_lines: NotRequired[bool]
        use_datashader: NotRequired[bool]
        fig_ax: NotRequired[Optional[Tuple[Figure, Axes]]]
        plot_type: NotRequired[str]
        classify: NotRequired[str]

    class _DiffKwargs(TypedDict):
        name_one: NotRequired[str]
        name_two: NotRequired[str]
        elements: NotRequired[Union[None, List[str], Tuple[str, ...]]]
        units: NotRequired[str]
        save_plot: NotRequired[bool]
        save_path: NotRequired[Optional[Path]]
        time_units: NotRequired[str]
        start_date: NotRequired[str]
        rms_xmin: NotRequired[_Time]
        rms_xmax: NotRequired[_Time]
        disp_xmin: NotRequired[_Time]
        disp_xmax: NotRequired[_Time]
        make_subplots: NotRequired[bool]
        single_lines: NotRequired[bool]
        colormap: NotRequired[Optional[_CM]]
        use_mean: NotRequired[bool]
        label_vert_lines: NotRequired[bool]
        plot_zero: NotRequired[bool]
        show_rms: NotRequired[bool]
        legend_loc: NotRequired[str]
        second_units: NotRequired[Union[None, str, int, float, Tuple[str, float]]]
        show_extra: NotRequired[bool]
        leg_scale: NotRequired[Union[None, str, int, float, Tuple[str, float]]]
        data_as_rows: NotRequired[bool]
        tolerance: NotRequired[_DeltaTime]
        use_zoh: NotRequired[bool]
        extra_plotter: NotRequired[Optional[ExtraPlotter]]
        use_datashader: NotRequired[bool]
        fig_ax: NotRequired[Optional[Tuple[Figure, Axes]]]
        classify: NotRequired[str]

    class _BarKwargs(TypedDict):
        name: NotRequired[str]
        save_plot: NotRequired[bool]
        save_path: NotRequired[Optional[Path]]
        elements: NotRequired[Union[None, List[str], Tuple[str, ...]]]
        units: NotRequired[str]
        time_units: NotRequired[str]
        start_date: NotRequired[str]
        rms_xmin: NotRequired[_Time]
        rms_xmax: NotRequired[_Time]
        disp_xmin: NotRequired[_Time]
        disp_xmax: NotRequired[_Time]
        single_lines: NotRequired[bool]
        colormap: NotRequired[Optional[_CM]]
        use_mean: NotRequired[bool]
        plot_zero: NotRequired[bool]
        show_rms: NotRequired[bool]
        legend_loc: NotRequired[str]
        second_units: NotRequired[Union[None, str, int, float, Tuple[str, float]]]
        ylabel: NotRequired[Union[None, str, List[str]]]
        data_as_rows: NotRequired[bool]
        extra_plotter: NotRequired[Optional[ExtraPlotter]]
        use_zoh: NotRequired[bool]
        label_vert_lines: NotRequired[bool]
        fig_ax: NotRequired[Optional[Tuple[Figure, Axes]]]
        classify: NotRequired[str]

    class _HistKwargs(TypedDict):
        legend_loc: NotRequired[str]


# %% Globals
logger = logging.getLogger(__name__)

_Plotter: bool = True


# %% Classes - Opts
class Opts(Frozen):
    r"""Optional plotting configurations."""

    def __init__(self, *args: Optional[Opts], **kwargs: Unpack[_OptsKwargs]) -> None:
        r"""
        Default configuration for plots.

        .case_name : str
            Name of the case to be plotted
        .date_zero : datetime
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
            Type of plot to save to disk, from {"png","jpg","fig","emf"}
        .show_warn : bool
            Whether to show warning if saving by title instead of window (i.e. no display is found)
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
        .lab_vert  : bool
            Flag for labeling the vertical lines in the legend when showing the RMS
        .show_zero : bool
            Flag for whether to show Y=0 on the plot axis
        .quat_comp : bool
            Flag to plot quaternion component differences or just the angle
        .show_xtra : bool
            Flag to show extra points in one vector or the other when plotting differences
        .time_base : str
            Base units of time, typically from {"sec", "months"}
        .time_unit : str
            Time unit for the x axis, from {"", "sec", "min", "hr", "day", "month", "year"}
        .colormap  : str
            Name of the colormap to use
        .leg_spot  : str
            Location to place the legend, from {"best", "upper right", "upper left",
            "lower left", "lower right", "right", "center left", "center right", "lower center",
            "upper center", "center" or tuple of position}
        .classify  : str
            Classification level to put on plots
        .names     : list of str
            Names of the data structures to be plotted
        """
        # fmt: off
        self.case_name: str   = ""
        self.date_zero: Optional[datetime.datetime] = None
        self.save_plot: bool  = False
        self.save_path: Optional[Path] = None
        self.show_plot: bool  = True
        self.show_link: bool  = False
        self.plot_type: str   = "png"
        self.show_warn: bool  = not is_notebook()
        self.sub_plots: bool  = True
        self.sing_line: bool  = False
        self.disp_xmin: _Time = -inf
        self.disp_xmax: _Time =  inf
        self.rms_xmin: _Time  = -inf
        self.rms_xmax: _Time  =  inf
        self.show_rms: bool   = True
        self.use_mean: bool   = False
        self.lab_vert: bool   = True
        self.show_zero: bool  = False
        self.quat_comp: bool  = True
        self.show_xtra: bool  = True
        self.time_base: str   = "sec"
        self.time_unit: str   = "sec"
        self.colormap:Optional[ Union[str, ColorMap]] = None
        self.leg_spot: str    = "best"
        self.classify: str    = ""
        self.names: List[str] = []
        # fmt: on
        for arg in args:
            if arg is None:
                continue
            if isinstance(arg, self.__class__):
                for key, value in vars(arg).items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                    else:
                        raise ValueError(f'Unexpected option of "{key}" passed to Opts initializer."')
            else:
                raise ValueError("Unexpected input argument receieved.")
        use_datetime = False
        for key, value in kwargs.items():
            if key == "use_datetime":
                use_datetime = value
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f'Unexpected option of "{key}" passed to Opts initializer."')
        if use_datetime:
            self.convert_dates("datetime")

    def __copy__(self) -> "Opts":
        r"""Allows a new copy to be generated with data from the original."""
        new = type(self)(self)
        return new

    def get_names(self, ix: int) -> str:
        r"""Get the specified name from the list."""
        if hasattr(self, "names") and len(self.names) >= ix + 1:
            name = self.names[ix]
        else:
            name = ""
        return name

    def get_date_zero_str(self, date: Optional[Union[datetime.datetime, List[int]]] = None) -> str:
        r"""
        Gets a string representation of date_zero, typically used to print on an X axis.

        Returns
        -------
        start_date : str
            String representing the date of time zero.

        Examples
        --------
        >>> from dstauffman.plotting import Opts
        >>> from datetime import datetime
        >>> opts = Opts()
        >>> opts.date_zero = datetime(2019, 4, 1, 18, 0, 0)
        >>> print(opts.get_date_zero_str())
          t(0) = 01-Apr-2019 18:00:00 Z

        """
        TIMESTR_FORMAT = "%d-%b-%Y %H:%M:%S"
        if date is None:
            if self.date_zero is None:
                start_date: str = ""
            else:
                start_date = "  t(0) = " + self.date_zero.strftime(TIMESTR_FORMAT) + " Z"
        else:
            if isinstance(date, datetime.datetime):
                start_date = "  t(0) = " + date.strftime(TIMESTR_FORMAT) + " Z"
            else:
                temp_date = datetime.datetime(*date)  # type: ignore[arg-type]
                start_date = "  t(0) = " + temp_date.strftime(TIMESTR_FORMAT) + " Z"
        return start_date

    def get_time_limits(self) -> Tuple[_Time, _Time, _Time, _Time]:
        r"""Returns the display and RMS limits in the current time units."""

        def _convert(value: _Time) -> _Time:
            if value is not None and isfinite(value):  # type: ignore[arg-type]
                return convert_time_units(value, self.time_base, self.time_unit)  # type: ignore[arg-type]
            return value

        if self.time_base == "datetime":
            return (self.disp_xmin, self.disp_xmax, self.rms_xmin, self.rms_xmax)

        disp_xmin = _convert(self.disp_xmin)
        disp_xmax = _convert(self.disp_xmax)
        rms_xmin  = _convert(self.rms_xmin)  # fmt: skip
        rms_xmax  = _convert(self.rms_xmax)  # fmt: skip
        return (disp_xmin, disp_xmax, rms_xmin, rms_xmax)

    def convert_dates(self, form: str, old_form: str = "sec", numpy_form: str = "datetime64[ns]") -> "Opts":
        r"""Converts between double and datetime representations."""
        assert form in {"datetime", "numpy", "sec"}, f'Unexpected form of "{form}".'
        self.time_base = form
        self.time_unit = form
        self.disp_xmin = convert_date(  # type: ignore[assignment]
            self.disp_xmin, form=form, date_zero=self.date_zero, old_form=old_form, numpy_form=numpy_form
        )
        self.disp_xmax = convert_date(  # type: ignore[assignment]
            self.disp_xmax, form=form, date_zero=self.date_zero, old_form=old_form, numpy_form=numpy_form
        )
        self.rms_xmin = convert_date(  # type: ignore[assignment]
            self.rms_xmin, form=form, date_zero=self.date_zero, old_form=old_form, numpy_form=numpy_form
        )
        self.rms_xmax = convert_date(  # type: ignore[assignment]
            self.rms_xmax, form=form, date_zero=self.date_zero, old_form=old_form, numpy_form=numpy_form
        )
        return self


# %% Functions - suppress_plots
def suppress_plots() -> None:
    r"""
    Function that allows you to globally suppres the display of any plots generated by the library.

    Notes
    -----
    #.  Modified from a class to a function based version by David C. Stauffer in November 2020.

    Examples
    --------
    >>> from dstauffman.plotting import suppress_plots
    >>> suppress_plots()

    """
    global _Plotter  # pylint: disable=global-statement
    _Plotter = False
    if HAVE_MPL:
        plt.ioff()


# %% Functions - unsuppress_plots
def unsuppress_plots() -> None:
    r"""
    Function that allows you to globally un-suppress the display of any plots so they will be shown again.

    Notes
    -----
    #.  Modified from a class to a function based version by David C. Stauffer in November 2020.

    Examples
    --------
    >>> from dstauffman.plotting import unsuppress_plots
    >>> unsuppress_plots()

    """
    global _Plotter  # pylint: disable=global-statement
    _Plotter = True


# %% Functions - plot_time_history
def plot_time_history(
    description: str,
    time: Optional[_Times],
    data: Optional[_Data],
    opts: Optional[Opts] = None,
    *,
    ignore_empties: bool = False,
    skip_setup_plots: bool = False,
    **kwargs: Unpack[_TimeKwargs],
) -> Optional[Figure]:
    r"""
    Plot multiple metrics over time.

    Parameters
    ----------
    description : str
        Name to label on the plots
    time : 1D ndarray
        time history
    data : 0D, 1D, or 2D ndarray
        data for corresponding time history, time is last dimension unless passing data_as_rows=False through
    opts : class Opts, optional
        plotting options
    ignore_empties : bool, optional
        Removes any entries from the plot and legend that contain only zeros or only NaNs
    skip_setup_plots : bool, optional, default is False
        Whether to skip the setup_plots step, in case you are manually adding to an existing axis
    save_plot : bool, optional
        Ability to overide the option in opts
    kwargs : dict
        Remaining keyword arguments will be passed to make_time_plot

    Returns
    -------
    fig : object
        figure handle, if None, no figure was created

    See Also
    --------
    make_time_plot

    Notes
    -----
    #.  Written by David C. Stauffer in September 2015.
    #.  Updated by David C. Stauffer in October 2017 to do comparsions of multiple runs.

    Examples
    --------
    >>> from dstauffman.plotting import plot_time_history
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> description = "Random Data"
    >>> time = np.arange(0, 5, 1./12) + 2000
    >>> data = np.random.rand(5, len(time)).cumsum(axis=1)
    >>> data = 10 * data / np.expand_dims(data[:, -1], axis=1)
    >>> fig  = plot_time_history(description, time, data)

    Date based version
    >>> time2 = np.datetime64("2020-05-01 00:00:00", "ns") + 10**9*np.arange(0, 5*60, 5, dtype=np.int64)
    >>> fig2 = plot_time_history(description, time2, data, time_units="datetime")

    Close plots
    >>> plt.close(fig)
    >>> plt.close(fig2)

    """
    # check for valid data
    if ignore_plot_data(data, ignore_empties):
        logger.log(LogLevel.L5, " %s plot skipped due to missing data.", description)
        return None

    # make local copy of opts that can be modified without changing the original
    this_opts = Opts() if opts is None else opts.__class__(opts)
    # opts overrides
    this_opts.save_plot = kwargs.pop("save_plot", this_opts.save_plot)
    this_opts.save_path = kwargs.pop("save_path", this_opts.save_path)
    if "classify" in kwargs:
        this_opts.classify = kwargs.pop("classify")

    # alias opts
    # fmt: off
    time_units   = kwargs.pop("time_units", this_opts.time_base)
    start_date   = kwargs.pop("start_date", this_opts.get_date_zero_str())
    rms_xmin     = kwargs.pop("rms_xmin", this_opts.rms_xmin)
    rms_xmax     = kwargs.pop("rms_xmax", this_opts.rms_xmax)
    disp_xmin    = kwargs.pop("disp_xmin", this_opts.disp_xmin)
    disp_xmax    = kwargs.pop("disp_xmax", this_opts.disp_xmax)
    single_lines = kwargs.pop("single_lines", this_opts.sing_line)
    colormap     = kwargs.pop("colormap", this_opts.colormap)
    use_mean     = kwargs.pop("use_mean", this_opts.use_mean)
    lab_vert     = kwargs.pop("label_vert_lines", this_opts.lab_vert)
    plot_zero    = kwargs.pop("plot_zero", this_opts.show_zero)
    show_rms     = kwargs.pop("show_rms", this_opts.show_rms)
    legend_loc   = kwargs.pop("legend_loc", this_opts.leg_spot)
    # fmt: on

    # call wrapper function for most of the details
    fig = make_time_plot(  # type: ignore[misc]
        description,
        time,
        data,
        time_units=time_units,
        start_date=start_date,
        rms_xmin=rms_xmin,
        rms_xmax=rms_xmax,
        disp_xmin=disp_xmin,
        disp_xmax=disp_xmax,
        single_lines=single_lines,
        colormap=colormap,
        use_mean=use_mean,
        label_vert_lines=lab_vert,
        plot_zero=plot_zero,
        show_rms=show_rms,
        legend_loc=legend_loc,
        **kwargs,
    )

    # setup plots
    if not skip_setup_plots:
        setup_plots(fig, this_opts)
    return fig


# %% Functions - plot_time_difference
def plot_time_difference(
    description: str,
    time_one: Optional[_Times],
    data_one: Optional[_Data],
    time_two: Optional[_Times],
    data_two: Optional[_Data],
    opts: Optional[Opts] = None,
    *,
    ignore_empties: bool = False,
    skip_setup_plots: bool = False,
    **kwargs: Unpack[_DiffKwargs],
) -> Union[None, Figure, _Figs, Tuple[_Figs, Dict[str, _N]]]:
    r"""
    Plot multiple metrics over time.

    Parameters
    ----------
    description : str
        Name to label on the plots
    time_one : 1D ndarray
        time history
    data_one : 0D, 1D, or 2D ndarray
        data for corresponding time history, time is last dimension unless passing data_as_rows=False through
    time_two : 1D ndarray
        time history for series two
    data_two : 0D, 1D, or 2D ndarray
        data for corresponding time history, time is last dimension unless passing data_as_rows=False through
    opts : class Opts, optional
        plotting options
    ignore_empties : bool, optional
        Removes any entries from the plot and legend that contain only zeros or only NaNs
    skip_setup_plots : bool, optional, default is False
        Whether to skip the setup_plots step, in case you are manually adding to an existing axis
    save_plot : bool, optional
        Ability to overide the option in opts
    kwargs : dict
        Remaining keyword arguments will be passed to make_time_plot

    Returns
    -------
    fig : object
        figure handle, if None, no figure was created

    See Also
    --------
    make_time_plot

    Notes
    -----
    #.  Written by David C. Stauffer in December 2022.

    Examples
    --------
    >>> from dstauffman.plotting import plot_time_difference
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> description = "Random Data"
    >>> time_one = np.arange(0, 5, 1./12) + 2000
    >>> data_one = np.random.rand(5, len(time_one)).cumsum(axis=1)
    >>> data_one[:] = 10 * data_one / np.expand_dims(data_one[:, -1], axis=1)
    >>> time_two = np.arange(0, 5, 0.5) + 2000
    >>> data_two = np.random.rand(5, len(time_two)).cumsum(axis=1)
    >>> data_two[:] = 10 * data_two / np.expand_dims(data_two[:, -1], axis=1)
    >>> figs1 = plot_time_difference(description, time_one, data_one, time_two, data_two)

    Date based version
    >>> time1 = np.datetime64("2020-05-01 00:00:00", "ns") + 10**9*np.arange(0, 5*60, 5, dtype=np.int64)
    >>> time2 = np.datetime64("2020-05-01 00:00:00", "ns") + 10**9*np.arange(0, 5*60, 30, dtype=np.int64)
    >>> figs2 = plot_time_difference(description, time1, data_one, time2, data_two, time_units="datetime")

    Close plots
    >>> for fig in figs1:
    ...     plt.close(fig)
    >>> for fig in figs2:
    ...     plt.close(fig)

    """
    # check for valid data
    if ignore_plot_data(data_one, ignore_empties) and ignore_plot_data(data_two, ignore_empties):
        logger.log(LogLevel.L5, " %s plot skipped due to missing data.", description)
        return None

    # make local copy of opts that can be modified without changing the original
    this_opts = Opts() if opts is None else opts.__class__(opts)
    # opts overrides
    this_opts.save_plot = kwargs.pop("save_plot", this_opts.save_plot)
    this_opts.save_path = kwargs.pop("save_path", this_opts.save_path)
    if "classify" in kwargs:
        this_opts.classify = kwargs.pop("classify")

    # alias opts
    # fmt: off
    time_units   = kwargs.pop("time_units", this_opts.time_base)
    start_date   = kwargs.pop("start_date", this_opts.get_date_zero_str())
    rms_xmin     = kwargs.pop("rms_xmin", this_opts.rms_xmin)
    rms_xmax     = kwargs.pop("rms_xmax", this_opts.rms_xmax)
    disp_xmin    = kwargs.pop("disp_xmin", this_opts.disp_xmin)
    disp_xmax    = kwargs.pop("disp_xmax", this_opts.disp_xmax)
    single_lines = kwargs.pop("single_lines", this_opts.sing_line)
    colormap     = kwargs.pop("colormap", this_opts.colormap)
    use_mean     = kwargs.pop("use_mean", this_opts.use_mean)
    lab_vert     = kwargs.pop("label_vert_lines", this_opts.lab_vert)
    plot_zero    = kwargs.pop("plot_zero", this_opts.show_zero)
    show_rms     = kwargs.pop("show_rms", this_opts.show_rms)
    legend_loc   = kwargs.pop("legend_loc", this_opts.leg_spot)
    show_extra   = kwargs.pop("show_extra", this_opts.show_xtra)
    # fmt: on

    # call wrapper function for most of the details
    fig = make_difference_plot(  # type: ignore[call-overload, misc]
        description=description,
        time_one=time_one,
        time_two=time_two,
        data_one=data_one,
        data_two=data_two,
        time_units=time_units,
        start_date=start_date,
        rms_xmin=rms_xmin,
        rms_xmax=rms_xmax,
        disp_xmin=disp_xmin,
        disp_xmax=disp_xmax,
        single_lines=single_lines,
        colormap=colormap,
        use_mean=use_mean,
        label_vert_lines=lab_vert,
        plot_zero=plot_zero,
        show_rms=show_rms,
        legend_loc=legend_loc,
        show_extra=show_extra,
        **kwargs,
    )

    # setup plots
    if not skip_setup_plots:
        setup_plots(fig, this_opts)
    return fig  # type: ignore[no-any-return]


# %% Functions - plot_correlation_matrix
def plot_correlation_matrix(
    data: _M,
    labels: Optional[Union[List[str], List[List[str]]]] = None,
    units: str = "",
    *,
    opts: Optional[Opts] = None,
    matrix_name: str = "Correlation Matrix",
    cmin: Union[int, float] = 0,
    cmax: Union[int, float] = 1,
    xlabel: str = "",
    ylabel: str = "",
    plot_lower_only: bool = True,
    label_values: bool = False,
    x_lab_rot: Union[int, float] = 90,
    colormap: Optional[_CM] = None,
    plot_border: Optional[str] = None,
    leg_scale: Optional[Union[str, int, float, Tuple[str, float]]] = "unity",
    fig_ax: Optional[Tuple[Figure, Axes]] = None,
    skip_setup_plots: bool = False,
) -> Figure:
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
    leg_scale : str, optional
        factor to use when scaling the value in the legend, default is "unity"
    fig_ax : (fig, ax) tuple, optional
        Figure and axis to use, otherwise create new ones
    skip_setup_plots : bool, optional, default is False
        Whether to skip the setup_plots step, in case you are manually adding to an existing axis

    Returns
    -------
    fig : object
        figure handle

    Notes
    -----
    #.  Written by David C. Stauffer in July 2015.

    Examples
    --------
    >>> from dstauffman.plotting import plot_correlation_matrix
    >>> from dstauffman import unit
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> data = unit(np.random.rand(10, 10), axis=0)
    >>> labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    >>> units = "m"
    >>> opts = None
    >>> matrix_name = "Correlation Matrix"
    >>> cmin = 0
    >>> cmax = 1
    >>> xlabel = ""
    >>> ylabel = ""
    >>> plot_lower_only = True
    >>> label_values = True
    >>> x_lab_rot = 90
    >>> colormap = None
    >>> plot_border=None
    >>> leg_scale = "centi"
    >>> fig_ax = None
    >>> skip_setup_plots = False
    >>> fig = plot_correlation_matrix(data, labels, units=units, opts=opts, matrix_name=matrix_name, \
    ...     cmin=cmin, cmax=cmax, xlabel=xlabel, ylabel=ylabel, plot_lower_only=plot_lower_only, \
    ...     label_values=label_values, x_lab_rot=x_lab_rot, colormap=colormap, plot_border=plot_border, \
    ...     leg_scale=leg_scale, fig_ax=fig_ax, skip_setup_plots=skip_setup_plots)

    Close plot
    >>> plt.close(fig)

    """
    # check optional inputs
    if opts is None:
        opts = Opts()
    if colormap is None:
        if opts.colormap is None:
            colormap = "cool"
        else:
            colormap = opts.colormap
    (new_units, scale) = get_unit_conversion(leg_scale, units)

    # Hard-coded values
    box_size = 1
    precision = 1e-12

    # get sizes
    (n, m) = data.shape

    # check labels
    if labels is None:
        xlab = [str(i) for i in range(m)]
        ylab = [str(i) for i in range(n)]
    else:
        if isinstance(labels[0], list):
            xlab = labels[0]
            ylab = labels[1]  # type: ignore[assignment]
        else:
            xlab = labels  # type: ignore[assignment]
            ylab = labels  # type: ignore[assignment]
    # check lengths
    if len(xlab) != m or len(ylab) != n:
        raise ValueError("Incorrectly sized labels.")

    # Determine if symmetric
    if m == n and np.all(  # pylint: disable=simplifiable-if-statement
        np.abs(np.subtract(data, np.transpose(data), out=np.zeros(data.shape, dtype=data.dtype), where=~np.isnan(data)))
        < precision
    ):
        is_symmetric = True
    else:
        is_symmetric = False
    plot_lower_only = plot_lower_only and is_symmetric

    # Override color ranges based on data
    # test if in -1 to 1 range instead of 0 to 1
    if np.all(find_in_range(data, min_=-1, max_=0, inclusive=True, precision=precision)) and cmin == 0 and cmax == 1:
        cmin = -1
    # test if outside the cmin to cmax range, and if so, adjust range.
    temp = np.min(data)
    if temp < cmin:
        cmin = temp  # type: ignore[assignment]
    temp = np.max(data)
    if temp > cmax:
        cmax = temp  # type: ignore[assignment]

    # determine which type of data to plot
    this_title = matrix_name + (" [" + new_units + "]" if new_units else "")

    # Create plots
    if fig_ax is None:
        # create figure
        fig = plt.figure()
        # get handle to axes for use later
        ax = fig.add_subplot(1, 1, 1)
    else:
        (fig, ax) = fig_ax
    # set figure title
    assert (manager := fig.canvas.manager) is not None
    if (sup := fig._suptitle) is None:  # type: ignore[attr-defined]  # pylint: disable=protected-access
        manager.set_window_title(matrix_name)
    else:
        manager.set_window_title(sup.get_text())
    # set axis color to none
    ax.patch.set_facecolor("none")
    # set title
    ax.set_title(this_title)
    # get colormap based on high and low limits
    cm = ColorMap(colormap, low=scale * cmin, high=scale * cmax)
    # loop through and plot each element with a corresponding color
    for i in range(m):
        for j in range(n):
            if not plot_lower_only or (i <= j):
                if not np.isnan(data[j, i]):
                    ax.add_patch(
                        Rectangle(
                            (box_size * i, box_size * j),
                            box_size,
                            box_size,
                            facecolor=cm.get_color(scale * data[j, i]),
                            edgecolor=plot_border,
                        )
                    )
                if label_values:
                    ax.annotate(
                        f"{scale * data[j, i]:.2g}",
                        xy=(box_size * i + box_size / 2, box_size * j + box_size / 2),
                        xycoords="data",
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=15,
                    )
    # show colorbar
    fig.colorbar(cm.get_smap(), ax=ax, shrink=0.9)
    # make square
    ax.set_aspect("equal")
    # set limits and tick labels
    ax.set_xlim(0, m)
    ax.set_ylim(0, n)
    ax.set_xticks(np.arange(0, m) + box_size / 2)
    ax.set_xticklabels(xlab, rotation=x_lab_rot)
    ax.set_yticks(np.arange(0, n) + box_size / 2)
    ax.set_yticklabels(ylab)
    # label axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # reverse the y axis
    ax.invert_yaxis()

    # Setup plots
    if not skip_setup_plots:
        setup_plots(fig, opts)
    return fig


# %% Functions - plot_bar_breakdown
def plot_bar_breakdown(
    description: str,
    time: Optional[_Times],
    data: Optional[_Data],
    opts: Optional[Opts] = None,
    *,
    ignore_empties: bool = False,
    skip_setup_plots: bool = False,
    **kwargs: Unpack[_BarKwargs],
) -> Optional[Figure]:
    r"""
    Plot the pie chart like breakdown by percentage in each category over time.

    Parameters
    ----------
    description : str
        Name to label on the plots
    time : array_like
        time history
    data : array_like
        data for corresponding time history, 2D: time by ratio in each category
    opts : class Opts, optional
        plotting options
    ignore_empties : bool, optional
        Removes any entries from the plot and legend that contain only zeros or only NaNs
    skip_setup_plots : bool, optional, default is False
        Whether to skip the setup_plots step, in case you are manually adding to an existing axis

    Returns
    -------
    fig : object
        figure handle

    Notes
    -----
    #.  Written by David C. Stauffer in June 2015.
    #.  Updated by David C. Stauffer in May 2021 to use wrap generic lower level function.

    Examples
    --------
    >>> from dstauffman.plotting import plot_bar_breakdown
    >>> import numpy as np
    >>> description = "Test"
    >>> time = np.arange(0, 5, 1./12) + 2000
    >>> data = np.random.rand(5, len(time))
    >>> mag  = np.sum(data, axis=0)
    >>> data = data / np.expand_dims(mag, axis=0)
    >>> fig  = plot_bar_breakdown(description, time, data)

    Close plot
    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)

    """
    # check for valid data
    if ignore_plot_data(data, ignore_empties):
        logger.log(LogLevel.L5, " %s plot skipped due to missing data.", description)
        return None

    # make local copy of opts that can be modified without changing the original
    this_opts = Opts() if opts is None else opts.__class__(opts)
    # opts overrides
    this_opts.save_plot = kwargs.pop("save_plot", this_opts.save_plot)
    this_opts.save_path = kwargs.pop("save_path", this_opts.save_path)
    if "classify" in kwargs:
        this_opts.classify = kwargs.pop("classify")

    # alias opts
    # fmt: off
    time_units   = kwargs.pop("time_units", this_opts.time_base)
    start_date   = kwargs.pop("start_date", this_opts.get_date_zero_str())
    rms_xmin     = kwargs.pop("rms_xmin", this_opts.rms_xmin)
    rms_xmax     = kwargs.pop("rms_xmax", this_opts.rms_xmax)
    disp_xmin    = kwargs.pop("disp_xmin", this_opts.disp_xmin)
    disp_xmax    = kwargs.pop("disp_xmax", this_opts.disp_xmax)
    single_lines = kwargs.pop("single_lines", this_opts.sing_line)
    colormap     = kwargs.pop("colormap", this_opts.colormap)
    use_mean     = kwargs.pop("use_mean", this_opts.use_mean)
    lab_vert     = kwargs.pop("label_vert_lines", this_opts.lab_vert)
    plot_zero    = kwargs.pop("plot_zero", this_opts.show_zero)
    show_rms     = kwargs.pop("show_rms", this_opts.show_rms)
    legend_loc   = kwargs.pop("legend_loc", this_opts.leg_spot)
    # fmt: on

    # hard-coded values
    scale = 100
    units = "%"

    # call wrapper function for most of the details
    fig = make_bar_plot(  # type: ignore[misc]
        description,
        time,
        scale * data,  # type: ignore[operator]
        units=units,
        time_units=time_units,
        start_date=start_date,
        rms_xmin=rms_xmin,
        rms_xmax=rms_xmax,
        disp_xmin=disp_xmin,
        disp_xmax=disp_xmax,
        single_lines=single_lines,
        colormap=colormap,
        use_mean=use_mean,
        label_vert_lines=lab_vert,
        plot_zero=plot_zero,
        show_rms=show_rms,
        legend_loc=legend_loc,
        **kwargs,
    )

    # Setup plots
    if not skip_setup_plots:
        setup_plots(fig, this_opts)
    return fig


# %% Functions - plot_histogram
def plot_histogram(
    description: str,
    data: _N,
    bins: Union[_N, List[float]],
    *,
    opts: Optional[Opts] = None,
    color: str = "#1f77b4",
    xlabel: str = "Data",
    ylabel: str = "Number",
    second_ylabel: str = "Distribution [%]",
    normalize_spacing: bool = False,
    use_exact_counts: bool = False,
    show_cdf: bool = False,
    cdf_x: Optional[Union[float, List[float]]] = None,
    cdf_y: Optional[Union[float, List[float]]] = None,
    cdf_colormap: Optional[_CM] = None,
    cdf_same_axis: bool = False,
    cdf_round_to_bin: bool = False,
    fig_ax: Optional[Tuple[Figure, Axes]] = None,
    skip_setup_plots: bool = False,
    **kwargs: Unpack[_HistKwargs],
) -> Figure:
    r"""
    Creates a histogram plot of the given data and bins.

    Parameters
    ----------
    description : str
        Name to label on the plots
    data : (N, ) ndarray
        data to bin into buckets
    bins : (A, ) ndarray
        boundaries of the bins to use for the histogram
    opts : class Opts, optional
        plotting options
    color : str or RGB or RGBA code, optional
        Name of color to use
    xlabel : str, optional
        Name to put on x-axis
    ylabel : str, optional
        Name to put on y-axis
    second_ylabel : str, optional
        Name to put on second y-axis
    normalize_spacing : bool, optional, default is False
        Whether to normalize all the bins to the same horizontal size
    use_exact_counts : bool, optional, default is False
        Whether to bin things based only on exactly equal values
    show_cdf : bool, optional, default is False
        Whether to draw the CDF result
    cdf_x : scalar or (B, ) ndarray
        X values to draw lines at CDF
    cdf_y : scalar or (C, ) ndarray
        Y values to draw lines at CDF
    cdf_colormap : str or matplotlib.colors.Colormap, optional
        Colors/colormap to use for CDF lines
    cdf_same_axis : bool, optional, default is False
        Whether to use the same axis for the CDF, or to create a secondary one
    cdF_round_to_bin : bool, optional, default is False
        Whether to round the CDF crossings to bin edges
    fig_ax : (fig, ax) tuple, optional
        Figure and axis to use, otherwise create new ones
    skip_setup_plots : bool, optional, default is False
        Whether to skip the setup_plots step, in case you are manually adding to an existing axis

    Returns
    -------
    fig : class matplotlib.figure.Figure
        Figure handle

    Notes
    -----
    #.  Written by David C. Stauffer in February 2021.

    Examples
    --------
    >>> from dstauffman.plotting import plot_histogram
    >>> import numpy as np
    >>> description = "Histogram"
    >>> data = np.array([0.5, 3.3, 1., 1.5, 1.5, 1.75, 2.5, 2.5])
    >>> bins = np.array([0., 1., 2., 3., 5., 7.])
    >>> fig = plot_histogram(description, data, bins)

    With CDF
    >>> fig2 = plot_histogram(description, data, bins, show_cdf=True, cdf_y=0.5)

    Close plot
    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)
    >>> plt.close(fig2)

    """
    # check optional inputs
    if opts is None:
        opts = Opts()
    legend_loc = kwargs.pop("legend_loc", opts.leg_spot)
    assert not bool(kwargs), f"Unexpected keyword arguments were passed in: {list(kwargs.keys())}."
    if use_exact_counts:
        counts = np.array([np.count_nonzero(data == this_bin) for this_bin in bins], dtype=int)
    else:
        # TODO: optionally allow this to not include 100% of the data by disabling some error
        # checks in np_digitize?
        counts = histcounts(data, bins)
    missing = data.size - np.sum(counts)
    num = np.size(bins)
    if normalize_spacing or use_exact_counts:
        xlab = [str(i) for i in bins]
        if use_exact_counts:
            num += 1
        if missing > 0:
            xlab += ["Unbinned Data"]
        plotting_bins = np.arange(num)
    else:
        plotting_bins = np.asanyarray(bins).copy()
        ix_pinf = np.isinf(plotting_bins) & (np.sign(plotting_bins) > 0)
        ix_ninf = np.isinf(plotting_bins) & (np.sign(plotting_bins) < 0)
        if np.any(ix_pinf):
            plotting_bins[ix_pinf] = np.max(data)
        if np.any(ix_ninf):
            plotting_bins[ix_ninf] = np.min(data)
    rects = []
    for i in range(num - 1):
        rects.append(Rectangle((plotting_bins[i], 0), plotting_bins[i + 1] - plotting_bins[i], counts[i]))
    if missing > 0:
        rects.append(Rectangle((plotting_bins[-1], 0), 1, missing))
    coll = PatchCollection(rects, facecolor=color, edgecolor="k", zorder=6)
    # create plot
    if fig_ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    else:
        (fig, ax) = fig_ax
    assert (manager := fig.canvas.manager) is not None
    if (sup := fig._suptitle) is None:  # type: ignore[attr-defined]  # pylint: disable=protected-access
        manager.set_window_title(description)
    else:
        manager.set_window_title(sup.get_text())
    ax.set_title(description)
    ax.add_collection(coll)
    ax.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if missing > 0:
        ax.set_xlim((np.min(plotting_bins), np.max(plotting_bins) + 1))
    else:
        ax.set_xlim((np.min(plotting_bins), np.max(plotting_bins)))
    if cdf_same_axis:
        ax.set_ylim((0, data.size))
    else:
        ax.set_ylim((0, 1.05 * np.max(counts)))
    if normalize_spacing:
        ax.set_xticks(plotting_bins)
        ax.set_xticklabels(xlab)
    elif use_exact_counts:
        if missing > 0:
            ax.set_xticks(plotting_bins + 0.5)
        else:
            ax.set_xticks(plotting_bins[:-1] + 0.5)
        ax.set_xticklabels(xlab)
    plot_second_yunits(ax, ylab=second_ylabel, multiplier=100 / data.size)
    # Optionally add CDF information
    using_cdf = show_cdf or cdf_x is not None or cdf_y is not None
    if using_cdf:
        # prepare the colormap
        if cdf_colormap is None:
            cdf_colormap = colors.ListedColormap(("xkcd:grass green", "xkcd:red", "xkcd:hot magenta"))
        cm = ColorMap(colormap=cdf_colormap, num_colors=3)
        # create fake items to add to legend
        p = Rectangle((0, 0), 1, 1, facecolor=color, linewidth=0, edgecolor="none")
        # create a transform with X in data units and Y in axes units
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        # create the CDF
        if cdf_round_to_bin:
            cdf = np.hstack([0, np.cumsum(counts)]) / data.size
            cdf_bin = plotting_bins
        else:
            cdf = np.hstack([np.arange(data.size) / data.size, 1.0])
            cdf_bin = np.hstack([0.0, np.sort(data)])
    if show_cdf:
        # plot the CDF
        if not cdf_same_axis:
            ax3 = ax.twinx()
            ax3.set_ylim(0, 100)
            ax3.spines.right.set_position(("axes", 1.06))
            ax3.yaxis.label.set_color(cm.get_color(0))
            ax3.set_ylabel("CDF Distribution [%]")
            ax3.tick_params(axis="y", colors=cm.get_color(0))
        # Note: plot on transformed axes instead of ax3 to maintain constant pan/zoom
        if normalize_spacing:
            temp = np_digitize(cdf_bin, bins)
            cdf_scaled = temp + (cdf_bin - bins[temp]) / (bins[temp + 1] - bins[temp])
            ax.step(cdf_scaled, cdf, color=cm.get_color(0), label="CDF", zorder=8, transform=trans)
        else:
            ax.step(cdf_bin, cdf, color=cm.get_color(0), label="CDF", zorder=8, transform=trans)
    if cdf_x is not None:
        try:
            _ = len(cdf_x)  # type: ignore[arg-type]
        except TypeError:
            cdf_x = [cdf_x]  # type: ignore[list-item]
        for this_x in cdf_x:  # type: ignore[union-attr]
            this_ix = np.argmax(cdf_bin >= this_x)
            this_bin = cdf_scaled[this_ix] if normalize_spacing else cdf_bin[this_ix]
            this_cdf = cdf[this_ix]
            ax.plot(
                [0, 1],
                [this_cdf, this_cdf],
                color=cm.get_color(1),
                label=f"{this_x:.3g}={100*this_cdf:.3g}p",
                zorder=9,
                transform=ax.transAxes,
            )
            ax.plot(
                this_bin,
                this_cdf,
                marker="o",
                markeredgecolor=cm.get_color(1),
                markerfacecolor="none",
                label="",
                zorder=10,
                transform=trans,
            )
    if cdf_y is not None:
        try:
            _ = len(cdf_y)  # type: ignore[arg-type]
        except TypeError:
            cdf_y = [cdf_y]  # type: ignore[list-item]
        for this_cdf in cdf_y:  # type: ignore[union-attr]
            this_ix = np.argmax(cdf >= this_cdf)
            this_label = f"{100*this_cdf:.3g}p={cdf_bin[this_ix]:.3g}"
            this_bin = cdf_scaled[this_ix] if normalize_spacing else cdf_bin[this_ix]
            ax.axvline(this_bin, label=this_label, color=cm.get_color(2), zorder=9)
            ax.plot(this_bin, cdf[this_ix], marker="x", color=cm.get_color(2), label="", zorder=10, transform=trans)
    if using_cdf:
        # Add a legend now, since there is something to display
        (handles, labels) = ax.get_legend_handles_labels()
        handles.insert(0, p)
        labels.insert(0, "PDF")
        ax.legend(handles, labels, loc=legend_loc)
    if not skip_setup_plots:
        setup_plots(fig, opts=opts)
    return fig


# %% Functions - setup_plots
def setup_plots(figs: Union[Figure, _Figs], opts: Opts, *, skip_tight: bool = False) -> None:
    r"""
    Combine common plot operations into one easy command.

    Parameters
    ----------
    figs : array_like
        List of figures
    opts : class Opts
        Optional plotting controls
    plot_type : optional, {"time", "time_no_yscale", "dist", "dist_no_yscale"}

    Notes
    -----
    #.  Written by David C. Stauffer in May 2015.

    Examples
    --------
    >>> from dstauffman.plotting import setup_plots, Opts
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
    >>> opts = Opts()
    >>> opts.case_name = "Testing"
    >>> opts.show_plot = True
    >>> opts.save_plot = False
    >>> setup_plots(fig, opts)

    Close plot
    >>> plt.close(fig)

    """
    # ensure figs is a list
    if not isinstance(figs, list):
        figs = [figs]

    # prepend a title
    if opts.case_name:
        titleprefix(figs, opts.case_name)

    # label plot classification
    (classification, caveat) = get_classification(opts.classify)
    if classification:
        for fig in figs:
            ax = fig.gca()
            plot_classification(ax, classification, caveat=caveat, location="figure")

    # pack the figures
    if not skip_tight:
        bottom = 0.03 if classification else 0.0
        for fig in figs:
            fig.tight_layout(rect=(0.0, bottom, 1.0, 0.97), h_pad=1.5, w_pad=1.5)

    # things to do if displaying the plots
    if opts.show_plot and _Plotter:  # pragma: no cover
        # add a custom toolbar
        figmenu(figs)
        # force drawing right away
        for fig in figs:
            fig.canvas.draw()
            fig.canvas.flush_events()
        # show the plot
        plt.show(block=False)

    # optionally save the plot
    if opts.save_plot:
        storefig(figs, opts.save_path, opts.plot_type, opts.show_warn)
        if opts.show_link & len(figs) > 0:
            print(f'Plots saved to <a href="{opts.save_path}">{opts.save_path}</a>')


# %% save_zoomed_version
def save_zoomed_version(
    fig: Figure,
    ax: Axes,
    ylims: Optional[Tuple[float, float]],
    *,
    ax2: Optional[Axes] = None,
    use_display: bool = True,
    opts: Optional[Opts] = None,
) -> None:
    r"""
    Create and save a zoomed version of the plot to disk.

    """
    if ylims is None:
        return


# %% Unit test
if __name__ == "__main__":
    plt.ioff()
    unittest.main(module="dstauffman.tests.test_plotting_plotting", exit=False)
    doctest.testmod(verbose=False)
